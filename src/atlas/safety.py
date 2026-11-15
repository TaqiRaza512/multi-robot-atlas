"""Runtime safety filter based on control barrier functions (CBFs).

:class:`SafetyFilter` wraps any navigation policy. Each tick it takes the
policy's velocity setpoints and, for every drone, checks whether the
acceleration that setpoint would cause keeps the drone safe with respect to
each neighbour. If it does, the setpoint passes through **unchanged**. If not,
the drone solves a small QP for the closest acceleration that is safe.

The barrier for a pair of drones follows Wang, Ames and Egerstedt, "Safety
barrier certificates for collisions-free multirobot systems" (T-RO 2017)::

    h = sqrt(2 * (b_i + b_j) * (distance - D)) + d(distance)/dt

``h >= 0`` means the pair can still stop before coming closer than ``D`` if
both brake with their assumed braking accelerations ``b_i`` and ``b_j``. The
filter enforces ``dh/dt >= -gain * h``, which is linear in the two
accelerations, and splits the obligation between the drones in proportion to
their braking authority. Each drone therefore needs only its neighbours'
positions and velocities: the filter is decentralised.

What this is not: a planner. It keeps drones apart; it does not get them past
each other, so a policy with no avoidance of its own may be stopped short of
its goal. The barrier also assumes accelerations are applied instantly, while
the real drone has to tilt first; the safety margin absorbs that lag.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import osqp
from scipy import sparse

from atlas.control import clip_norm
from atlas.dmpc import _ball_directions
from atlas.policies import Policy
from atlas.types import FloatArray, Robot, WorldState


@dataclass(frozen=True)
class SafetyFilterConfig:
    """Safety filter parameters.

    Attributes:
        safety_margin: Separation kept on top of the two collision radii [m].
        braking_fraction: Fraction of each robot's ``max_acceleration`` the
            barrier assumes is available for braking. Lower is more cautious.
        barrier_gain: How fast the barrier value may decay toward zero [1/s].
            Lower makes drones start yielding earlier and more gently.
        velocity_gain: Velocity-loop gain of the autopilot [1/s]. Must match
            :attr:`atlas.control.ControlGains.velocity`.
        slack_weight: Penalty on violating a barrier constraint when no
            acceleration within the limits satisfies all of them.
    """

    safety_margin: float = 0.15
    braking_fraction: float = 0.6
    barrier_gain: float = 3.0
    velocity_gain: float = 4.0
    slack_weight: float = 1.0e3

    def __post_init__(self) -> None:
        if self.safety_margin < 0:
            raise ValueError("safety_margin must be non-negative")
        if not 0 < self.braking_fraction <= 1:
            raise ValueError("braking_fraction must be in (0, 1]")
        if not self.barrier_gain > 0:
            raise ValueError("barrier_gain must be positive")
        if not self.velocity_gain > 0:
            raise ValueError("velocity_gain must be positive")


class SafetyFilter:
    """Wraps a policy and minimally corrects setpoints that would be unsafe.

    Implements the :class:`atlas.policies.Policy` protocol.

    Attributes:
        interventions: Number of (drone, tick) pairs in which the filter
            changed the wrapped policy's setpoint since the last reset.
        ticks: Number of policy ticks since the last reset.
        max_slack: Largest barrier-constraint violation the filter had to
            accept [m/s^2]. Zero means every correction was fully safe
            according to the barrier.
        solver_failures: Number of QPs that failed; the drone was told to stop.
    """

    def __init__(
        self,
        policy: Policy,
        robots: Sequence[Robot],
        config: SafetyFilterConfig | None = None,
    ) -> None:
        if not robots:
            raise ValueError("at least one robot is required")
        self.policy = policy
        self.config = config if config is not None else SafetyFilterConfig()
        cfg = self.config
        n = self._n = len(robots)
        self._max_speeds = np.array([r.config.max_speed for r in robots])
        self._max_accels = np.array([r.config.max_acceleration for r in robots])
        self._braking = cfg.braking_fraction * self._max_accels
        radii = np.array([r.config.radius for r in robots])
        self._min_distance = radii[:, None] + radii[None, :] + cfg.safety_margin
        self._pair_braking = self._braking[:, None] + self._braking[None, :]
        self._off_diagonal = ~np.eye(n, dtype=bool)

        # QP per drone over [acceleration (3), slack per neighbour (n - 1)].
        # Row blocks: barriers (changing normals), slack >= 0, acceleration
        # limit, speed limit. Built in CSC order by hand so the barrier
        # coefficients keep explicit entries even when they are zero.
        self._directions, self._limit_scale = _ball_directions()
        m, n_dirs = n - 1, self._directions.shape[0]
        dir_r, dir_c = np.nonzero(self._directions)
        dir_v = self._directions[dir_r, dir_c]
        neighbour = np.arange(m)
        rows = np.concatenate(
            [
                np.repeat(neighbour, 3),
                neighbour,
                m + neighbour,
                2 * m + dir_r,
                2 * m + n_dirs + dir_r,
            ]
        )
        cols = np.concatenate(
            [np.tile(np.arange(3), m), 3 + neighbour, 3 + neighbour, dir_c, dir_c]
        )
        # The slack is stored scaled by sqrt(slack_weight), which makes the
        # QP's Hessian the identity and keeps the solver well conditioned.
        self._slack_scale = 1.0 / np.sqrt(cfg.slack_weight)
        data = np.concatenate(
            [np.zeros(3 * m), np.full(m, self._slack_scale), np.ones(m), dir_v, dir_v]
        )
        source = np.full(rows.size, -1, dtype=np.int64)
        source[: 3 * m] = np.arange(3 * m)
        order = np.lexsort((rows, cols))
        indptr = np.concatenate([[0], np.cumsum(np.bincount(cols, minlength=3 + m))])
        self._constraints = sparse.csc_matrix(
            (data[order], rows[order], indptr), shape=(2 * m + 2 * n_dirs, 3 + m)
        )
        source = source[order]
        self._normal_slots = np.flatnonzero(source >= 0)
        self._normal_source = source[self._normal_slots]
        self._hessian = sparse.identity(3 + m, format="csc")

        self._solvers: list[osqp.OSQP | None] = [None] * n
        self.interventions = 0
        self.ticks = 0
        self.max_slack = 0.0
        self.solver_failures = 0

    def reset(self) -> None:
        """Clear statistics and solver state, and reset the wrapped policy."""
        reset_policy = getattr(self.policy, "reset", None)
        if callable(reset_policy):
            reset_policy()
        self._solvers = [None] * self._n
        self.interventions = 0
        self.ticks = 0
        self.max_slack = 0.0
        self.solver_failures = 0

    @property
    def intervention_rate(self) -> float:
        """Fraction of (drone, tick) pairs in which the filter intervened."""
        return self.interventions / max(1, self.ticks * self._n)

    def barrier_constraints(self, world: WorldState) -> tuple[FloatArray, FloatArray]:
        """Constraints ``normals[i, j] . a_i >= bounds[i, j]`` on drone ``i``'s
        acceleration with respect to each drone ``j``.

        Returns arrays of shape ``(N, N, 3)`` and ``(N, N)``. Diagonal entries
        are vacuous (zero normal, ``-inf`` bound).
        """
        cfg = self.config
        delta_p = world.positions[:, None, :] - world.positions[None, :, :]
        delta_v = world.velocities[:, None, :] - world.velocities[None, :, :]
        distance = np.linalg.norm(delta_p, axis=2)
        safe_distance = np.where(self._off_diagonal, np.maximum(distance, 1e-9), 1.0)
        normals = delta_p / safe_distance[:, :, None]
        range_rate = np.einsum("ijc,ijc->ij", normals, delta_v)
        gap = distance - self._min_distance

        outside = gap > 1e-6
        root = np.sqrt(2 * self._pair_braking * np.where(outside, gap, 1.0))
        barrier = root + range_rate
        tangential = np.einsum("ijc,ijc->ij", delta_v, delta_v) - range_rate**2
        # dh/dt + gain * h >= 0  <=>  normal . (a_i - a_j) >= -slack_term
        slack_term = (
            cfg.barrier_gain * barrier
            + self._pair_braking * range_rate / root
            + tangential / safe_distance
        )
        share = self._braking[:, None] / self._pair_braking
        # Inside the minimum distance the barrier is undefined: push apart
        # with the full assumed braking acceleration instead.
        bounds = np.where(outside, -share * slack_term, self._braking[:, None])
        # A drone cannot be asked for more than its assumed braking, and any
        # bound below -max_acceleration holds automatically. Clamping to that
        # range changes nothing physically and keeps the QP well scaled.
        bounds = np.clip(bounds, -2 * self._max_accels[:, None], self._braking[:, None])
        bounds = np.where(self._off_diagonal, bounds, -np.inf)
        return normals, bounds

    def __call__(self, world: WorldState) -> FloatArray:
        if world.num_robots != self._n:
            raise ValueError(f"filter was built for {self._n} robots, got {world.num_robots}")
        gain = self.config.velocity_gain
        nominal = clip_norm(np.asarray(self.policy(world), dtype=np.float64), self._max_speeds)
        self.ticks += 1
        if self._n == 1:
            return nominal

        # The acceleration the autopilot would command for the nominal setpoint.
        accel = clip_norm(gain * (nominal - world.velocities), self._max_accels)
        normals, bounds = self.barrier_constraints(world)
        unsafe = (np.einsum("ijc,ic->ij", normals, accel) < bounds - 1e-9).any(axis=1)

        filtered = nominal.copy()
        for i in np.flatnonzero(unsafe):
            others = np.flatnonzero(self._off_diagonal[i])
            safe_accel = self._solve(
                int(i), accel[i], world.velocities[i], normals[i, others], bounds[i, others]
            )
            filtered[i] = world.velocities[i] + safe_accel / gain
            self.interventions += 1
        return filtered

    def _solve(
        self,
        i: int,
        nominal_accel: FloatArray,
        velocity: FloatArray,
        normals: FloatArray,
        bounds: FloatArray,
    ) -> FloatArray:
        cfg = self.config
        m, n_dirs = self._n - 1, self._directions.shape[0]
        linear = np.concatenate([-nominal_accel, np.full(m, cfg.slack_weight * self._slack_scale)])

        lower = np.full(2 * m + 2 * n_dirs, -np.inf)
        upper = np.full(2 * m + 2 * n_dirs, np.inf)
        lower[:m] = bounds
        lower[m : 2 * m] = 0.0
        upper[2 * m : 2 * m + n_dirs] = self._limit_scale * self._max_accels[i]
        # Keep the resulting setpoint v + a / gain within the speed limit.
        upper[2 * m + n_dirs :] = cfg.velocity_gain * (
            self._limit_scale * self._max_speeds[i] - self._directions @ velocity
        )

        values = self._constraints.data.copy()
        values[self._normal_slots] = normals.reshape(3 * m)[self._normal_source]

        solver = self._solvers[i]
        if solver is None:
            matrix = self._constraints.copy()
            matrix.data = values
            solver = osqp.OSQP()
            solver.setup(
                self._hessian,
                linear,
                matrix,
                lower,
                upper,
                verbose=False,
                eps_abs=1e-5,
                eps_rel=1e-5,
                max_iter=20000,
                polishing=False,
                adaptive_rho_interval=25,  # fixed, so runs are reproducible
            )
            self._solvers[i] = solver
        else:
            solver.update(q=linear, l=lower, u=upper, Ax=values)
        solution = solver.solve(raise_error=False)
        if solution.info.status_val not in (1, 2) or solution.x is None:
            # No usable answer: command a stop (setpoint zero).
            self.solver_failures += 1
            stop: FloatArray = -cfg.velocity_gain * velocity
            return stop
        self.max_slack = max(self.max_slack, self._slack_scale * float(solution.x[3:].max()))
        safe: FloatArray = solution.x[:3]
        return safe
