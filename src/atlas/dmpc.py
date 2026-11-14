"""Distributed model predictive control (DMPC) for collision avoidance.

Every drone solves its own small quadratic program each replanning period:

* **Model.** The drone under its autopilot behaves like a first-order velocity
  tracker, ``dv/dt = gain * (u - v)``, where ``u`` is the velocity setpoint.
  The plan is a sequence of ``horizon`` setpoints, each held for ``step`` seconds.
* **Limits.** Setpoint speed and commanded acceleration stay inside the
  robot's limits (norm balls approximated from the inside by polytopes).
* **Collision avoidance.** For every neighbour and every step of the horizon,
  the two drones' *previously announced* predicted positions define a
  separating plane through their midpoint. Each drone constrains its new plan
  to stay at least half the safety distance on its own side. Because both
  drones use the same plane, any two plans that satisfy their constraints are
  at least the full safety distance apart at that step.
* **Symmetry breaking.** The plane's normal is rotated by a fixed angle in a
  common sense for every pair, so drones meeting head-on sidestep the same way
  and pass with more room, rather than relying on numerical asymmetry to
  resolve a perfectly symmetric conflict.

The constraints are soft (heavily penalised slack) so the program always has a
solution; a non-zero slack means the safety distance could not be met at the
prediction level. Decisions use only a drone's own state and its neighbours'
announced plans, so the scheme is decentralised even though this class
computes all drones in one process.

Guarantees are at the level of the planning model, at the sampled steps. The
physical drone tracks the plan imperfectly, which the safety margin absorbs.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import osqp
from scipy import sparse

from atlas.types import FloatArray, Robot, WorldState


@dataclass(frozen=True)
class DMPCConfig:
    """Planner parameters.

    Attributes:
        horizon: Number of steps in the plan.
        step: Duration of one step, and the replanning period [s].
        velocity_gain: Velocity-loop gain of the autopilot [1/s]. Must match
            :attr:`atlas.control.ControlGains.velocity`.
        safety_margin: Extra separation kept on top of the two collision radii [m].
        sidestep_angle: Rotation of the separating planes that breaks head-on
            symmetry [rad]. Zero disables it.
        goal_weight: Weight on squared distance to the goal along the horizon.
        terminal_weight: Additional weight on the final step's distance to the goal.
        effort_weight: Weight on squared setpoints.
        smoothness_weight: Weight on squared setpoint changes between steps.
        slack_weight: Penalty ``w * (s + s^2 / 2)`` on violating a separating plane by ``s`` metres.
        limit_usage: Fraction of the robot's speed and acceleration limits the
            planner allows itself, leaving headroom for tracking error.
    """

    horizon: int = 15
    step: float = 0.1
    velocity_gain: float = 4.0
    safety_margin: float = 0.15
    sidestep_angle: float = 0.5
    goal_weight: float = 1.0
    terminal_weight: float = 10.0
    effort_weight: float = 0.01
    smoothness_weight: float = 0.5
    slack_weight: float = 1.0e3
    limit_usage: float = 0.9

    def __post_init__(self) -> None:
        if self.horizon < 1:
            raise ValueError("horizon must be at least 1")
        if not self.step > 0:
            raise ValueError("step must be positive")
        if not self.velocity_gain > 0:
            raise ValueError("velocity_gain must be positive")
        if self.safety_margin < 0:
            raise ValueError("safety_margin must be non-negative")
        if not 0 <= self.sidestep_angle < np.pi / 2:
            raise ValueError("sidestep_angle must be in [0, pi/2)")
        if not 0 < self.limit_usage <= 1:
            raise ValueError("limit_usage must be in (0, 1]")


def _ball_directions() -> tuple[FloatArray, float]:
    """Unit directions ``D`` and a scale ``c`` such that ``D @ x <= c * r`` for
    every row implies ``|x| <= r`` (an inner polytope approximation of a ball)."""
    grid = np.array(
        [
            [x, y, z]
            for x in (-1, 0, 1)
            for y in (-1, 0, 1)
            for z in (-1, 0, 1)
            if (x, y, z) != (0,) * 3
        ],
        dtype=np.float64,
    )
    directions = grid / np.linalg.norm(grid, axis=1, keepdims=True)
    # The polytope's farthest points are its vertices; bound them by the worst
    # alignment between any direction on the sphere and its nearest row.
    index = np.arange(4000) + 0.5
    polar = np.arccos(1 - 2 * index / 4000)
    azimuth = np.pi * (1 + np.sqrt(5.0)) * index
    samples = np.stack(
        [np.sin(polar) * np.cos(azimuth), np.sin(polar) * np.sin(azimuth), np.cos(polar)], axis=1
    )
    worst_alignment = float((samples @ directions.T).max(axis=1).min())
    return directions, 0.98 * worst_alignment


class DMPCPolicy:
    """Decentralised MPC navigation policy with collision avoidance.

    Implements the :class:`atlas.policies.Policy` protocol. The policy is
    stateful (it remembers every drone's announced plan); the simulator calls
    :meth:`reset` at the start of each run.

    Attributes:
        plans: Latest announced plans, positions of shape ``(N, horizon + 1, 3)``.
        max_slack: Largest separating-plane violation any drone has had to
            accept in a plan since the last reset [m]. Zero means every plan
            met the safety distance.
        solver_failures: Number of times a drone's program failed to solve and
            the drone fell back to braking.
    """

    def __init__(self, robots: Sequence[Robot], config: DMPCConfig | None = None) -> None:
        if not robots:
            raise ValueError("at least one robot is required")
        self.config = config if config is not None else DMPCConfig()
        cfg = self.config
        self._n = len(robots)
        self._radii = np.array([r.config.radius for r in robots])
        self._max_speeds = np.array([r.config.max_speed for r in robots])
        self._max_accels = np.array([r.config.max_acceleration for r in robots])

        horizon, n_u = cfg.horizon, 3 * cfg.horizon

        # Exact discretisation of dp/dt = v, dv/dt = gain * (u - v) per axis.
        decay = float(np.exp(-cfg.velocity_gain * cfg.step))
        c1 = (1 - decay) / cfg.velocity_gain
        a_mat = np.array([[1.0, c1], [0.0, decay]])
        b_vec = np.array([cfg.step - c1, 1 - decay])

        # state_k = sx[k] @ state_0 + su[k] @ inputs, for k = 0..horizon (per axis).
        sx = np.zeros((horizon + 1, 2, 2))
        su = np.zeros((horizon + 1, 2, horizon))
        sx[0] = np.eye(2)
        for k in range(horizon):
            sx[k + 1] = a_mat @ sx[k]
            su[k + 1] = a_mat @ su[k]
            su[k + 1, :, k] += b_vec
        eye3 = np.eye(3)
        # Positions after each step (k = 1..horizon) and velocities at the start
        # of each step (k = 0..horizon-1), as affine maps of (p0, v0, inputs).
        self._pos_p0 = np.kron(sx[1:, 0, 0:1], eye3)
        self._pos_v0 = np.kron(sx[1:, 0, 1:2], eye3)
        self._pos_u = np.kron(su[1:, 0, :], eye3)
        vel_v0 = np.kron(sx[:-1, 1, 1:2], eye3)
        vel_u = np.kron(su[:-1, 1, :], eye3)

        # Cost: goal tracking, effort and smoothness.
        weights = np.full(horizon, cfg.goal_weight)
        weights[-1] += cfg.terminal_weight
        self._goal_weights = np.repeat(weights, 3)
        diff = np.eye(n_u) - np.eye(n_u, k=-3)
        hessian_u = 2 * (
            self._pos_u.T @ (self._goal_weights[:, None] * self._pos_u)
            + cfg.effort_weight * np.eye(n_u)
            + cfg.smoothness_weight * diff.T @ diff
        )
        self._n_slack = (self._n - 1) * horizon
        self._hessian = sparse.block_diag(
            [hessian_u, cfg.slack_weight * sparse.identity(self._n_slack)], format="csc"
        )

        # Limits: speed of each setpoint, and acceleration gain * (u_k - v_k).
        directions, scale = _ball_directions()
        self._limit_scale = scale * cfg.limit_usage
        speed_rows = np.kron(np.eye(horizon), directions)
        accel_map = cfg.velocity_gain * (np.eye(n_u) - vel_u)
        self._limit_rows = np.vstack([speed_rows, speed_rows @ accel_map])
        self._accel_v0 = cfg.velocity_gain * speed_rows @ vel_v0
        self._n_dirs = directions.shape[0]

        # Constraint matrix with a fixed sparsity pattern, so each solver is
        # factorised once and warm-started afterwards. Row blocks: limits,
        # separating planes (dense in the inputs, plus a slack), slack >= 0.
        n_limit, n_slack = self._limit_rows.shape[0], self._n_slack
        n_vars = n_u + n_slack
        limit_r, limit_c = np.nonzero(self._limit_rows)
        plane_r, plane_c = np.divmod(np.arange(n_slack * n_u), n_u)
        slack = np.arange(n_slack)
        rows = np.concatenate(
            [limit_r, n_limit + plane_r, n_limit + slack, n_limit + n_slack + slack]
        )
        cols = np.concatenate([limit_c, plane_c, n_u + slack, n_u + slack])
        data = np.concatenate(
            [self._limit_rows[limit_r, limit_c], np.zeros(n_slack * n_u), np.ones(2 * n_slack)]
        )
        # Index into the flattened plane coefficients, or -1 for constant entries.
        source = np.full(rows.size, -1, dtype=np.int64)
        source[limit_r.size : limit_r.size + n_slack * n_u] = np.arange(n_slack * n_u)

        order = np.lexsort((rows, cols))  # CSC order: by column, then row
        indptr = np.concatenate([[0], np.cumsum(np.bincount(cols, minlength=n_vars))])
        self._constraints = sparse.csc_matrix(
            (data[order], rows[order], indptr), shape=(n_limit + 2 * n_slack, n_vars)
        )
        source = source[order]
        self._plane_slots = np.flatnonzero(source >= 0)
        self._plane_source = source[self._plane_slots]
        self._n_limit = n_limit

        self._plans = np.zeros((self._n, horizon + 1, 3))
        self._setpoints = np.zeros((self._n, 3))
        self._next_replan: float | None = None
        self._solvers: list[osqp.OSQP | None] = [None] * self._n
        self.max_slack = 0.0
        self.solver_failures = 0

    @property
    def plans(self) -> FloatArray:
        return self._plans.copy()

    def reset(self) -> None:
        """Forget all announced plans."""
        self._next_replan = None
        self._setpoints = np.zeros((self._n, 3))
        # Warm-start state would make a second run differ from the first.
        self._solvers = [None] * self._n
        self.max_slack = 0.0
        self.solver_failures = 0

    def __call__(self, world: WorldState) -> FloatArray:
        if world.num_robots != self._n:
            raise ValueError(f"policy was built for {self._n} robots, got {world.num_robots}")
        if self._next_replan is None:
            # Nothing announced yet: every drone is assumed to stay where it is.
            self._plans = np.repeat(world.positions[:, None, :], self.config.horizon + 1, axis=1)
            self._replan(world)
            self._next_replan = world.time + self.config.step
        elif world.time >= self._next_replan - 1e-9:
            # Plans were announced one step ago: shift them forward in time.
            self._plans = np.concatenate([self._plans[:, 1:], self._plans[:, -1:]], axis=1)
            self._replan(world)
            self._next_replan += self.config.step
        return self._setpoints.copy()

    # -- planning ------------------------------------------------------------

    def _separating_planes(self, i: int, others: FloatArray) -> tuple[FloatArray, FloatArray]:
        """Plane normals ``(J, K, 3)`` and offsets ``(J, K)`` for drone ``i``:
        its plan must satisfy ``normal . position >= offset`` at every step."""
        announced = self._plans
        mine = announced[i, 1:][None]
        theirs = announced[others, 1:]
        delta = mine - theirs
        distance = np.linalg.norm(delta, axis=2)

        # Coincident predictions have no direction; separate along x by index.
        fallback = np.where(i < others, -1.0, 1.0)[:, None, None] * np.array([1.0, 0.0, 0.0])
        safe = np.maximum(distance, 1e-9)[:, :, None]
        normal = np.where(distance[:, :, None] > 1e-6, delta / safe, fallback)

        required = (self._radii[i] + self._radii[others] + self.config.safety_margin)[:, None]

        # Rotate each normal about a shared axis. The rotation is the same
        # linear map for both drones of a pair, so their planes still coincide.
        # It is capped so the announced plans stay on the correct sides.
        angle = np.minimum(
            self.config.sidestep_angle, np.arccos(np.clip(required / safe[:, :, 0], 0.0, 1.0))
        )[:, :, None]
        vertical = np.abs(normal[:, :, 2:3]) > 0.9
        axis = np.where(vertical, np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0]))
        tangent = np.cross(axis, normal)
        tangent /= np.linalg.norm(tangent, axis=2, keepdims=True)
        normal = np.cos(angle) * normal + np.sin(angle) * tangent

        midpoint = 0.5 * (mine + theirs)
        offset = 0.5 * required + np.einsum("jkc,jkc->jk", normal, midpoint)
        return normal, offset

    def _solve(self, i: int, world: WorldState) -> FloatArray | None:
        cfg = self.config
        horizon, n_u, n_slack = cfg.horizon, 3 * cfg.horizon, self._n_slack
        p0, v0, goal = world.positions[i], world.velocities[i], world.goals[i]
        free_positions = self._pos_p0 @ p0 + self._pos_v0 @ v0

        # Cost vector.
        goal_error = free_positions - np.tile(goal, horizon)
        linear_u = 2 * self._pos_u.T @ (self._goal_weights * goal_error)
        linear_u[:3] -= 2 * cfg.smoothness_weight * self._setpoints[i]
        linear = np.concatenate([linear_u, np.full(n_slack, cfg.slack_weight)])

        # Bounds: limits, then separating planes, then slack >= 0.
        half = self._n_limit // 2
        lower = np.full(self._n_limit + 2 * n_slack, -np.inf)
        upper = np.full(self._n_limit + 2 * n_slack, np.inf)
        upper[:half] = self._limit_scale * self._max_speeds[i]
        upper[half : self._n_limit] = self._limit_scale * self._max_accels[i] + self._accel_v0 @ v0
        lower[self._n_limit + n_slack :] = 0.0

        values = self._constraints.data.copy()
        if n_slack:
            others = np.array([j for j in range(self._n) if j != i])
            normal, offset = self._separating_planes(i, others)
            pos_u = self._pos_u.reshape(horizon, 3, n_u)
            rows = np.einsum("jkc,kcu->jku", normal, pos_u).reshape(n_slack * n_u)
            values[self._plane_slots] = rows[self._plane_source]
            free = free_positions.reshape(horizon, 3)
            lower[self._n_limit : self._n_limit + n_slack] = (
                offset - np.einsum("jkc,kc->jk", normal, free)
            ).reshape(n_slack)

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
                eps_abs=1e-4,
                eps_rel=1e-4,
                max_iter=4000,
                polishing=False,
                # A fixed interval: the default adapts on wall-clock time,
                # which would make runs non-reproducible.
                adaptive_rho_interval=25,
            )
            self._solvers[i] = solver
        else:
            solver.update(q=linear, l=lower, u=upper, Ax=values)
        solution = solver.solve(raise_error=False)
        if solution.info.status_val not in (1, 2) or solution.x is None:
            return None
        if n_slack:
            self.max_slack = max(self.max_slack, float(solution.x[n_u:].max()))
        inputs: FloatArray = solution.x[:n_u]
        return inputs

    def _replan(self, world: WorldState) -> None:
        """Every drone plans against the announced plans; then all announce together."""
        horizon = self.config.horizon
        new_plans = np.empty_like(self._plans)
        for i in range(self._n):
            p0, v0 = world.positions[i], world.velocities[i]
            inputs = self._solve(i, world)
            if inputs is None:
                # Could not plan: command a stop.
                self.solver_failures += 1
                inputs = np.zeros(3 * horizon)
            positions = self._pos_p0 @ p0 + self._pos_v0 @ v0 + self._pos_u @ inputs
            new_plans[i, 0] = p0
            new_plans[i, 1:] = positions.reshape(horizon, 3)
            self._setpoints[i] = inputs[:3]
        self._plans = new_plans
