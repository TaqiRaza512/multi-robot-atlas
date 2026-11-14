"""Deterministic multi-drone simulator on MuJoCo.

The simulator contains no randomness: the same robots, policy and
configuration give bit-identical trajectories on the same machine and MuJoCo
build.

Collisions are physical (MuJoCo resolves the contact, so drones bounce off
each other) and every contact is *recorded*. Nothing here avoids them.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import mujoco
import numpy as np

from atlas.control import ControlGains, clip_norm, quat_to_mat, tracking_wrench
from atlas.model import ACTUATORS_PER_ROBOT, build_model
from atlas.policies import Policy
from atlas.types import BoolArray, FloatArray, Robot, SimConfig, WorldState

_ContactKey = tuple[int, int | None]


@dataclass(frozen=True)
class CollisionEvent:
    """One contact episode, from first touch until the bodies separate.

    Attributes:
        time: Time at which the contact began [s].
        duration: Time for which the bodies stayed in contact [s].
        robot_a: Index of the robot (the lower index for a robot pair).
        robot_b: Index of the other robot, or ``None`` for the ground.
        max_penetration: Deepest overlap during the episode [m].
    """

    time: float
    duration: float
    robot_a: int
    robot_b: int | None
    max_penetration: float


@dataclass(frozen=True)
class SimulationResult:
    """Outcome of one run, sampled at the policy rate.

    Attributes:
        times: Sample times, shape ``(T,)`` [s].
        positions: World-frame positions, shape ``(T, N, 3)`` [m].
        quaternions: Orientations as ``w, x, y, z``, shape ``(T, N, 4)``.
        velocities: World-frame linear velocities, shape ``(T, N, 3)`` [m/s].
        reached_goal: Whether each robot was at rest at its goal at the end, shape ``(N,)``.
        collisions: Contact episodes in order of start time.
        min_clearance: Smallest surface-to-surface gap between any two robots'
            collision spheres over the run, checked every physics step [m].
            Negative if robots overlapped; ``inf`` for a single robot.
    """

    times: FloatArray
    positions: FloatArray
    quaternions: FloatArray
    velocities: FloatArray
    reached_goal: BoolArray
    collisions: tuple[CollisionEvent, ...]
    min_clearance: float

    @property
    def duration(self) -> float:
        return float(self.times[-1])

    @property
    def all_reached_goal(self) -> bool:
        return bool(self.reached_goal.all())

    @property
    def robot_collisions(self) -> tuple[CollisionEvent, ...]:
        """Contact episodes between two robots."""
        return tuple(e for e in self.collisions if e.robot_b is not None)

    @property
    def ground_collisions(self) -> tuple[CollisionEvent, ...]:
        """Contact episodes between a robot and the ground."""
        return tuple(e for e in self.collisions if e.robot_b is None)

    @property
    def collision_free(self) -> bool:
        return not self.collisions

    @property
    def colliding_pairs(self) -> list[tuple[int, int]]:
        """Distinct robot pairs that touched at any point, sorted."""
        return sorted({(e.robot_a, e.robot_b) for e in self.collisions if e.robot_b is not None})


class Simulator:
    """Simulates a team of quadrotors under one shared navigation policy.

    Use :meth:`run` for a complete headless run, or :meth:`reset` and
    :meth:`step` to drive the loop yourself (for example to sync a viewer).
    ``model`` and ``data`` are the underlying MuJoCo objects.
    """

    def __init__(
        self,
        robots: Sequence[Robot],
        policy: Policy,
        config: SimConfig | None = None,
        gains: ControlGains | None = None,
    ) -> None:
        self.robots = tuple(robots)
        self.policy = policy
        self.config = config if config is not None else SimConfig()
        self.gains = gains if gains is not None else ControlGains()

        for robot in self.robots:
            if robot.config.max_acceleration >= self.config.gravity:
                raise ValueError(
                    f"{robot.name}: max_acceleration must be below gravity ({self.config.gravity})"
                )

        self.model = build_model(self.robots, self.config)
        self.data = mujoco.MjData(self.model)

        n = len(self.robots)
        configs = [robot.config for robot in self.robots]
        self._goals = np.stack([robot.goal for robot in self.robots])
        self._radii = np.array([c.radius for c in configs])
        self._max_speeds = np.array([c.max_speed for c in configs])
        self._max_accels = np.array([c.max_acceleration for c in configs])
        self._masses = np.array([c.mass for c in configs])
        self._inertias = np.array([c.inertia for c in configs])

        self._body_to_robot = np.full(self.model.nbody, -1, dtype=np.int64)
        for index, robot in enumerate(self.robots):
            self._body_to_robot[self.model.body(robot.name).id] = index
        self._pair_a, self._pair_b = np.triu_indices(n, k=1)
        self._pair_radii = self._radii[self._pair_a] + self._radii[self._pair_b]

        self._active: dict[_ContactKey, list[float]] = {}
        self._events: list[CollisionEvent] = []
        self._min_clearance = float("inf")
        self._times: list[float] = []
        self._qpos: list[FloatArray] = []
        self._vels: list[FloatArray] = []
        self.reset()

    # -- state ---------------------------------------------------------------

    @property
    def num_robots(self) -> int:
        return len(self.robots)

    @property
    def time(self) -> float:
        return float(self.data.time)

    def _poses(self) -> FloatArray:
        poses: FloatArray = self.data.qpos.reshape(self.num_robots, 7)
        return poses

    def _twists(self) -> FloatArray:
        twists: FloatArray = self.data.qvel.reshape(self.num_robots, 6)
        return twists

    def world_state(self) -> WorldState:
        """Snapshot of the current state (arrays are copies)."""
        poses, twists = self._poses(), self._twists()
        return WorldState(
            time=self.time,
            positions=poses[:, :3].copy(),
            velocities=twists[:, :3].copy(),
            rotations=quat_to_mat(poses[:, 3:]),
            angular_velocities=twists[:, 3:].copy(),
            goals=self._goals,
            radii=self._radii,
            max_speeds=self._max_speeds,
        )

    def arrived(self) -> BoolArray:
        """Which robots are within the goal tolerance and nearly at rest."""
        poses, twists = self._poses(), self._twists()
        error = np.linalg.norm(self._goals - poses[:, :3], axis=1)
        speed = np.linalg.norm(twists[:, :3], axis=1)
        result: BoolArray = (error <= self.config.goal_tolerance) & (
            speed <= self.config.arrival_speed
        )
        return result

    @property
    def done(self) -> bool:
        """True once every robot has arrived or ``max_time`` has elapsed."""
        timed_out = self.time >= self.config.max_time - 0.5 * self.config.timestep
        return timed_out or bool(self.arrived().all())

    # -- stepping ------------------------------------------------------------

    def reset(self) -> None:
        """Return every robot to its start, at rest, and clear all records."""
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        self._active = {}
        self._events = []
        self._min_clearance = float("inf")
        self._times, self._qpos, self._vels = [], [], []
        self._scan_contacts(0.0)
        self._update_clearance()
        self._record()

    def step(self) -> None:
        """Advance one policy period: query the policy once, then run the autopilot
        and physics for ``policy_decimation`` steps."""
        n = self.num_robots
        setpoint = np.asarray(self.policy(self.world_state()), dtype=np.float64)
        if setpoint.shape != (n, 3):
            raise ValueError(f"policy returned shape {setpoint.shape}, expected {(n, 3)}")
        if not np.all(np.isfinite(setpoint)):
            raise ValueError("policy returned a non-finite velocity")
        setpoint = clip_norm(setpoint, self._max_speeds)

        for _ in range(self.config.policy_decimation):
            poses, twists = self._poses(), self._twists()
            wrench = tracking_wrench(
                setpoint,
                twists[:, :3],
                quat_to_mat(poses[:, 3:]),
                twists[:, 3:],
                self._masses,
                self._inertias,
                self._max_accels,
                self.gains,
                self.config.gravity,
            )
            self.data.ctrl[:] = wrench.reshape(n * ACTUATORS_PER_ROBOT)
            step_start = self.time
            mujoco.mj_step(self.model, self.data)
            # mj_step detects contacts for the state at the start of the step.
            self._scan_contacts(step_start)
            self._update_clearance()
        self._record()

    def run(self) -> SimulationResult:
        """Reset, simulate until :attr:`done`, and return the result."""
        self.reset()
        while not self.done:
            self.step()
        return self.result()

    def result(self) -> SimulationResult:
        """Result of everything simulated since the last reset."""
        events = list(self._events)
        events.extend(self._finish(key, episode) for key, episode in self._active.items())
        events.sort(key=lambda e: (e.time, e.robot_a, -1 if e.robot_b is None else e.robot_b))
        qpos = np.stack(self._qpos)
        return SimulationResult(
            times=np.array(self._times),
            positions=qpos[:, :, :3].copy(),
            quaternions=qpos[:, :, 3:].copy(),
            velocities=np.stack(self._vels),
            reached_goal=self.arrived(),
            collisions=tuple(events),
            min_clearance=self._min_clearance,
        )

    # -- bookkeeping ---------------------------------------------------------

    def _record(self) -> None:
        self._times.append(self.time)
        self._qpos.append(self._poses().copy())
        self._vels.append(self._twists()[:, :3].copy())

    def _update_clearance(self) -> None:
        if self._pair_a.size == 0:
            return
        positions = self._poses()[:, :3]
        gaps = (
            np.linalg.norm(positions[self._pair_a] - positions[self._pair_b], axis=1)
            - self._pair_radii
        )
        self._min_clearance = min(self._min_clearance, float(gaps.min()))

    def _finish(self, key: _ContactKey, episode: list[float]) -> CollisionEvent:
        start, last, penetration = episode
        return CollisionEvent(
            time=start,
            duration=last - start + self.config.timestep,
            robot_a=key[0],
            robot_b=key[1],
            max_penetration=penetration,
        )

    def _scan_contacts(self, time: float) -> None:
        """Fold the contacts currently in ``data`` into the contact episodes."""
        current: dict[_ContactKey, float] = {}
        count = int(self.data.ncon)
        if count:
            contacts = self.data.contact
            owners = self._body_to_robot[self.model.geom_bodyid[contacts.geom[:count]]]
            for (a, b), dist in zip(owners.tolist(), contacts.dist[:count].tolist(), strict=True):
                if dist >= 0 or (a < 0 and b < 0):
                    continue
                key: _ContactKey = (max(a, b), None) if a < 0 or b < 0 else (min(a, b), max(a, b))
                current[key] = max(current.get(key, 0.0), -dist)

        for key, penetration in current.items():
            episode = self._active.get(key)
            if episode is None:
                self._active[key] = [time, time, penetration]
            else:
                episode[1] = time
                episode[2] = max(episode[2], penetration)
        for key in [k for k in self._active if k not in current]:
            self._events.append(self._finish(key, self._active.pop(key)))
