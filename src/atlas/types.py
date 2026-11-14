"""Typed robot configuration, simulation configuration and world state.

All geometry is 3D and in SI units. The world frame is z-up. Per-robot
vectors have shape ``(3,)``; team arrays have shape ``(N, 3)``.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]

_NAME_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")


def as_vec3(value: Sequence[float] | FloatArray) -> FloatArray:
    """Return ``value`` as a read-only float64 vector of shape ``(3,)``."""
    vec = np.array(value, dtype=np.float64)
    if vec.shape != (3,):
        raise ValueError(f"expected a 3D vector, got shape {vec.shape}")
    if not np.all(np.isfinite(vec)):
        raise ValueError("vector components must be finite")
    vec.setflags(write=False)
    return vec


def _require_positive(name: str, value: float) -> None:
    if not (math.isfinite(value) and value > 0):
        raise ValueError(f"{name} must be positive and finite, got {value}")


@dataclass(frozen=True)
class RobotConfig:
    """Physical parameters of one quadrotor.

    Attributes:
        radius: Radius of the collision sphere enclosing the airframe [m].
        mass: Total mass [kg].
        inertia: Principal moments of inertia about the body x, y, z axes [kg m^2].
        max_speed: Largest speed a policy may command [m/s].
        max_acceleration: Largest acceleration the velocity loop may request,
            excluding gravity compensation [m/s^2]. Must be below gravity so
            the commanded thrust direction always points upward.
        thrust_to_weight: Maximum collective thrust as a multiple of weight.
        max_torque: Maximum body torque about each axis [N m].
    """

    radius: float = 0.18
    mass: float = 0.5
    inertia: tuple[float, float, float] = (2.5e-3, 2.5e-3, 4.5e-3)
    max_speed: float = 2.0
    max_acceleration: float = 5.0
    thrust_to_weight: float = 2.5
    max_torque: float = 0.5

    def __post_init__(self) -> None:
        _require_positive("radius", self.radius)
        _require_positive("mass", self.mass)
        _require_positive("max_speed", self.max_speed)
        _require_positive("max_acceleration", self.max_acceleration)
        _require_positive("max_torque", self.max_torque)
        if len(self.inertia) != 3:
            raise ValueError("inertia must have three components")
        for component in self.inertia:
            _require_positive("inertia component", component)
        if not (math.isfinite(self.thrust_to_weight) and self.thrust_to_weight > 1.0):
            raise ValueError(
                f"thrust_to_weight must exceed 1 to hover, got {self.thrust_to_weight}"
            )


@dataclass(frozen=True)
class Robot:
    """A quadrotor with a navigation task: fly from ``start`` to ``goal``.

    ``name`` becomes the MuJoCo body name, so it must be a unique identifier
    made of letters, digits and underscores.
    """

    name: str
    start: FloatArray
    goal: FloatArray
    config: RobotConfig = field(default_factory=RobotConfig)

    def __post_init__(self) -> None:
        if not _NAME_PATTERN.match(self.name):
            raise ValueError(f"invalid robot name {self.name!r}")
        object.__setattr__(self, "start", as_vec3(self.start))
        object.__setattr__(self, "goal", as_vec3(self.goal))


@dataclass(frozen=True)
class SimConfig:
    """Simulation parameters.

    Attributes:
        timestep: Physics step; the attitude controller runs at this rate [s].
        policy_decimation: Physics steps per policy update. The default gives a
            500 Hz autopilot under a 50 Hz policy.
        max_time: Simulated time after which a run stops regardless [s].
        goal_tolerance: Position error below which a robot counts as arrived [m].
        arrival_speed: Speed below which a robot counts as arrived [m/s].
        gravity: Gravitational acceleration along -z [m/s^2].
    """

    timestep: float = 0.002
    policy_decimation: int = 10
    max_time: float = 30.0
    goal_tolerance: float = 0.05
    arrival_speed: float = 0.05
    gravity: float = 9.81

    def __post_init__(self) -> None:
        _require_positive("timestep", self.timestep)
        _require_positive("goal_tolerance", self.goal_tolerance)
        _require_positive("arrival_speed", self.arrival_speed)
        _require_positive("gravity", self.gravity)
        if self.policy_decimation < 1:
            raise ValueError(f"policy_decimation must be >= 1, got {self.policy_decimation}")
        if not (math.isfinite(self.max_time) and self.max_time >= 0):
            raise ValueError(f"max_time must be non-negative and finite, got {self.max_time}")

    @property
    def policy_period(self) -> float:
        """Time between policy updates [s]."""
        return self.timestep * self.policy_decimation


@dataclass(frozen=True)
class WorldState:
    """Snapshot of the whole team, as seen by a policy.

    Attributes:
        time: Simulation time [s].
        positions: Centres of mass in the world frame, shape ``(N, 3)`` [m].
        velocities: Linear velocities in the world frame, shape ``(N, 3)`` [m/s].
        rotations: Body-to-world rotation matrices, shape ``(N, 3, 3)``.
        angular_velocities: Angular velocities in the body frame, shape ``(N, 3)`` [rad/s].
        goals: Goal positions, shape ``(N, 3)`` [m].
        radii: Collision-sphere radii, shape ``(N,)`` [m].
        max_speeds: Speed limits, shape ``(N,)`` [m/s].
    """

    time: float
    positions: FloatArray
    velocities: FloatArray
    rotations: FloatArray
    angular_velocities: FloatArray
    goals: FloatArray
    radii: FloatArray
    max_speeds: FloatArray

    @property
    def num_robots(self) -> int:
        return int(self.positions.shape[0])
