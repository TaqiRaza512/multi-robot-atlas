"""Navigation policies: world state to a velocity setpoint per robot.

This is the seam where collision-avoidance planners will plug in. The
simulator limits every setpoint to the robot's ``max_speed`` and hands it to
the autopilot in :mod:`atlas.control`.
"""

from __future__ import annotations

from typing import Protocol

from atlas.control import clip_norm
from atlas.types import FloatArray, WorldState


class Policy(Protocol):
    def __call__(self, world: WorldState) -> FloatArray:
        """Return world-frame velocity setpoints of shape ``(N, 3)``."""
        ...


class StraightLinePolicy:
    """Fly every robot straight at its goal, slowing down on approach.

    This policy ignores all other robots. It performs **no collision
    avoidance** and is the naive baseline that later planners are measured
    against.

    Args:
        gain: Position error to speed [1/s]. The robot cruises at ``max_speed``
            until it is within ``max_speed / gain`` of the goal.
    """

    def __init__(self, gain: float = 1.5) -> None:
        if not gain > 0:
            raise ValueError(f"gain must be positive, got {gain}")
        self.gain = gain

    def __call__(self, world: WorldState) -> FloatArray:
        return clip_norm(self.gain * (world.goals - world.positions), world.max_speeds)
