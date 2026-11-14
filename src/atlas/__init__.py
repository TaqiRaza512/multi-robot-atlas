"""Multi-Robot Atlas: deterministic 3D multi-drone simulation on MuJoCo."""

from atlas.control import ControlGains, tracking_wrench
from atlas.model import build_mjcf, build_model
from atlas.policies import Policy, StraightLinePolicy
from atlas.scenarios import antipodal_sphere, head_on, parallel_lanes
from atlas.simulator import CollisionEvent, SimulationResult, Simulator
from atlas.types import Robot, RobotConfig, SimConfig, WorldState

__all__ = [
    "CollisionEvent",
    "ControlGains",
    "Policy",
    "Robot",
    "RobotConfig",
    "SimConfig",
    "SimulationResult",
    "Simulator",
    "StraightLinePolicy",
    "WorldState",
    "antipodal_sphere",
    "build_mjcf",
    "build_model",
    "head_on",
    "parallel_lanes",
    "tracking_wrench",
]

__version__ = "0.1.0"
