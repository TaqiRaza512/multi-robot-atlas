"""Builds the MuJoCo model for a team of quadrotors.

Each robot is a free-floating rigid body with:

* an explicit inertial (mass and principal inertia from :class:`RobotConfig`),
* one collision sphere of ``config.radius`` (geom group 3, hidden by default),
* purely visual airframe geoms that never collide,
* four actuators on a site at the centre of mass: collective thrust along the
  body z-axis and a torque about each body axis.

The thrust-plus-body-torque interface is the standard abstraction of a
quadrotor's four rotors. Individual rotor dynamics and aerodynamic drag are
not modelled.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import mujoco

from atlas.types import Robot, SimConfig

ACTUATORS_PER_ROBOT = 4

_PALETTE = (
    (0.12, 0.47, 0.71),
    (1.00, 0.50, 0.05),
    (0.17, 0.63, 0.17),
    (0.84, 0.15, 0.16),
    (0.58, 0.40, 0.74),
    (0.55, 0.34, 0.29),
    (0.89, 0.47, 0.76),
    (0.09, 0.75, 0.81),
)


def _vec(values: Iterable[float]) -> str:
    return " ".join(repr(float(v)) for v in values)


def _robot_body(robot: Robot, color: tuple[float, float, float]) -> str:
    name, cfg, r = robot.name, robot.config, robot.config.radius
    rgb = _vec(color)
    arm = 0.68 * r
    offset = arm * 0.7071067811865476
    rotors = "\n".join(
        f'      <geom class="visual" type="cylinder" size="{0.3 * r!r} 0.004" '
        f'pos="{sx * offset!r} {sy * offset!r} {0.07 * r!r}" rgba="0.1 0.1 0.1 0.6"/>'
        for sx, sy in ((1, 1), (1, -1), (-1, 1), (-1, -1))
    )
    return f"""
    <body name="{name}" pos="{_vec(robot.start)}">
      <freejoint name="{name}_free"/>
      <inertial pos="0 0 0" mass="{cfg.mass!r}" diaginertia="{_vec(cfg.inertia)}"/>
      <site name="{name}_com" size="0.005" rgba="0 0 0 0"/>
      <geom name="{name}_collision" type="sphere" size="{r!r}" group="3" rgba="{rgb} 0.2"/>
      <geom class="visual" type="ellipsoid" size="{0.3 * r!r} {0.2 * r!r} {0.1 * r!r}"
            rgba="{rgb} 1"/>
      <geom class="visual" type="box" size="{arm!r} {0.04 * r!r} {0.03 * r!r}"
            euler="0 0 0.7853981633974483" rgba="{rgb} 1"/>
      <geom class="visual" type="box" size="{arm!r} {0.04 * r!r} {0.03 * r!r}"
            euler="0 0 -0.7853981633974483" rgba="{rgb} 1"/>
{rotors}
    </body>"""


def _robot_actuators(robot: Robot, gravity: float) -> str:
    name, cfg = robot.name, robot.config
    max_thrust = cfg.thrust_to_weight * cfg.mass * gravity
    limit = f"{-cfg.max_torque!r} {cfg.max_torque!r}"
    site = f"{name}_com"
    return f"""
    <motor name="{name}_thrust" site="{site}" gear="0 0 1 0 0 0" ctrlrange="0 {max_thrust!r}"/>
    <motor name="{name}_roll" site="{site}" gear="0 0 0 1 0 0" ctrlrange="{limit}"/>
    <motor name="{name}_pitch" site="{site}" gear="0 0 0 0 1 0" ctrlrange="{limit}"/>
    <motor name="{name}_yaw" site="{site}" gear="0 0 0 0 0 1" ctrlrange="{limit}"/>"""


def build_mjcf(robots: Sequence[Robot], config: SimConfig | None = None) -> str:
    """Return the MJCF XML describing the ground plane and all robots.

    Bodies and actuators are emitted in the order of ``robots``, so robot ``i``
    owns ``qpos[7*i : 7*i+7]``, ``qvel[6*i : 6*i+6]`` and ``ctrl[4*i : 4*i+4]``.
    """
    if not robots:
        raise ValueError("at least one robot is required")
    names = [robot.name for robot in robots]
    if len(set(names)) != len(names):
        raise ValueError("robot names must be unique")
    cfg = config if config is not None else SimConfig()

    colors = [_PALETTE[i % len(_PALETTE)] for i in range(len(robots))]
    goals = "".join(
        f'\n    <site name="{robot.name}_goal" type="sphere" size="{0.35 * robot.config.radius!r}" '
        f'pos="{_vec(robot.goal)}" rgba="{_vec(color)} 0.45"/>'
        for robot, color in zip(robots, colors, strict=True)
    )
    bodies = "".join(_robot_body(robot, color) for robot, color in zip(robots, colors, strict=True))
    actuators = "".join(_robot_actuators(robot, cfg.gravity) for robot in robots)

    return f"""<mujoco model="multi_robot_atlas">
  <compiler angle="radian"/>
  <option timestep="{cfg.timestep!r}" gravity="0 0 {-cfg.gravity!r}"/>
  <visual>
    <global offwidth="1280" offheight="720"/>
    <headlight ambient="0.4 0.4 0.4" diffuse="0.6 0.6 0.6" specular="0 0 0"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.75 0.85 0.95" rgb2="0.3 0.45 0.65"
             width="256" height="256"/>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.85 0.85 0.85" rgb2="0.75 0.75 0.75"
             width="256" height="256"/>
    <material name="grid" texture="grid" texrepeat="1 1" texuniform="true" reflectance="0.05"/>
  </asset>
  <default>
    <default class="visual">
      <geom contype="0" conaffinity="0" group="1"/>
    </default>
  </default>
  <worldbody>
    <light pos="0 0 20" dir="0 0 -1" directional="true"/>
    <geom name="ground" type="plane" size="0 0 0.05" material="grid"/>{goals}{bodies}
  </worldbody>
  <actuator>{actuators}
  </actuator>
</mujoco>
"""


def build_model(robots: Sequence[Robot], config: SimConfig | None = None) -> mujoco.MjModel:
    """Compile the MuJoCo model for ``robots``."""
    return mujoco.MjModel.from_xml_string(build_mjcf(robots, config))
