"""Deterministic 3D scenarios (no random number generation)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from atlas.types import Robot, RobotConfig, as_vec3


def antipodal_sphere(
    num_robots: int = 5,
    sphere_radius: float = 3.0,
    center: Sequence[float] = (0.0, 0.0, 4.0),
    config: RobotConfig | None = None,
) -> list[Robot]:
    """Robots on a sphere, each sent to the diametrically opposite point.

    Starts are spread with a Fibonacci lattice. Every straight-line path
    passes through the sphere's centre, so this is a worst case for policies
    that ignore other robots.
    """
    if num_robots < 1:
        raise ValueError("num_robots must be at least 1")
    cfg = config if config is not None else RobotConfig()
    origin = as_vec3(center)
    if origin[2] - sphere_radius <= cfg.radius:
        raise ValueError("the sphere must be entirely above the ground")
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    robots = []
    for i in range(num_robots):
        z = 1.0 - (2.0 * i + 1.0) / num_robots
        ring = np.sqrt(1.0 - z * z)
        theta = golden_angle * i
        offset = sphere_radius * np.array([ring * np.cos(theta), ring * np.sin(theta), z])
        robots.append(
            Robot(name=f"drone_{i}", start=origin + offset, goal=origin - offset, config=cfg)
        )
    return robots


def parallel_lanes(
    num_robots: int = 5,
    length: float = 6.0,
    spacing: float = 1.0,
    altitude: float = 2.0,
    config: RobotConfig | None = None,
) -> list[Robot]:
    """Robots flying along +x in parallel lanes, staggered in altitude.

    Paths never cross, so straight-line flight is collision-free as long as
    ``spacing`` comfortably exceeds the robot diameter.
    """
    if num_robots < 1:
        raise ValueError("num_robots must be at least 1")
    cfg = config if config is not None else RobotConfig()
    robots = []
    for i in range(num_robots):
        y = spacing * i
        z = altitude + 0.5 * spacing * (i % 2)
        robots.append(
            Robot(
                name=f"drone_{i}",
                start=np.array([0.0, y, z]),
                goal=np.array([length, y, z]),
                config=cfg,
            )
        )
    return robots


def head_on(
    separation: float = 6.0,
    altitude: float = 2.0,
    config: RobotConfig | None = None,
) -> list[Robot]:
    """Two robots swapping places along the x-axis: a guaranteed head-on conflict."""
    cfg = config if config is not None else RobotConfig()
    left = np.array([-0.5 * separation, 0.0, altitude])
    right = np.array([0.5 * separation, 0.0, altitude])
    return [
        Robot(name="drone_0", start=left, goal=right, config=cfg),
        Robot(name="drone_1", start=right, goal=left, config=cfg),
    ]
