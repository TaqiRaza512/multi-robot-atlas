# Multi-Robot Atlas

A deterministic 3D simulator and (eventually) benchmark for multi-drone
navigation in shared airspace, built on [MuJoCo](https://mujoco.org).

**Status: milestone M0.** The simulator, a quadrotor autopilot, a naive
straight-line baseline and collision *detection* are in place.
**Collision avoidance is not implemented yet**: drones that are sent through
each other will collide, and the simulator reports it.

## What is here

- **Quadrotor dynamics in MuJoCo.** Each drone is a free rigid body driven by
  collective thrust and three body torques.
- **Geometric SE(3) autopilot** (500 Hz) that tracks velocity setpoints.
- **Policy interface** (50 Hz): world state in, one velocity setpoint per
  drone out. `StraightLinePolicy` flies each drone directly at its goal and
  ignores every other drone.
- **Collision detection** from MuJoCo contacts between the drones' bounding
  spheres, plus the minimum clearance over the run.
- **Deterministic scenarios**: `antipodal_sphere`, `parallel_lanes`, `head_on`.
- Live viewer and offscreen video rendering.

## Quick start

Requires [uv](https://docs.astral.sh/uv/). Works natively on Apple Silicon
and on Linux.

```bash
git clone https://github.com/TaqiRaza512/multi-robot-atlas.git
cd multi-robot-atlas
uv sync                                   # creates .venv with Python 3.12
uv run examples/five_drones.py
```

Five drones start on a sphere and each flies to the opposite side. All paths
cross at the centre, so the naive baseline fails:

```text
simulated 30.00 s
  drone_0: DID NOT reach goal (final error 3.298 m)
  ...
minimum clearance between drones: -0.019 m
drone-drone collisions: 14 contact(s), pairs [(0, 1), (0, 2), (0, 3), (1, 3), (1, 4), (2, 4), (3, 4)]
  t =  1.666 s  drone_1 x drone_4  lasted  0.068 s  penetration 18.2 mm
  ...
```

The drones collide at the centre at about 1.67 s and then stay jammed against
each other until the time limit. Fixing this is the point of the next
milestones.

### Watching it

```bash
# Live 3D viewer. On macOS it must be launched with mjpython:
uv run mjpython examples/five_drones.py --view

# Render to a video file:
uv sync --extra video
uv run examples/five_drones.py --video swap.mp4
```

## Using the library

```python
from atlas import Simulator, StraightLinePolicy, parallel_lanes

sim = Simulator(parallel_lanes(num_robots=5), StraightLinePolicy())
result = sim.run()

result.all_reached_goal  # True
result.collision_free  # True
result.min_clearance  # smallest gap between any two drones [m]
result.positions  # (T, N, 3) trajectory
```

A policy is any callable `WorldState -> (N, 3)` array of velocity setpoints.

## Model and limitations

- Drones are commanded through collective thrust and body torques, the
  standard abstraction of four rotors. Individual rotor dynamics, motor lag,
  aerodynamic drag and downwash are not modelled.
- The autopilot has perfect state knowledge; there is no sensor noise.
- A drone's collision shape is a single sphere.
- Collisions are resolved physically by MuJoCo (drones bounce); there is no
  damage model.
- Determinism holds for the same machine and MuJoCo build. Results are not
  guaranteed to be bit-identical across platforms.

## Development

```bash
uv run pytest
uv run ruff check . && uv run ruff format --check .
uv run mypy
```

## Roadmap

| Milestone | Deliverable |
| --- | --- |
| **M0 (this)** | 3D MuJoCo simulator, autopilot, straight-line baseline, collision detection, tests, CI |
| M1 | Scenario suite and visualization |
| M2 | ORCA collision-avoidance baseline (3D) |
| M3 | MPC navigation controller |
| M4 | Runtime safety assurance |
| M5 | Benchmark suite and reproducible experiments |
| M6 | ROS 2 / Gazebo integration |
