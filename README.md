# Multi-Robot Atlas

A deterministic 3D simulator and (eventually) benchmark for multi-drone
navigation in shared airspace, built on [MuJoCo](https://mujoco.org).

**Status: milestone M1.** The simulator, a quadrotor autopilot and a
distributed MPC planner with collision avoidance are in place, alongside a
naive straight-line baseline that shows what happens without it.

## What is here

- **Quadrotor dynamics in MuJoCo.** Each drone is a free rigid body driven by
  collective thrust and three body torques.
- **Geometric SE(3) autopilot** (500 Hz) that tracks velocity setpoints.
- **Policy interface** (50 Hz): world state in, one velocity setpoint per
  drone out.
- **`DMPCPolicy`**: distributed model predictive control. Each drone solves
  its own small QP (OSQP) every 0.1 s over a 1.5 s horizon, respecting its
  speed and acceleration limits and keeping a safety distance from the plans
  its neighbours have announced.
- **`StraightLinePolicy`**: flies each drone directly at its goal and ignores
  every other drone. It performs no collision avoidance.
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

Five drones start on a sphere and each flies to the opposite side, so every
straight path crosses at the centre. With DMPC all five get through:

```text
policy: dmpc, drones: 5
simulated 5.24 s
  drone_0: reached goal (final error 0.011 m)
  ...
minimum clearance between drones: +0.216 m
collisions: none
```

The naive baseline on the same scenario collides at the centre at about
1.67 s and stays jammed there until the time limit:

```bash
uv run examples/five_drones.py --policy straight
```

```text
  drone_0: DID NOT reach goal (final error 3.298 m)
  ...
minimum clearance between drones: -0.019 m
drone-drone collisions: 14 contact(s), pairs [(0, 1), (0, 2), (0, 3), (1, 3), (1, 4), (2, 4), (3, 4)]
```

### Results so far

Measured on one machine (Apple M1); "clearance" is the smallest gap between
any two collision spheres over the run.

| Scenario | Policy | Reached goal | Contacts | Min clearance | Time |
| --- | --- | --- | --- | --- | --- |
| Head-on pair | straight | 0 / 2 | 2 | -0.03 m | timeout |
| Head-on pair | DMPC | 2 / 2 | 0 | +0.22 m | 4.56 s |
| 5-drone swap | straight | 0 / 5 | 14 | -0.02 m | timeout |
| 5-drone swap | DMPC | 5 / 5 | 0 | +0.22 m | 5.24 s |
| 8-drone swap | DMPC | 8 / 8 | 0 | +0.17 m | 5.60 s |
| 12-drone swap | DMPC | 12 / 12 | 0 | +0.06 m | 7.48 s |

Try other fleet sizes with `--drones N`.

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
from atlas import DMPCPolicy, Simulator, antipodal_sphere

robots = antipodal_sphere(num_robots=5)
sim = Simulator(robots, DMPCPolicy(robots))
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
- DMPC's separation guarantee holds for its planning model at the sampled
  steps. The real drone tracks the plan imperfectly, which a 0.15 m safety
  margin absorbs; this is an empirical result, not a proof for the physical
  system.
- DMPC is not yet tuned for dense traffic. With 12 drones through one point,
  37 of roughly 900 solves hit the iteration limit (the drone brakes for that
  step) and clearance drops to 6 cm, still without contact.
- DMPC is not real-time yet: about 30 ms per drone per replan with 12 drones.
- There are no static obstacles.
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
| M0 (done) | 3D MuJoCo simulator, autopilot, straight-line baseline, collision detection, tests, CI |
| **M1 (this)** | Distributed MPC planner with collision avoidance |
| M2 | Control barrier function safety filter |
| M3 | ORCA baseline and benchmark metrics for comparison |
| M4 | Static obstacles and larger fleets |
| M5 | Benchmark suite and reproducible experiments |
| M6 | ROS 2 / Gazebo integration |
