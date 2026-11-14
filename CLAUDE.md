# Multi-Robot Atlas

Deterministic 3D multi-drone navigation simulator on MuJoCo. Python package `atlas` in `src/atlas`.

## Commands

```bash
uv sync                      # install (add --extra video for mp4 output)
uv run pytest                # tests (~3 s)
uv run ruff check . && uv run ruff format .
uv run mypy                  # strict, src only
uv run examples/five_drones.py
uv run mjpython examples/five_drones.py --view   # live viewer; macOS requires mjpython
```

All four checks (pytest, ruff check, ruff format --check, mypy) must pass before a change is proposed; CI runs them on Linux and Apple Silicon.

## Architecture

Control flows `Policy -> autopilot -> MuJoCo`:

- `types.py`: `RobotConfig`, `Robot`, `SimConfig`, `WorldState` (frozen dataclasses, SI units, z-up).
- `model.py`: generates the MJCF. Robot `i` owns `qpos[7i:7i+7]`, `qvel[6i:6i+6]`, `ctrl[4i:4i+4]` (thrust, torque x/y/z). The simulator relies on this layout.
- `policies.py`: `Policy` protocol, `WorldState -> (N, 3)` velocity setpoints at 50 Hz. New planners (ORCA, MPC) belong here.
- `control.py`: geometric SE(3) autopilot, velocity setpoint to thrust and torques, at the physics rate (500 Hz). Vectorised over robots.
- `simulator.py`: `Simulator` (`reset` / `step` / `run` / `result`), contact-episode bookkeeping, `SimulationResult`.
- `scenarios.py`: deterministic scenario generators.
- `video.py`: offscreen replay rendering.

## Rules

- **No randomness in the simulator or scenarios.** Runs must be bit-reproducible on one machine. If a scenario ever needs randomness, take an explicit seed.
- **Do not claim collision avoidance.** `StraightLinePolicy` ignores other robots by design; the simulator only detects collisions. Keep docs honest about what is implemented.
- Keep dependencies minimal: runtime is `mujoco` and `numpy` only. No ROS 2 until M6.
- Development machine is an Apple Silicon Mac. Do not add dependencies that need CUDA or an NVIDIA GPU (Isaac Sim/Lab, etc.).
- Free-joint conventions: `qvel[:3]` is world-frame linear velocity, `qvel[3:]` is body-frame angular velocity. Quaternions are `w, x, y, z`.
- After `mj_step`, `data.contact` and `data.xpos`/`xmat` describe the state at the *start* of that step. Read poses from `qpos`, as the simulator does.
- MuJoCo ships no type stubs; mypy treats it as `Any`.
- Each milestone is one GitHub issue and one pull request. Do not commit directly to `main` after the initial commit.
