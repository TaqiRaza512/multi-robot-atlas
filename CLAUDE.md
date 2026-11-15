# Multi-Robot Atlas

Deterministic 3D multi-drone navigation simulator on MuJoCo. Python package `atlas` in `src/atlas`.

## Commands

```bash
uv sync                      # install (add --extra video for mp4 output)
uv run pytest                # tests (~20 s)
uv run ruff check . && uv run ruff format .
uv run mypy                  # strict, src only
uv run examples/five_drones.py
uv run examples/benchmark.py     # policy x scenario table (~90 s)
uv run mjpython examples/five_drones.py --view   # live viewer; macOS requires mjpython
```

All four checks (pytest, ruff check, ruff format --check, mypy) must pass before a change is proposed; CI runs them on Linux and Apple Silicon.

## Architecture

Control flows `Policy -> autopilot -> MuJoCo`:

- `types.py`: `RobotConfig`, `Robot`, `SimConfig`, `WorldState` (frozen dataclasses, SI units, z-up).
- `model.py`: generates the MJCF. Robot `i` owns `qpos[7i:7i+7]`, `qvel[6i:6i+6]`, `ctrl[4i:4i+4]` (thrust, torque x/y/z). The simulator relies on this layout.
- `policies.py`: `Policy` protocol, `WorldState -> (N, 3)` velocity setpoints at 50 Hz, and the naive `StraightLinePolicy`. If a policy defines `reset()`, the simulator calls it on every reset.
- `dmpc.py`: `DMPCPolicy`, distributed MPC with collision avoidance. One OSQP problem per drone, replanned every 0.1 s against neighbours' announced plans (separating planes through predicted midpoints, soft constraints).
- `control.py`: geometric SE(3) autopilot, velocity setpoint to thrust and torques, at the physics rate (500 Hz). Vectorised over robots.
- `simulator.py`: `Simulator` (`reset` / `step` / `run` / `result`), contact-episode bookkeeping, `SimulationResult`.
- `safety.py`: `SafetyFilter`, a CBF layer that wraps any policy (Wang-Ames-Egerstedt barrier, one tiny QP per drone, solved only when the nominal command is unsafe).
- `scenarios.py`: deterministic scenario generators.
- `examples/benchmark.py`: regenerates the README results table.
- `video.py`: offscreen replay rendering.
- `examples/render_media.py`: regenerates the README animations in `docs/media` (the only tracked GIFs; other media is git-ignored).

## Rules

- **No randomness in the simulator or scenarios.** Runs must be bit-reproducible on one machine. If a scenario ever needs randomness, take an explicit seed.
- **Keep claims honest.** `StraightLinePolicy` ignores other robots by design. DMPC's guarantee is for its planning model at sampled steps, not for the physical drone. Results tables in the README must come from real runs.
- **Stateful policies must be deterministic too.** Recreate solver state in `reset()`, and never use solver settings that adapt on wall-clock time (OSQP needs a fixed `adaptive_rho_interval`).
- QP conditioning matters with OSQP: scale slack variables so the Hessian is near identity, and clamp constraint bounds to physically meaningful ranges. Badly scaled problems hit the iteration cap.
- `DMPCConfig.velocity_gain` and `SafetyFilterConfig.velocity_gain` must equal `ControlGains.velocity`: the planner models the autopilot's velocity loop.
- Keep dependencies minimal: runtime is `mujoco`, `numpy`, `osqp` and `scipy`. No ROS 2 until M6.
- Development machine is an Apple Silicon Mac. Do not add dependencies that need CUDA or an NVIDIA GPU (Isaac Sim/Lab, etc.).
- Free-joint conventions: `qvel[:3]` is world-frame linear velocity, `qvel[3:]` is body-frame angular velocity. Quaternions are `w, x, y, z`.
- After `mj_step`, `data.contact` and `data.xpos`/`xmat` describe the state at the *start* of that step. Read poses from `qpos`, as the simulator does.
- MuJoCo, OSQP and SciPy have no usable type stubs here; mypy treats them as `Any`.
- Each milestone is one GitHub issue and one pull request. Do not commit directly to `main` after the initial commit.
