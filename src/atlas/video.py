"""Offscreen rendering of a finished run.

Rendering needs an OpenGL context (available by default on macOS, including
Apple Silicon). Writing video files additionally needs the ``video`` extra:
``uv sync --extra video``.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
from numpy.typing import NDArray

from atlas.simulator import SimulationResult, Simulator

Frame = NDArray[np.uint8]


def default_camera(sim: Simulator) -> mujoco.MjvCamera:
    """A free camera framing every start and goal."""
    points = np.concatenate([[r.start for r in sim.robots], [r.goal for r in sim.robots]])
    low, high = points.min(axis=0), points.max(axis=0)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = 0.5 * (low + high)
    camera.distance = 1.1 * max(float(np.linalg.norm(high - low)), 2.0)
    camera.azimuth = 130.0
    camera.elevation = -20.0
    return camera


def render_frames(
    sim: Simulator,
    result: SimulationResult,
    fps: int = 30,
    width: int = 960,
    height: int = 720,
    show_collision_spheres: bool = False,
) -> list[Frame]:
    """Replay ``result`` and return RGB frames at ``fps``.

    The replay uses a private ``MjData``, so ``sim`` is left untouched.
    """
    data = mujoco.MjData(sim.model)
    camera = default_camera(sim)
    option = mujoco.MjvOption()
    option.geomgroup[3] = int(show_collision_spheres)

    sample_times = np.arange(0.0, result.duration + 0.5 / fps, 1.0 / fps)
    indices = np.minimum(np.searchsorted(result.times, sample_times), len(result.times) - 1)

    frames: list[Frame] = []
    with mujoco.Renderer(sim.model, height=height, width=width) as renderer:
        for index in indices:
            data.qpos[:] = np.concatenate(
                [result.positions[index], result.quaternions[index]], axis=1
            ).ravel()
            mujoco.mj_forward(sim.model, data)
            renderer.update_scene(data, camera=camera, scene_option=option)
            frames.append(renderer.render().copy())
    return frames


def save_video(
    sim: Simulator,
    result: SimulationResult,
    path: str | Path,
    fps: int = 30,
    **render_options: int | bool,
) -> Path:
    """Render ``result`` to a video file (format chosen by the extension, e.g. ``.mp4``)."""
    try:
        import imageio.v3 as iio
    except ImportError as error:
        raise ImportError("saving video needs the 'video' extra: uv sync --extra video") from error

    frames = render_frames(sim, result, fps=fps, **render_options)  # type: ignore[arg-type]
    target = Path(path)
    iio.imwrite(target, np.stack(frames), fps=fps)
    return target
