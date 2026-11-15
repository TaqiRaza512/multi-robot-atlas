"""Render the animations shown in the README into ``docs/media``.

    uv sync --extra video
    uv run examples/render_media.py

Runs are deterministic, so the animations can be regenerated exactly.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

from atlas import (
    DMPCPolicy,
    Policy,
    Robot,
    SafetyFilter,
    SimConfig,
    Simulator,
    StraightLinePolicy,
    antipodal_sphere,
    head_on,
)
from atlas.video import render_frames

MEDIA = Path(__file__).resolve().parent.parent / "docs" / "media"
WIDTH, HEIGHT, FPS = 640, 480, 20
ZOOM = 1.25
# The naive policy never finishes; show only the approach and the pile-up.
NAIVE_CONFIG = SimConfig(max_time=6.0)


def write_gif(frames: list[np.ndarray], path: Path) -> None:
    """Encode frames as a GIF with a palette computed from the whole clip."""
    import imageio_ffmpeg

    command = [
        imageio_ffmpeg.get_ffmpeg_exe(),
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{WIDTH}x{HEIGHT}",
        "-r",
        str(FPS),
        "-i",
        "-",
        "-filter_complex",
        "split[a][b];[a]palettegen=max_colors=96[p];[b][p]paletteuse=dither=bayer:bayer_scale=4",
        str(path),
    ]
    subprocess.run(command, input=np.stack(frames).tobytes(), check=True)


def render(
    name: str,
    robots: list[Robot],
    policy: Policy,
    config: SimConfig | None = None,
) -> None:
    sim = Simulator(robots, policy, config)
    result = sim.run()
    frames = render_frames(
        sim,
        result,
        fps=FPS,
        width=WIDTH,
        height=HEIGHT,
        show_collision_spheres=True,  # the bubbles that must never touch
        zoom=ZOOM,
    )
    frames += [frames[-1]] * FPS  # hold the last frame for a second
    path = MEDIA / f"{name}.gif"
    write_gif(frames, path)
    print(
        f"{path.name}: {result.duration:.2f} s simulated, "
        f"{int(result.reached_goal.sum())}/{len(robots)} reached, "
        f"{len(result.robot_collisions)} contact(s), "
        f"clearance {result.min_clearance:+.2f} m, {path.stat().st_size / 1e6:.1f} MB"
    )


def main() -> None:
    MEDIA.mkdir(parents=True, exist_ok=True)

    five = antipodal_sphere(5)
    render("swap5_straight", five, StraightLinePolicy(), NAIVE_CONFIG)
    render("swap5_dmpc", five, DMPCPolicy(five))

    pair = head_on()
    render("head_on_straight", pair, StraightLinePolicy(), NAIVE_CONFIG)
    render("head_on_filter", pair, SafetyFilter(StraightLinePolicy(), pair))

    twelve = antipodal_sphere(12)
    render("swap12_dmpc_filter", twelve, SafetyFilter(DMPCPolicy(twelve), twelve))


if __name__ == "__main__":
    main()
