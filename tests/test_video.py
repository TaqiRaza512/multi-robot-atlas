import numpy as np
import pytest

from atlas import SimConfig, Simulator, StraightLinePolicy, head_on


def test_render_frames() -> None:
    from atlas.video import render_frames

    sim = Simulator(head_on(), StraightLinePolicy(), SimConfig(max_time=0.5))
    result = sim.run()
    time_before = sim.time
    try:
        frames = render_frames(sim, result, fps=10, width=160, height=120)
    except Exception as error:  # no OpenGL context, e.g. on a headless CI runner
        pytest.skip(f"offscreen rendering unavailable: {error}")

    assert len(frames) == 6
    assert frames[0].shape == (120, 160, 3)
    assert frames[0].dtype == np.uint8
    assert frames[0].std() > 0  # not a blank image
    assert sim.time == time_before  # replay must not disturb the simulator
