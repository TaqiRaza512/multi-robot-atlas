import numpy as np
import pytest

from atlas import Robot, RobotConfig, SimConfig


def test_robot_normalises_vectors_and_freezes_them() -> None:
    robot = Robot(name="a", start=[0, 0, 1], goal=(1, 2, 3))  # type: ignore[arg-type]
    assert robot.start.dtype == np.float64
    assert robot.goal.tolist() == [1.0, 2.0, 3.0]
    with pytest.raises(ValueError):
        robot.start[0] = 5.0


@pytest.mark.parametrize("bad", [[0, 0], [0, 0, 0, 0], [0, np.nan, 0], [np.inf, 0, 0]])
def test_robot_rejects_bad_vectors(bad: list[float]) -> None:
    with pytest.raises(ValueError):
        Robot(name="a", start=bad, goal=[0, 0, 1])  # type: ignore[arg-type]


@pytest.mark.parametrize("name", ["", "1abc", "has space", "a-b", 'x"y'])
def test_robot_rejects_bad_names(name: str) -> None:
    with pytest.raises(ValueError):
        Robot(name=name, start=[0, 0, 1], goal=[0, 0, 2])  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"radius": 0.0},
        {"mass": -1.0},
        {"max_speed": float("nan")},
        {"max_acceleration": 0.0},
        {"thrust_to_weight": 1.0},
        {"inertia": (1e-3, 0.0, 1e-3)},
        {"max_torque": float("inf")},
    ],
)
def test_robot_config_validation(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        RobotConfig(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "kwargs",
    [{"timestep": 0.0}, {"policy_decimation": 0}, {"max_time": -1.0}, {"goal_tolerance": 0.0}],
)
def test_sim_config_validation(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        SimConfig(**kwargs)  # type: ignore[arg-type]


def test_policy_period() -> None:
    assert SimConfig(timestep=0.002, policy_decimation=10).policy_period == pytest.approx(0.02)
