import numpy as np
import pytest

from atlas import (
    Robot,
    RobotConfig,
    SimConfig,
    Simulator,
    StraightLinePolicy,
    WorldState,
    antipodal_sphere,
    head_on,
    parallel_lanes,
)
from atlas.types import FloatArray


def _single(start: list[float], goal: list[float], **config: float) -> list[Robot]:
    return [Robot("solo", np.array(start), np.array(goal), RobotConfig(**config))]


def test_hover_holds_position() -> None:
    sim = Simulator(_single([0, 0, 2], [0, 0, 2]), StraightLinePolicy())
    for _ in range(100):
        sim.step()
    result = sim.result()
    assert result.duration == pytest.approx(2.0)
    np.testing.assert_allclose(result.positions[:, 0], np.tile([0, 0, 2], (101, 1)), atol=1e-9)
    assert result.collision_free


def test_already_at_goal_runs_zero_steps() -> None:
    result = Simulator(_single([0, 0, 2], [0, 0, 2]), StraightLinePolicy()).run()
    assert result.duration == 0.0
    assert result.all_reached_goal
    assert result.min_clearance == float("inf")


def test_single_robot_flies_a_straight_line_to_its_goal() -> None:
    start, goal = np.array([0.0, 0.0, 2.0]), np.array([5.0, 1.0, 3.0])
    sim = Simulator(_single(list(start), list(goal)), StraightLinePolicy())
    result = sim.run()

    assert result.all_reached_goal
    assert result.duration < 6.0
    assert np.linalg.norm(result.positions[-1, 0] - goal) <= sim.config.goal_tolerance

    direction = (goal - start) / np.linalg.norm(goal - start)
    offset = result.positions[:, 0] - start
    lateral = np.linalg.norm(offset - np.outer(offset @ direction, direction), axis=1)
    assert lateral.max() < 0.05

    speed = np.linalg.norm(result.velocities[:, 0], axis=1)
    assert speed.max() <= 1.02 * sim.robots[0].config.max_speed
    assert speed.max() > 0.95 * sim.robots[0].config.max_speed


def test_runs_are_deterministic() -> None:
    def run() -> tuple[FloatArray, FloatArray, int]:
        result = Simulator(antipodal_sphere(5), StraightLinePolicy(), SimConfig(max_time=4.0)).run()
        return result.positions, result.quaternions, len(result.collisions)

    first, second = run(), run()
    assert np.array_equal(first[0], second[0])
    assert np.array_equal(first[1], second[1])
    assert first[2] == second[2]


def test_reset_reproduces_the_same_run() -> None:
    sim = Simulator(head_on(), StraightLinePolicy(), SimConfig(max_time=4.0))
    first = sim.run()
    second = sim.run()
    assert np.array_equal(first.positions, second.positions)
    assert first.collisions == second.collisions


def test_parallel_lanes_are_collision_free() -> None:
    result = Simulator(parallel_lanes(num_robots=5), StraightLinePolicy()).run()
    assert result.all_reached_goal
    assert result.collision_free
    assert result.colliding_pairs == []
    assert result.min_clearance > 0.5


def test_head_on_collision_is_detected() -> None:
    robots = head_on(separation=6.0)
    result = Simulator(robots, StraightLinePolicy(), SimConfig(max_time=6.0)).run()

    assert not result.collision_free
    assert result.colliding_pairs == [(0, 1)]
    assert result.min_clearance < 0
    first = result.robot_collisions[0]
    # Each drone covers 3 m minus one radius at up to 2 m/s, after accelerating.
    assert 1.3 < first.time < 2.5
    assert first.max_penetration > 0
    assert first.duration > 0
    # The naive policy does not get both drones past each other.
    assert not result.all_reached_goal


def test_five_robot_crossing_collides() -> None:
    result = Simulator(antipodal_sphere(5), StraightLinePolicy(), SimConfig(max_time=5.0)).run()
    assert result.positions.shape[1:] == (5, 3)
    assert len(result.colliding_pairs) >= 1
    assert result.min_clearance < 0


def test_overlapping_starts_are_reported_at_time_zero() -> None:
    robots = [
        Robot("a", np.array([0.0, 0.0, 2.0]), np.array([0.0, 0.0, 2.0])),
        Robot("b", np.array([0.2, 0.0, 2.0]), np.array([0.2, 0.0, 2.0])),
    ]
    sim = Simulator(robots, StraightLinePolicy())
    result = sim.result()
    assert result.colliding_pairs == [(0, 1)]
    assert result.collisions[0].time == 0.0
    assert result.min_clearance == pytest.approx(0.2 - 2 * 0.18)


def test_ground_contact_is_reported_separately() -> None:
    def descend(world: WorldState) -> FloatArray:
        return np.tile([0.0, 0.0, -1.0], (world.num_robots, 1))

    sim = Simulator(_single([0, 0, 0.5], [0, 0, 0.5]), descend, SimConfig(max_time=2.0))
    for _ in range(100):
        sim.step()
    result = sim.result()
    assert result.robot_collisions == ()
    assert [(e.robot_a, e.robot_b) for e in result.ground_collisions][0] == (0, None)


def test_setpoints_are_limited_to_max_speed() -> None:
    def too_fast(world: WorldState) -> FloatArray:
        return np.tile([100.0, 0.0, 0.0], (world.num_robots, 1))

    sim = Simulator(_single([0, 0, 2], [50, 0, 2], max_speed=1.5), too_fast, SimConfig())
    for _ in range(150):
        sim.step()
    speed = np.linalg.norm(sim.result().velocities[:, 0], axis=1)
    assert speed.max() == pytest.approx(1.5, rel=0.02)


def test_max_time_stops_the_run() -> None:
    config = SimConfig(max_time=0.5)
    result = Simulator(_single([0, 0, 2], [50, 0, 2]), StraightLinePolicy(), config).run()
    assert result.duration == pytest.approx(0.5)
    assert not result.all_reached_goal
    assert len(result.times) == 26


def test_bad_policy_output_is_rejected() -> None:
    robots = _single([0, 0, 2], [1, 0, 2])
    with pytest.raises(ValueError, match="shape"):
        Simulator(robots, lambda world: np.zeros(3)).step()
    with pytest.raises(ValueError, match="non-finite"):
        Simulator(robots, lambda world: np.full((1, 3), np.nan)).step()


def test_max_acceleration_must_be_below_gravity() -> None:
    with pytest.raises(ValueError, match="gravity"):
        Simulator(_single([0, 0, 2], [1, 0, 2], max_acceleration=12.0), StraightLinePolicy())
