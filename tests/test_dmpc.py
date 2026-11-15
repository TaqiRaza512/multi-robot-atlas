import numpy as np
import pytest

from atlas import (
    DMPCConfig,
    DMPCPolicy,
    Robot,
    SimConfig,
    Simulator,
    StraightLinePolicy,
    antipodal_sphere,
    head_on,
)
from atlas.dmpc import _ball_directions


def test_ball_directions_are_an_inner_approximation() -> None:
    directions, scale = _ball_directions()
    rng = np.random.default_rng(0)
    points = rng.normal(size=(5000, 3))
    points /= np.linalg.norm(points, axis=1, keepdims=True)
    # Scale every point to the polytope's boundary; none may leave the unit ball.
    boundary = points * (scale / (points @ directions.T).max(axis=1))[:, None]
    assert np.linalg.norm(boundary, axis=1).max() <= 1.0
    assert scale > 0.85  # and the approximation is not wasteful


@pytest.mark.parametrize(
    "kwargs",
    [{"horizon": 0}, {"step": 0.0}, {"safety_margin": -0.1}, {"sidestep_angle": 2.0}],
)
def test_config_validation(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        DMPCConfig(**kwargs)  # type: ignore[arg-type]


def test_single_drone_reaches_goal_within_limits() -> None:
    robots = [Robot("solo", np.array([0.0, 0.0, 2.0]), np.array([5.0, 1.0, 3.0]))]
    policy = DMPCPolicy(robots)
    result = Simulator(robots, policy).run()
    assert result.all_reached_goal
    assert result.duration < 6.0
    speed = np.linalg.norm(result.velocities[:, 0], axis=1)
    assert speed.max() <= robots[0].config.max_speed
    assert policy.solver_failures == 0


def test_head_on_pair_passes_without_contact() -> None:
    robots = head_on()
    policy = DMPCPolicy(robots)
    result = Simulator(robots, policy, SimConfig(max_time=10.0)).run()

    assert result.all_reached_goal
    assert result.collision_free
    assert result.min_clearance > 0.1
    assert policy.solver_failures == 0

    # The same scenario under the naive baseline collides.
    baseline = Simulator(robots, StraightLinePolicy(), SimConfig(max_time=4.0)).run()
    assert not baseline.collision_free


def test_five_drone_swap_is_collision_free() -> None:
    robots = antipodal_sphere(5)
    policy = DMPCPolicy(robots)
    result = Simulator(robots, policy, SimConfig(max_time=15.0)).run()

    assert result.all_reached_goal
    assert result.collision_free
    assert result.min_clearance > 0.1
    assert policy.solver_failures == 0
    assert policy.max_slack < 0.02


def test_sidestep_increases_clearance_in_a_head_on_pass() -> None:
    robots = head_on()

    def clearance(angle: float) -> float:
        policy = DMPCPolicy(robots, DMPCConfig(sidestep_angle=angle))
        result = Simulator(robots, policy, SimConfig(max_time=10.0)).run()
        assert result.collision_free
        return result.min_clearance

    assert clearance(0.5) > clearance(0.0)


def test_runs_are_deterministic_and_reset_clears_state() -> None:
    robots = head_on()
    sim = Simulator(robots, DMPCPolicy(robots), SimConfig(max_time=3.0))
    first = sim.run()
    second = sim.run()
    fresh = Simulator(robots, DMPCPolicy(robots), SimConfig(max_time=3.0)).run()
    assert np.array_equal(first.positions, second.positions)
    assert np.array_equal(first.positions, fresh.positions)


def test_plans_have_the_documented_shape() -> None:
    robots = head_on()
    policy = DMPCPolicy(robots, DMPCConfig(horizon=8))
    sim = Simulator(robots, policy)
    sim.step()
    assert policy.plans.shape == (2, 9, 3)
    np.testing.assert_allclose(policy.plans[:, 0], [r.start for r in robots], atol=1e-9)


def test_robot_count_mismatch_is_rejected() -> None:
    policy = DMPCPolicy(head_on())
    with pytest.raises(ValueError, match="robots"):
        Simulator(antipodal_sphere(3), policy).step()
