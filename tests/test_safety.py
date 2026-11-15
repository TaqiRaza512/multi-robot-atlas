import numpy as np
import pytest

from atlas import (
    DMPCPolicy,
    Robot,
    SafetyFilter,
    SafetyFilterConfig,
    SimConfig,
    Simulator,
    StraightLinePolicy,
    WorldState,
    antipodal_sphere,
    head_on,
    parallel_lanes,
)


def _world(positions: list[list[float]], velocities: list[list[float]]) -> WorldState:
    n = len(positions)
    return WorldState(
        time=0.0,
        positions=np.array(positions, dtype=np.float64),
        velocities=np.array(velocities, dtype=np.float64),
        rotations=np.tile(np.eye(3), (n, 1, 1)),
        angular_velocities=np.zeros((n, 3)),
        goals=np.array(positions, dtype=np.float64)[::-1].copy(),
        radii=np.full(n, 0.18),
        max_speeds=np.full(n, 2.0),
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"safety_margin": -0.1},
        {"braking_fraction": 0.0},
        {"braking_fraction": 1.5},
        {"barrier_gain": 0.0},
    ],
)
def test_config_validation(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        SafetyFilterConfig(**kwargs)


def test_no_conflict_means_no_intervention() -> None:
    robots = parallel_lanes(5)
    plain = Simulator(robots, StraightLinePolicy()).run()
    safety = SafetyFilter(StraightLinePolicy(), robots)
    filtered = Simulator(robots, safety).run()
    assert safety.interventions == 0
    assert safety.ticks > 0
    # Passing setpoints through untouched reproduces the unfiltered run exactly.
    assert np.array_equal(plain.positions, filtered.positions)


def test_far_apart_drones_pass_through_unchanged() -> None:
    robots = head_on(separation=40.0)
    safety = SafetyFilter(StraightLinePolicy(), robots)
    world = _world([[-20, 0, 2], [20, 0, 2]], [[2, 0, 0], [-2, 0, 0]])
    np.testing.assert_array_equal(safety(world), StraightLinePolicy()(world))
    assert safety.interventions == 0


def test_closing_drones_are_told_to_brake() -> None:
    robots = head_on(separation=2.0)
    safety = SafetyFilter(StraightLinePolicy(), robots)
    # 1.2 m apart and closing at 4 m/s: far too fast to stop in time.
    world = _world([[-0.6, 0, 2], [0.6, 0, 2]], [[2, 0, 0], [-2, 0, 0]])
    setpoints = safety(world)
    assert safety.interventions == 2
    # Each drone is asked to decelerate with its full assumed braking (0.6 * 5 m/s^2).
    gain = safety.config.velocity_gain
    assert gain * (2.0 - setpoints[0, 0]) == pytest.approx(3.0, abs=0.05)
    assert gain * (setpoints[1, 0] + 2.0) == pytest.approx(3.0, abs=0.05)
    assert np.linalg.norm(setpoints, axis=1).max() <= 2.0 + 1e-6


def test_barrier_constraints_are_antisymmetric_and_loose_when_separating() -> None:
    robots = head_on(separation=2.0)
    safety = SafetyFilter(StraightLinePolicy(), robots)
    separating = _world([[-0.6, 0, 2], [0.6, 0, 2]], [[-1, 0, 0], [1, 0, 0]])
    normals, bounds = safety.barrier_constraints(separating)
    np.testing.assert_allclose(normals[0, 1], -normals[1, 0])
    np.testing.assert_allclose(normals[0, 1], [-1, 0, 0])
    assert bounds[0, 1] == pytest.approx(bounds[1, 0])
    assert bounds[0, 1] <= -5.0  # no acceleration within limits can violate it
    assert np.isneginf(bounds[0, 0])


def test_overlapping_drones_are_pushed_apart() -> None:
    robots = head_on(separation=2.0)
    safety = SafetyFilter(StraightLinePolicy(), robots)
    world = _world([[-0.2, 0, 2], [0.2, 0, 2]], [[0, 0, 0], [0, 0, 0]])
    setpoints = safety(world)
    assert setpoints[0, 0] < 0 < setpoints[1, 0]


def test_filter_makes_the_naive_policy_collision_free() -> None:
    for robots in (head_on(), antipodal_sphere(5)):
        baseline = Simulator(robots, StraightLinePolicy(), SimConfig(max_time=4.0)).run()
        assert not baseline.collision_free

        safety = SafetyFilter(StraightLinePolicy(), robots)
        result = Simulator(robots, safety, SimConfig(max_time=15.0)).run()
        assert result.collision_free
        assert result.min_clearance > 0.05
        assert safety.interventions > 0
        assert safety.solver_failures <= 0.01 * safety.interventions


def test_filter_composes_with_dmpc() -> None:
    robots = antipodal_sphere(5)
    safety = SafetyFilter(DMPCPolicy(robots), robots)
    result = Simulator(robots, safety, SimConfig(max_time=15.0)).run()
    assert result.all_reached_goal
    assert result.collision_free
    assert result.min_clearance > 0.15


def test_runs_are_deterministic_and_reset_propagates() -> None:
    robots = head_on()
    safety = SafetyFilter(DMPCPolicy(robots), robots)
    sim = Simulator(robots, safety, SimConfig(max_time=3.0))
    first = sim.run()
    interventions = safety.interventions
    second = sim.run()
    assert np.array_equal(first.positions, second.positions)
    assert safety.interventions == interventions  # counters were reset, not accumulated


def test_single_drone_is_never_filtered() -> None:
    robots = [Robot("solo", np.array([0.0, 0.0, 2.0]), np.array([3.0, 0.0, 2.0]))]
    safety = SafetyFilter(StraightLinePolicy(), robots)
    result = Simulator(robots, safety).run()
    assert result.all_reached_goal
    assert safety.interventions == 0
    assert safety.intervention_rate == 0.0
