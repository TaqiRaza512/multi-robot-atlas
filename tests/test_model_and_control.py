import mujoco
import numpy as np
import pytest

from atlas import ControlGains, Robot, RobotConfig, SimConfig, build_model, tracking_wrench
from atlas.control import clip_norm, quat_to_mat
from atlas.scenarios import antipodal_sphere, head_on, parallel_lanes


def test_model_layout() -> None:
    robots = antipodal_sphere(num_robots=5)
    model = build_model(robots)
    assert model.nq == 7 * 5
    assert model.nv == 6 * 5
    assert model.nu == 4 * 5
    for robot in robots:
        body = model.body(robot.name)
        assert body.mass[0] == pytest.approx(robot.config.mass)
        np.testing.assert_allclose(body.inertia, robot.config.inertia)
        np.testing.assert_allclose(body.pos, robot.start)


def test_model_uses_sim_config() -> None:
    model = build_model(head_on(), SimConfig(timestep=0.001, gravity=3.7))
    assert model.opt.timestep == pytest.approx(0.001)
    np.testing.assert_allclose(model.opt.gravity, [0, 0, -3.7])


def test_only_the_collision_sphere_collides() -> None:
    robots = head_on()
    model = build_model(robots)
    collidable = [i for i in range(model.ngeom) if model.geom_contype[i] != 0]
    names = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) for i in collidable}
    assert names == {"ground", "drone_0_collision", "drone_1_collision"}


def test_duplicate_names_rejected() -> None:
    robot = Robot(name="a", start=np.array([0.0, 0, 1]), goal=np.array([1.0, 0, 1]))
    with pytest.raises(ValueError, match="unique"):
        build_model([robot, robot])
    with pytest.raises(ValueError):
        build_model([])


def test_hover_wrench_is_weight_and_zero_torque() -> None:
    cfg = RobotConfig()
    zeros = np.zeros((1, 3))
    wrench = tracking_wrench(
        desired_velocities=zeros,
        velocities=zeros,
        rotations=np.eye(3)[None],
        angular_velocities=zeros,
        masses=np.array([cfg.mass]),
        inertias=np.array([cfg.inertia]),
        max_accelerations=np.array([cfg.max_acceleration]),
        gains=ControlGains(),
        gravity=9.81,
    )
    np.testing.assert_allclose(wrench, [[cfg.mass * 9.81, 0, 0, 0]], atol=1e-12)


def test_forward_command_pitches_nose_down() -> None:
    cfg = RobotConfig()
    zeros = np.zeros((1, 3))
    wrench = tracking_wrench(
        desired_velocities=np.array([[1.0, 0.0, 0.0]]),
        velocities=zeros,
        rotations=np.eye(3)[None],
        angular_velocities=zeros,
        masses=np.array([cfg.mass]),
        inertias=np.array([cfg.inertia]),
        max_accelerations=np.array([cfg.max_acceleration]),
        gains=ControlGains(),
        gravity=9.81,
    )
    # Accelerating along +x needs a positive rotation about body y, and nothing else.
    assert wrench[0, 2] > 0
    assert wrench[0, 1] == pytest.approx(0.0, abs=1e-12)
    assert wrench[0, 3] == pytest.approx(0.0, abs=1e-12)


def test_clip_norm() -> None:
    vectors = np.array([[3.0, 4.0, 0.0], [0.0, 0.0, 0.0], [0.1, 0.0, 0.0]])
    clipped = clip_norm(vectors, np.array([1.0, 1.0, 1.0]))
    np.testing.assert_allclose(clipped, [[0.6, 0.8, 0.0], [0, 0, 0], [0.1, 0, 0]])


def test_quat_to_mat_matches_mujoco() -> None:
    quats = np.array([[1.0, 0, 0, 0], [0.5, 0.5, 0.5, 0.5], [0.8, 0.0, 0.6, 0.0]])
    expected = np.empty((3, 9))
    for i, quat in enumerate(quats):
        mujoco.mju_quat2Mat(expected[i], quat)
    np.testing.assert_allclose(quat_to_mat(quats).reshape(3, 9), expected, atol=1e-15)


def test_antipodal_sphere_geometry() -> None:
    robots = antipodal_sphere(num_robots=7, sphere_radius=2.0, center=(1.0, 2.0, 5.0))
    center = np.array([1.0, 2.0, 5.0])
    for robot in robots:
        assert np.linalg.norm(robot.start - center) == pytest.approx(2.0)
        np.testing.assert_allclose(robot.start + robot.goal, 2 * center)
    assert len({robot.name for robot in robots}) == 7
    with pytest.raises(ValueError):
        antipodal_sphere(sphere_radius=5.0, center=(0, 0, 4))


def test_parallel_lanes_do_not_cross() -> None:
    robots = parallel_lanes(num_robots=4, length=5.0, spacing=1.0)
    for robot in robots:
        np.testing.assert_allclose(robot.goal - robot.start, [5.0, 0, 0])
    starts = np.stack([robot.start for robot in robots])
    gaps = np.linalg.norm(starts[:, None] - starts[None], axis=2)[np.triu_indices(4, 1)]
    assert gaps.min() > 2 * robots[0].config.radius
