"""Low-level quadrotor autopilot: velocity setpoint to thrust and body torques.

A cascaded geometric controller in the style of Lee, Leok and McClamroch,
"Geometric tracking control of a quadrotor UAV on SE(3)" (CDC 2010):

1. The velocity error gives a desired acceleration (norm-limited).
2. Adding gravity compensation gives a desired thrust vector, whose direction
   defines the desired attitude (yaw is held at zero).
3. A PD law on the SO(3) attitude error gives the body torques.

Everything is vectorised over the team: inputs have a leading dimension ``N``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from atlas.types import FloatArray


@dataclass(frozen=True)
class ControlGains:
    """Autopilot gains, normalised by mass and inertia.

    Attributes:
        velocity: Velocity error to acceleration [1/s].
        attitude: Attitude error to angular acceleration [1/s^2].
        angular_rate: Angular rate to angular acceleration [1/s].
    """

    velocity: float = 4.0
    attitude: float = 400.0
    angular_rate: float = 36.0


def clip_norm(vectors: FloatArray, limits: FloatArray) -> FloatArray:
    """Scale each row of ``vectors`` so its norm does not exceed its limit."""
    norms = np.linalg.norm(vectors, axis=1)
    scale = np.minimum(1.0, np.divide(limits, norms, out=np.ones_like(norms), where=norms > 0))
    clipped: FloatArray = vectors * scale[:, None]
    return clipped


def quat_to_mat(quaternions: FloatArray) -> FloatArray:
    """Convert unit quaternions ``(N, 4)`` in ``w, x, y, z`` order to rotation matrices."""
    w, x, y, z = quaternions.T
    rotations = np.empty((quaternions.shape[0], 3, 3))
    rotations[:, 0, 0] = 1 - 2 * (y * y + z * z)
    rotations[:, 0, 1] = 2 * (x * y - w * z)
    rotations[:, 0, 2] = 2 * (x * z + w * y)
    rotations[:, 1, 0] = 2 * (x * y + w * z)
    rotations[:, 1, 1] = 1 - 2 * (x * x + z * z)
    rotations[:, 1, 2] = 2 * (y * z - w * x)
    rotations[:, 2, 0] = 2 * (x * z - w * y)
    rotations[:, 2, 1] = 2 * (y * z + w * x)
    rotations[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return rotations


def tracking_wrench(
    desired_velocities: FloatArray,
    velocities: FloatArray,
    rotations: FloatArray,
    angular_velocities: FloatArray,
    masses: FloatArray,
    inertias: FloatArray,
    max_accelerations: FloatArray,
    gains: ControlGains,
    gravity: float,
) -> FloatArray:
    """Compute ``[thrust, torque_x, torque_y, torque_z]`` for every robot.

    Args:
        desired_velocities: World-frame velocity setpoints, ``(N, 3)``.
        velocities: World-frame velocities, ``(N, 3)``.
        rotations: Body-to-world rotation matrices, ``(N, 3, 3)``.
        angular_velocities: Body-frame angular velocities, ``(N, 3)``.
        masses: ``(N,)``.
        inertias: Principal moments of inertia, ``(N, 3)``.
        max_accelerations: Acceleration limits, each below ``gravity``, ``(N,)``.
        gains: Controller gains.
        gravity: Gravitational acceleration [m/s^2].

    Returns:
        Unsaturated commands of shape ``(N, 4)``; actuator limits are applied
        by MuJoCo.
    """
    accel = clip_norm(gains.velocity * (desired_velocities - velocities), max_accelerations)
    accel[:, 2] += gravity
    force = masses[:, None] * accel

    body_z = rotations[:, :, 2]
    thrust = np.einsum("ij,ij->i", force, body_z)

    # Desired attitude: body z along the force, zero yaw. The force always has
    # a positive z component because max_acceleration < gravity, so none of
    # these normalisations can divide by zero.
    z_des = force / np.linalg.norm(force, axis=1, keepdims=True)
    y_des = np.cross(z_des, np.array([1.0, 0.0, 0.0]))
    y_des /= np.linalg.norm(y_des, axis=1, keepdims=True)
    x_des = np.cross(y_des, z_des)
    rot_des = np.stack([x_des, y_des, z_des], axis=2)

    # SO(3) error: vee(R_des^T R - R^T R_des) / 2.
    rel = np.einsum("nji,njk->nik", rot_des, rotations)
    skew = rel - np.swapaxes(rel, 1, 2)
    attitude_error = 0.5 * np.stack([skew[:, 2, 1], skew[:, 0, 2], skew[:, 1, 0]], axis=1)

    angular_accel = -gains.attitude * attitude_error - gains.angular_rate * angular_velocities
    torque = inertias * angular_accel + np.cross(angular_velocities, inertias * angular_velocities)

    command: FloatArray = np.concatenate([thrust[:, None], torque], axis=1)
    return command
