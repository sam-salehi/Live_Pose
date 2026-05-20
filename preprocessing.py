"""Pose normalization and body-frame motion signal extraction."""

import numpy as np
from scipy.signal import savgol_filter

from live_pose3d import (
    normalize_to_body_frame,
    arm_elevation_angle,
    elbow_included_angle_deg,
)

# H36M joint indices
_PELVIS = 0
_R_HIP = 1
_L_HIP = 4
_THORAX = 8
_L_SHOULDER = 11
_L_ELBOW = 12
_L_WRIST = 13
_R_SHOULDER = 14
_R_ELBOW = 15
_R_WRIST = 16


def normalize_pose(joints_3d, scale_by_torso=True):
    """
    Express a single-frame H36M pose in a body-fixed frame (per-frame axes).

    For time-series velocity, use ``poses_to_body_frame_clip`` instead so the
    reference frame is stable across frames.
    """
    return normalize_to_body_frame(joints_3d, scale_by_torso=scale_by_torso)


def stack_poses(all_joints):
    """
    Convert a pose list to (T, 17, 3), forward-filling missing frames.

    Parameters
    ----------
    all_joints : list
        Length-T list of (17, 3) arrays or None (cached MotionBERT output).
    """
    T = len(all_joints)
    poses = np.zeros((T, 17, 3), dtype=np.float32)
    last = None
    for i, joints in enumerate(all_joints):
        if joints is not None:
            last = np.asarray(joints, dtype=np.float32)
        if last is None:
            raise ValueError('First pose frame is missing; cannot build sequence.')
        poses[i] = last
    return poses


def clip_body_frame_rotation(poses):
    """
  Rotation matrix for the clip-stable (absolute) body frame.

    Rows are [X, Y, Z] unit axes from the median pose.  For pelvis-centred
    row vectors ``v``, clip coordinates are ``v @ R.T``.

    Parameters
    ----------
    poses : ndarray, shape (T, 17, 3)
        Root-relative MotionBERT joints (pelvis-centred per frame).

    Returns
    -------
    ndarray, shape (3, 3)
        ``R_clip``, or identity if the pose is degenerate.
    """
    poses = np.asarray(poses, dtype=np.float64)
    q = poses - poses[:, [_PELVIS], :]
    med = np.median(q, axis=0)

    x_raw = med[_R_SHOULDER] - med[_L_SHOULDER]
    xn = np.linalg.norm(x_raw)
    if xn < 1e-8:
        return np.eye(3, dtype=np.float32)

    x_hat = x_raw / xn
    z_raw = med[_THORAX] - med[_PELVIS]
    z_raw -= np.dot(z_raw, x_hat) * x_hat
    zn = np.linalg.norm(z_raw)
    if zn < 1e-8:
        return np.eye(3, dtype=np.float32)

    z_hat = z_raw / zn
    y_hat = np.cross(z_hat, x_hat)
    return np.stack([x_hat, y_hat, z_hat], axis=0).astype(np.float32)


def poses_to_body_frame_clip(poses):
    """
    Map a pose sequence into one clip-stable body frame (torso-normalised).

    Axes are estimated from the median pose over the clip, then applied to
    every frame.  This avoids mixing coordinate bases when differentiating
    wrist position for velocity.

    Parameters
    ----------
    poses : ndarray, shape (T, 17, 3)
        Root-relative MotionBERT joints (pelvis-centred per frame).

    Returns
    -------
    ndarray, shape (T, 17, 3)
        Joints in body frame: X=lateral, Y=forward, Z=vertical.
    """
    poses = np.asarray(poses, dtype=np.float64)
    q = poses - poses[:, [_PELVIS], :]
    R = clip_body_frame_rotation(poses).astype(np.float64)
    if np.allclose(R, np.eye(3)):
        return q.astype(np.float32)

    q = q @ R.T
    torso = float(np.median(np.linalg.norm(q[:, _THORAX] - q[:, _PELVIS], axis=-1)))
    if torso > 1e-6:
        q /= torso
    return q.astype(np.float32)


# Alias used by punch_feature_analysis
to_body_frame_clip = poses_to_body_frame_clip


def savitzky_golay_smooth(x, order=2, window_size=7):
    """Savitzky–Golay smooth a 1D sequence."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError(f'expected 1D input, got shape {x.shape}')

    w = int(window_size)
    if w % 2 == 0:
        w += 1
    p = min(int(order), w - 1)

    if x.size < w:
        return x.copy()

    return savgol_filter(x, window_length=w, polyorder=p, mode='interp')


def smooth_poses(poses, order=2, window_size=7):
    """Savitzky–Golay smooth all joints along time. ``poses``: (T, 17, 3)."""
    poses = np.asarray(poses, dtype=np.float64)
    T = poses.shape[0]
    w = int(window_size)
    if w % 2 == 0:
        w += 1
    p = min(int(order), w - 1)
    if T < w:
        return poses.copy()

    flat = poses.reshape(T, -1)
    smoothed = savgol_filter(flat, window_length=w, polyorder=p, axis=0, mode='interp')
    return smoothed.reshape(poses.shape)


def extract_motion_signals(all_joints, fps):
    """
    Build body-frame wrist velocities, arm angles, reach, forearm, and shoulder yaw.

    All signals use a **single** body frame for the clip (median-pose axes), so
    velocity is not corrupted by per-frame axis redefinition.

    Parameters
    ----------
    all_joints : list
        Length-N list of (17, 3) arrays or None (cached MotionBERT output).
    fps : float
        Video frame rate.

    Returns
    -------
    dict
        Keys: left_vel, right_vel, left_elev, right_elev, left_elbow, right_elbow,
        left_reach, right_reach, left_forearm, right_forearm, sh_yaw,
        left_speed, right_speed, body_frame (T, 17, 3).
    """
    poses = stack_poses(all_joints)
    R_clip = clip_body_frame_rotation(poses)
    body = poses_to_body_frame_clip(poses)
    T = body.shape[0]
    dt = 1.0 / fps

    left_w = body[:, _L_WRIST]
    right_w = body[:, _R_WRIST]

    # Wrist velocity in the fixed clip body frame
    left_vel = np.zeros((T, 3), dtype=np.float32)
    right_vel = np.zeros((T, 3), dtype=np.float32)
    if T > 1:
        left_vel[1:] = np.diff(left_w, axis=0) / dt
        right_vel[1:] = np.diff(right_w, axis=0) / dt

    left_elev = np.empty(T, dtype=np.float32)
    right_elev = np.empty(T, dtype=np.float32)
    left_elbow = np.empty(T, dtype=np.float32)
    right_elbow = np.empty(T, dtype=np.float32)
    left_reach = np.empty(T, dtype=np.float32)
    right_reach = np.empty(T, dtype=np.float32)
    left_forearm = np.empty((T, 3), dtype=np.float32)
    right_forearm = np.empty((T, 3), dtype=np.float32)
    sh_yaw = np.empty(T, dtype=np.float32)

    for t in range(T):
        frame = body[t]
        left_elev[t] = arm_elevation_angle(frame, side='left', already_normalized=True)
        right_elev[t] = arm_elevation_angle(frame, side='right', already_normalized=True)
        left_elbow[t] = elbow_included_angle_deg(frame, side='left')
        right_elbow[t] = elbow_included_angle_deg(frame, side='right')
        left_reach[t] = float(np.linalg.norm(frame[_L_WRIST] - frame[_L_SHOULDER]))
        right_reach[t] = float(np.linalg.norm(frame[_R_WRIST] - frame[_R_SHOULDER]))
        left_forearm[t] = frame[_L_WRIST] - frame[_L_ELBOW]
        right_forearm[t] = frame[_R_WRIST] - frame[_R_ELBOW]
        sh_vec = frame[_R_SHOULDER] - frame[_L_SHOULDER]
        sh_yaw[t] = float(np.degrees(np.arctan2(sh_vec[1], sh_vec[0])))

    # Avoid false 360° shoulder "rotation" when angle wraps
    sh_yaw = np.degrees(np.unwrap(np.radians(sh_yaw)))

    return {
        'left_vel': left_vel,
        'right_vel': right_vel,
        'left_elev': left_elev,
        'right_elev': right_elev,
        'left_elbow': left_elbow,
        'right_elbow': right_elbow,
        'left_reach': left_reach,
        'right_reach': right_reach,
        'left_forearm': left_forearm,
        'right_forearm': right_forearm,
        'sh_yaw': sh_yaw.astype(np.float32),
        'left_speed': np.linalg.norm(left_vel, axis=1),
        'right_speed': np.linalg.norm(right_vel, axis=1),
        'body_frame': body,
        'R_clip': R_clip,
    }
