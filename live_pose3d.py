#!/usr/bin/env python3
"""
Live 3D Pose Estimation from Webcam
Webcam (OpenCV) -> YOLOv8-pose (2D) -> MotionBERT-Lite (3D) -> matplotlib 3D skeleton
"""

import sys
import os
import copy
from collections import deque

import numpy as np
import cv2
import torch
import torch.nn as nn
from functools import partial
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from ultralytics import YOLO
from easydict import EasyDict as edict
import yaml

# ── Add MotionBERT to path ──────────────────────────────────────────────────
MOTIONBERT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'MotionBERT')
sys.path.insert(0, MOTIONBERT_ROOT)
from lib.model.DSTformer import DSTformer

# ── Constants ────────────────────────────────────────────────────────────────
CLIP_LEN = 243
DEVICE = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')

# H36M joint names (17 joints):
#  0: Hip (pelvis/root)     1: RHip        2: RKnee       3: RAnkle
#  4: LHip                  5: LKnee       6: LAnkle      7: Spine (mid-torso)
#  8: Neck (thorax)         9: Nose/Jaw   10: Head top
# 11: LShoulder            12: LElbow     13: LWrist
# 14: RShoulder            15: RElbow     16: RWrist

H36M_BONES = [
    (0, 1), (1, 2), (2, 3),      # right leg
    (0, 4), (4, 5), (5, 6),      # left leg
    (0, 7), (7, 8), (8, 9), (9, 10),  # spine -> head
    (8, 11), (11, 12), (12, 13), # left arm
    (8, 14), (14, 15), (15, 16), # right arm
]

# COCO keypoint indices (YOLOv8-pose output):
#  0: nose      1: left_eye   2: right_eye  3: left_ear   4: right_ear
#  5: left_shoulder  6: right_shoulder  7: left_elbow  8: right_elbow
#  9: left_wrist    10: right_wrist   11: left_hip  12: right_hip
# 13: left_knee     14: right_knee    15: left_ankle 16: right_ankle

COCO_BONES = [
    (5, 7), (7, 9),    # left arm
    (6, 8), (8, 10),   # right arm
    (11, 13), (13, 15), # left leg
    (12, 14), (14, 16), # right leg
    (5, 6), (11, 12),   # shoulders, hips
    (5, 11), (6, 12),   # torso
    (0, 5), (0, 6),     # nose to shoulders
]


def coco_to_h36m(coco_kpts):
    """
    Convert COCO 17 keypoints to H36M 17 joints.
    coco_kpts: (17, 3) where last dim is (x, y, conf)
    Returns: (17, 3)  -- (x, y, conf)
    """
    h36m = np.zeros((17, 3), dtype=np.float32)

    left_hip = coco_kpts[11]
    right_hip = coco_kpts[12]
    left_shoulder = coco_kpts[5]
    right_shoulder = coco_kpts[6]

    # 0: Hip center (pelvis)
    h36m[0] = (left_hip + right_hip) / 2.0
    # 1: Right hip
    h36m[1] = right_hip
    # 2: Right knee
    h36m[2] = coco_kpts[14]
    # 3: Right ankle
    h36m[3] = coco_kpts[16]
    # 4: Left hip
    h36m[4] = left_hip
    # 5: Left knee
    h36m[5] = coco_kpts[13]
    # 6: Left ankle
    h36m[6] = coco_kpts[15]
    # 7: Spine (midpoint of hip center and neck)
    neck = (left_shoulder + right_shoulder) / 2.0
    hip_center = h36m[0]
    h36m[7] = (hip_center + neck) / 2.0
    # 8: Neck / thorax
    h36m[8] = neck
    # 9: Nose / jaw
    h36m[9] = coco_kpts[0]
    # 10: Head top (approximate: nose + offset upward)
    nose = coco_kpts[0]
    head_offset = nose - neck  # direction from neck to nose
    h36m[10] = nose + head_offset * 0.5
    h36m[10, 2] = min(nose[2], neck[2])  # confidence = min
    # 11: Left shoulder
    h36m[11] = left_shoulder
    # 12: Left elbow
    h36m[12] = coco_kpts[7]
    # 13: Left wrist
    h36m[13] = coco_kpts[9]
    # 14: Right shoulder
    h36m[14] = right_shoulder
    # 15: Right elbow
    h36m[15] = coco_kpts[8]
    # 16: Right wrist
    h36m[16] = coco_kpts[10]

    return h36m


def load_motionbert(config_path, checkpoint_path, device):
    """Load MotionBERT-Lite model."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    args = edict(config)

    model = DSTformer(
        dim_in=3, dim_out=3,
        dim_feat=args.dim_feat,
        dim_rep=args.dim_rep,
        depth=args.depth,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        maxlen=args.maxlen,
        num_joints=args.num_joints,
    )

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint['model_pos']
    # Strip 'module.' prefix if present (from DataParallel)
    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace('module.', '') if k.startswith('module.') else k
        new_state_dict[new_key] = v
    model.load_state_dict(new_state_dict, strict=True)
    model = model.to(device)
    model.eval()
    print(f'MotionBERT loaded on {device}')
    return model, args


def normalize_keypoints(kpts_buffer, frame_w, frame_h):
    """
    Normalize a buffer of H36M keypoints for MotionBERT input.
    kpts_buffer: (T, 17, 3) -- x, y, conf
    Returns: (T, 17, 3) -- normalized x, y, conf (scaled to [-1, 1])
    """
    motion = copy.deepcopy(kpts_buffer)
    scale = min(frame_w, frame_h) / 2.0
    # Center on frame center
    motion[:, :, 0] = (motion[:, :, 0] - frame_w / 2.0) / scale
    motion[:, :, 1] = (motion[:, :, 1] - frame_h / 2.0) / scale
    return motion.astype(np.float32)


def normalize_to_body_frame(joints_3d, scale_by_torso=False):
    """
    Rotate and translate 3D joints into a body-fixed coordinate frame.

    Axes (right-handed):
      X: left hip  -> right hip   (lateral)
      Z: hip center -> neck       (vertical, made orthogonal to X)
      Y: Z x X                    (forward, out of chest)

    Origin is the hip center (H36M joint 0). The frame is rebuilt from the
    pose itself, so the result is invariant to camera orientation and to the
    subject rotating around the vertical axis.

    Args:
        joints_3d: (17, 3) H36M joints, in any consistent world frame.
        scale_by_torso: if True, also divide all coordinates by the torso
            length (distance from hip center to neck) so the pose is
            scale-normalized.

    Returns:
        (17, 3) joints expressed in the body frame.
    """
    hip_c = joints_3d[0]
    r_hip = joints_3d[1]
    l_hip = joints_3d[4]
    neck  = joints_3d[8]

    # X: left hip -> right hip
    x_raw = r_hip - l_hip
    x_axis = x_raw / (np.linalg.norm(x_raw) + 1e-8)

    # Z: hip center -> neck, orthogonalized against X (Gram-Schmidt)
    z_raw = neck - hip_c
    z_raw = z_raw - np.dot(z_raw, x_axis) * x_axis
    z_axis = z_raw / (np.linalg.norm(z_raw) + 1e-8)

    # Y: Z x X (forward)
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / (np.linalg.norm(y_axis) + 1e-8)

    # Rows of R are body axes in world coords => v_body = (v_world - origin) @ R.T
    R = np.stack([x_axis, y_axis, z_axis], axis=0).astype(np.float32)

    centered = (joints_3d - hip_c[None, :]).astype(np.float32)
    normalized = centered @ R.T

    if scale_by_torso:
        torso_len = np.linalg.norm(neck - hip_c) + 1e-8
        normalized = normalized / torso_len

    return normalized


def angle_with_xz_plane(vec):
    """
    Signed angle (degrees) between a 3D vector and the XZ plane.

    The XZ plane has Y as its normal, so the angle is simply
        arcsin(vec_y / ||vec||)
    in [-90, 90]. Positive -> vector points in +Y (out of chest in the
    body frame), negative -> -Y (behind the body), 0 -> vector lies in
    the plane.
    """
    vec = np.asarray(vec, dtype=np.float32)
    norm = np.linalg.norm(vec) + 1e-8
    sin_theta = vec[1] / norm
    return float(np.degrees(np.arcsin(np.clip(sin_theta, -1.0, 1.0))))


def arm_elevation_angle(joints_3d, side='right', already_normalized=False):
    """
    Angle (degrees, [0, 180]) between the upper arm and the body's
    downward direction (-Z in the body-fixed frame built by
    `normalize_to_body_frame`).

    Independent of which way the arm is raised — only the magnitude of
    the lift matters:
          0  -> arm at rest, hanging along the spine
         90  -> arm raised horizontally in any direction
                (sideways, forward, backward)
        180  -> arm straight overhead

    Because the metric uses only the body-frame Z component (which is
    closely aligned with image-vertical, the most reliable monocular
    axis), it is far more stable than measures that depend on the depth
    direction.

    Args:
        joints_3d: (17, 3) H36M joints. By default they are normalized
            into the body frame internally.
        side: 'right' or 'left'.
        already_normalized: set True if `joints_3d` is already body-frame.
    """
    body = joints_3d if already_normalized else normalize_to_body_frame(joints_3d)
    sh_idx, el_idx = (14, 15) if side == 'right' else (11, 12)
    arm = body[el_idx] - body[sh_idx]
    norm = float(np.linalg.norm(arm)) + 1e-8
    cos_t = -arm[2] / norm  # angle with -Z (rest direction)
    return float(np.degrees(np.arccos(np.clip(cos_t, -1.0, 1.0))))


def elbow_included_angle_deg(joints_3d, side='right'):
    """
    Included angle at the elbow between upper arm and forearm (3D).

    Uses vectors from the elbow toward the shoulder and toward the wrist.
    Translation-invariant and rotation-invariant, so any consistent joint
    frame (e.g. root-relative MotionBERT output) is fine.

    Interpretation (degrees, [0, 180]):
        ~180  -> arm straight (fully extended at the elbow)
        ~90   -> right-angle bend
        small -> strongly bent / folded

    H36M indices: L 11-12-13, R 14-15-16.
    """
    if side == 'right':
        sh, el, wr = 14, 15, 16
    else:
        sh, el, wr = 11, 12, 13

    v_upper = joints_3d[sh] - joints_3d[el]
    v_fore = joints_3d[wr] - joints_3d[el]
    nu = float(np.linalg.norm(v_upper)) + 1e-8
    nf = float(np.linalg.norm(v_fore)) + 1e-8
    cos_t = float(np.dot(v_upper, v_fore)) / (nu * nf)
    return float(np.degrees(np.arccos(np.clip(cos_t, -1.0, 1.0))))


def shoulder_to_elbow_angle_with_body_plane(joints_3d, side='right',
                                            already_normalized=False):
    """
    Angle the upper arm (shoulder -> elbow) makes with the body's XZ plane
    (the coronal / frontal plane) in the body-fixed frame built by
    `normalize_to_body_frame`. Origin of that frame is the pelvis.

    Interpretation (degrees, range [-90, 90]):
          0  -> arm lies in the coronal plane
                 (e.g. hanging straight down, or out to the side at shoulder height)
        +deg -> arm reaches forward (out of the chest)
        -deg -> arm reaches backward

    Args:
        joints_3d: (17, 3) H36M joints. By default these are taken in the
            raw MotionBERT/world frame and normalized internally.
        side: 'right' or 'left'.
        already_normalized: set True if `joints_3d` is already in the body
            frame (skips re-normalization).

    Returns:
        float angle in degrees.
    """
    body = joints_3d if already_normalized else normalize_to_body_frame(joints_3d)
    sh_idx, el_idx = (14, 15) if side == 'right' else (11, 12)
    upper_arm = body[el_idx] - body[sh_idx]
    return angle_with_xz_plane(upper_arm)


def draw_2d_skeleton(frame, coco_kpts):
    """Draw COCO 2D skeleton on frame."""
    for i in range(17):
        x, y, conf = coco_kpts[i]
        if conf > 0.3:
            cv2.circle(frame, (int(x), int(y)), 4, (0, 255, 0), -1)
    for (i, j) in COCO_BONES:
        if coco_kpts[i, 2] > 0.3 and coco_kpts[j, 2] > 0.3:
            pt1 = (int(coco_kpts[i, 0]), int(coco_kpts[i, 1]))
            pt2 = (int(coco_kpts[j, 0]), int(coco_kpts[j, 1]))
            cv2.line(frame, pt1, pt2, (0, 255, 255), 2)


def update_3d_plot(ax, joints_3d):
    """Update the 3D matplotlib plot with new joint positions."""
    ax.cla()
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    ax.set_zlabel('Y (up)')
    ax.set_title('3D Pose (MotionBERT)')

    # joints_3d: (17, 3) -- x, y, z from model
    # Remap for display: X=x, Y=z (depth), Z=-y (up)
    x = joints_3d[:, 0]
    y = joints_3d[:, 2]
    z = -joints_3d[:, 1]

    ax.scatter(x, y, z, c='red', s=20)
    for (i, j) in H36M_BONES:
        ax.plot([x[i], x[j]], [y[i], y[j]], [z[i], z[j]], c='blue', linewidth=2)


def main():
    # ── Load models ──────────────────────────────────────────────────────
    print('Loading YOLOv8-pose...')
    yolo = YOLO('yolov8m-pose.pt')

    config_path = os.path.join(MOTIONBERT_ROOT, 'configs', 'pose3d', 'MB_ft_h36m_global_lite.yaml')
    ckpt_path = os.path.join(MOTIONBERT_ROOT, 'checkpoint', 'pose3d',
                             'FT_MB_lite_MB_ft_h36m_global_lite', 'best_epoch.bin')
    print('Loading MotionBERT-Lite...')
    motionbert, mb_args = load_motionbert(config_path, ckpt_path, DEVICE)

    # ── Webcam setup ─────────────────────────────────────────────────────
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print('Error: Cannot open webcam')
        sys.exit(1)

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f'Webcam: {frame_w}x{frame_h}')

    # ── Matplotlib 3D setup ──────────────────────────────────────────────
    plt.ion()
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title('Waiting for pose...')
    fig.show()

    # ── Sliding window buffer ────────────────────────────────────────────
    kpts_buffer = deque(maxlen=CLIP_LEN)

    print('Running... Press q to quit.')

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # ── YOLOv8-pose 2D detection ─────────────────────────────────
            results = yolo(frame, verbose=False)
            result = results[0]

            coco_kpts = None
            if result.keypoints is not None and len(result.keypoints) > 0:
                # Pick the most prominent person (highest confidence bbox)
                kp_data = result.keypoints.data  # (N, 17, 3)
                if kp_data.shape[0] > 0:
                    # Use the first detection (highest confidence from YOLO)
                    coco_kpts = kp_data[0].cpu().numpy()  # (17, 3)

            if coco_kpts is not None:
                # Draw 2D overlay
                draw_2d_skeleton(frame, coco_kpts)

                # Convert to H36M
                h36m_kpts = coco_to_h36m(coco_kpts)  # (17, 3)
                kpts_buffer.append(h36m_kpts)

                # Build input tensor (pad if needed)
                buf = list(kpts_buffer)
                while len(buf) < CLIP_LEN:
                    buf.insert(0, buf[0])  # replicate first frame
                buf = np.array(buf, dtype=np.float32)  # (243, 17, 3)

                # Normalize
                buf_norm = normalize_keypoints(buf, frame_w, frame_h)

                # ── MotionBERT inference ─────────────────────────────────
                input_tensor = torch.from_numpy(buf_norm).unsqueeze(0).to(DEVICE)  # (1, 243, 17, 3)
                with torch.no_grad():
                    pred_3d = motionbert(input_tensor)  # (1, 243, 17, 3)

                # Get the center frame prediction (or last frame if buffer not full)
                center_idx = min(len(kpts_buffer) - 1, CLIP_LEN // 2)
                # Map to the padded buffer index
                pad_count = CLIP_LEN - len(kpts_buffer)
                frame_idx = pad_count + center_idx
                joints_3d = pred_3d[0, frame_idx].cpu().numpy()  # (17, 3)

                # Root-relative
                joints_3d = joints_3d - joints_3d[0:1]

                # Update 3D plot
                update_3d_plot(ax, joints_3d)
                fig.canvas.draw_idle()
                fig.canvas.flush_events()

            # Show camera feed
            cv2.imshow('Webcam - 2D Pose (press q to quit)', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        plt.close('all')
        print('Done.')


if __name__ == '__main__':
    main()
