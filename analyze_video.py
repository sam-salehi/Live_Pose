#!/usr/bin/env python3
"""
Batch video analysis client.
Sends an entire video file to the batch server, receives all 3D pose estimates,
then plays back the video with skeleton overlay and live wrist velocity plots.
Plots remain open after playback ends.

Usage:
    python analyze_video.py --server-ip <PC_IP> --video /path/to/video.mov
    python analyze_video.py --server-ip <PC_IP> --vid /path/to/video.mov
"""

import sys
import os
import json
import hashlib
import socket
import argparse

import numpy as np
import cv2
import matplotlib

# TkAgg + OpenCV highgui on macOS often crashes the interpreter with
# "PyEval_RestoreThread: the function must be called with the GIL held"
# when pumping both GUIs from the same loop. Use the native macOS backend.
if sys.platform == 'darwin':
    matplotlib.use('MacOSX')
else:
    matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from live_pose3d import (
    draw_2d_skeleton,
    H36M_BONES,
)
from preprocessing import extract_motion_signals
from protocol import send_msg, recv_msg


def detect_punches(speed, elbow_angle, reach, fps,
                   speed_thresh_factor=2.0,
                   min_speed=3.0,
                   min_elbow_peak=90.0,
                   window_ms=250):
    """
    Detect punch events using a multi-signal state machine.

    Parameters
    ----------
    speed : ndarray, shape (N,)
        Scalar wrist speed per frame (torso-lengths/s).
    elbow_angle : ndarray, shape (N,)
        Elbow included angle per frame (degrees).
    reach : ndarray, shape (N,)
        Wrist-to-shoulder distance per frame (torso-lengths).
    fps : float
        Video frame rate.
    speed_thresh_factor : float
        Adaptive threshold = rolling_mean + factor * rolling_std.
    min_speed : float
        Absolute minimum speed to trigger a candidate (torso-L/s).
    min_elbow_peak : float
        Minimum elbow angle peak within window to confirm punch (degrees).
    window_ms : float
        Candidate window duration after trigger (milliseconds).

    Returns
    -------
    list of dict
        Each dict has keys: 'start', 'peak', 'end' (frame indices),
        'peak_speed', 'peak_elbow'.
    """
    N = len(speed)
    if N == 0:
        return []

    window_frames = max(1, int(window_ms / 1000.0 * fps))
    rolling_window = int(2.0 * fps)  # 2-second rolling window for adaptive threshold

    punches = []

    i = 0
    while i < N:
        # Compute adaptive threshold from recent window
        win_start = max(0, i - rolling_window)
        recent_speed = speed[win_start:i + 1]
        if len(recent_speed) > 5:
            adaptive_thresh = np.mean(recent_speed) + speed_thresh_factor * np.std(recent_speed)
        else:
            adaptive_thresh = min_speed

        thresh = max(adaptive_thresh, min_speed)

        # Phase 1: Check if speed crosses threshold (candidate trigger)
        if speed[i] < thresh:
            i += 1
            continue

        # We have a candidate — define the window
        win_end = min(N, i + window_frames)
        window_speed = speed[i:win_end]
        window_elbow = elbow_angle[i:win_end]
        window_reach = reach[i:win_end]

        # Phase 2: Confirmation checks
        peak_speed_idx = np.argmax(window_speed)
        peak_speed_val = window_speed[peak_speed_idx]
        peak_elbow_val = np.max(window_elbow)

        # Check elbow extension
        if peak_elbow_val < min_elbow_peak:
            i += 1
            continue

        # Check extend-retract pattern in reach:
        # reach should increase then decrease within the window
        reach_peak_idx = np.argmax(window_reach)
        has_extend_retract = (reach_peak_idx > 0 and reach_peak_idx < len(window_reach) - 1)

        # For short windows or edge cases, also accept if reach simply increased
        # significantly from start (the retract may extend beyond the window)
        reach_increased = (np.max(window_reach) - window_reach[0]) > 0.05

        if not (has_extend_retract or reach_increased):
            i += 1
            continue

        # Confirmed punch
        start_frame = i
        peak_frame = i + int(peak_speed_idx)
        end_frame = win_end - 1

        punches.append({
            'start': start_frame,
            'peak': peak_frame,
            'end': end_frame,
            'peak_speed': float(peak_speed_val),
            'peak_elbow': float(peak_elbow_val),
        })

        i = win_end  # skip past this window

    return punches


PUNCH_COLORS = {
    'jab':      'blue',
    'hook':     'green',
    'uppercut': 'orange',
}


def classify_punch(elevation, elbow_angle, sh_yaw, start, end, fps,
                    uppercut_elev_ext=44.0,
                    hook_sh_yaw_range=11.0):
    """
    Classify a detected punch as jab, hook, or uppercut.

    Rules from ANOVA analysis of 1508 annotated punches (V4-V10):
    Jab n=489, Lead Hook n=531, Lead Uppercut n=488.

      1. Extension frame = frame of peak elbow angle in the window.
         Arm elevation at that frame (elev_at_ext) cleanly separates
         uppercuts (F=170): Jab=58°, Hook=51°, Uppercut=37°.
         → elev_at_ext < uppercut_elev_ext  →  UPPERCUT

      2. Shoulder yaw range (shoulder rotation relative to hips = xfactor).
         Hooks involve ~2x more shoulder rotation than jabs (F=138):
         Hook=15°, Jab=7°.
         → sh_yaw_range >= hook_sh_yaw_range  →  HOOK
         → otherwise                          →  JAB

    Note: reach_peak is nearly identical across all three classes
    (F=0.86, p=0.43) on the full dataset and is no longer used.

    Parameters
    ----------
    elevation : ndarray, shape (N,)
        Arm elevation per frame (degrees, 0=hanging, 90=horizontal).
    elbow_angle : ndarray, shape (N,)
        Elbow included angle per frame (degrees, 180=straight).
    sh_yaw : ndarray, shape (N,)
        Shoulder-vector angle in the hip-normalised body frame per frame
        (degrees). Equals the shoulder-hip xfactor.
    start, end : int
        Frame indices of the punch window (inclusive).
    fps : float
        Video frame rate (used for diagnostic output).
    uppercut_elev_ext : float
        Elevation threshold at the extension frame; below this → UPPERCUT.
        Default 44° (midpoint of Hook 51° and Uppercut 37°).
    hook_sh_yaw_range : float
        Shoulder yaw range threshold; at or above this → HOOK.
        Default 11° (midpoint of Jab 7° and Hook 15°).

    Returns
    -------
    str
        One of 'jab', 'hook', 'uppercut'.
    """
    win_elev  = elevation[start:end + 1]
    win_elbow = elbow_angle[start:end + 1]
    win_sh    = sh_yaw[start:end + 1]

    # Extension frame: frame of maximum elbow angle (most extended arm)
    ext_local   = int(np.argmax(win_elbow))
    elev_at_ext = float(win_elev[ext_local])
    sh_range    = float(np.max(win_sh) - np.min(win_sh))

    t = start / fps
    print(f'    [classify] t={t:.2f}s  elev_at_ext={elev_at_ext:.1f}°  '
          f'sh_yaw_range={sh_range:.1f}°', end='')

    if elev_at_ext < uppercut_elev_ext:
        print(' → UPPERCUT')
        return 'uppercut'

    if sh_range >= hook_sh_yaw_range:
        print(' → HOOK')
        return 'hook'

    print(' → JAB')
    return 'jab'

def extract_punch_features(p, speed, elbow_angle, elevation, reach,
                           forearm_dir, vel, fps):
    """Extract a comprehensive feature dict for a single punch window."""
    s, e = p['start'], p['end']
    win_speed = speed[s:e + 1]
    win_elbow = elbow_angle[s:e + 1]
    win_elev = elevation[s:e + 1]
    win_reach = reach[s:e + 1]

    # Extension frame: first frame after peak speed where speed drops
    peak_local = int(np.argmax(win_speed))
    ext_idx = peak_local
    for i in range(peak_local + 1, len(win_speed)):
        if win_speed[i] < win_speed[i - 1]:
            ext_idx = i
            break
    ext_frame = s + ext_idx

    # Forearm at extension
    fa = forearm_dir[ext_frame]
    fa_mag = np.linalg.norm(fa) + 1e-8
    fa_abs = np.abs(fa)

    # Velocity at extension
    v = vel[ext_frame]
    v_mag = np.linalg.norm(v) + 1e-8
    v_abs = np.abs(v)

    # Forearm elevation angle (degrees above horizontal)
    fa_elev_deg = np.degrees(np.arcsin(np.clip(fa[2] / fa_mag, -1, 1)))

    feats = {
        't_start':        s / fps,
        't_ext':          ext_frame / fps,
        # Forearm direction fractions at extension
        'fa_X_frac':      fa_abs[0] / fa_mag,
        'fa_Y_frac':      fa_abs[1] / fa_mag,
        'fa_Z_frac':      fa_abs[2] / fa_mag,
        'fa_elev_deg':    fa_elev_deg,
        # Raw forearm components (signed, body frame)
        'fa_X':           fa[0],
        'fa_Y':           fa[1],
        'fa_Z':           fa[2],
        # Velocity direction fractions at extension
        'vel_X_frac':     v_abs[0] / v_mag,
        'vel_Y_frac':     v_abs[1] / v_mag,
        'vel_Z_frac':     v_abs[2] / v_mag,
        # Elbow
        'elbow_at_ext':   elbow_angle[ext_frame],
        'elbow_peak':     float(np.max(win_elbow)),
        'elbow_min':      float(np.min(win_elbow)),
        # Elevation
        'elev_at_ext':    elevation[ext_frame],
        'elev_delta':     float(np.max(win_elev) - np.min(win_elev)),
        'elev_peak':      float(np.max(win_elev)),
        # Speed
        'peak_speed':     float(np.max(win_speed)),
        # Reach
        'reach_at_ext':   reach[ext_frame],
        'reach_peak':     float(np.max(win_reach)),
        'reach_delta':    float(np.max(win_reach) - np.min(win_reach)),
        # Window duration
        'duration_ms':    (e - s) / fps * 1000,
    }
    return feats


def render_3d_cv(joints_3d, img_size=400):
    """Render 3D skeleton with body-frame axes using OpenCV."""
    img = np.zeros((img_size, img_size, 3), dtype=np.uint8)

    # 3/4 view projection: swap axes so camera looks from an angle
    x = joints_3d[:, 0]
    y = joints_3d[:, 2]
    z = -joints_3d[:, 1]

    angle = np.radians(25)
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    px = x * cos_a + y * sin_a
    py = z

    scale = img_size * 0.35
    cx, cy = img_size // 2, img_size // 2
    sx = (px * scale + cx).astype(int)
    sy = (-py * scale + cy).astype(int)

    # Draw skeleton
    for (i, j) in H36M_BONES:
        cv2.line(img, (sx[i], sy[i]), (sx[j], sy[j]), (255, 200, 0), 2)
    for k in range(17):
        cv2.circle(img, (sx[k], sy[k]), 4, (0, 0, 255), -1)

    # Draw body-frame axes at hip center (joint 0)
    # Reconstruct body-frame axes in world coords
    hip_c = joints_3d[0]
    r_hip = joints_3d[1]
    l_hip = joints_3d[4]
    neck  = joints_3d[8]

    x_raw = r_hip - l_hip
    x_axis = x_raw / (np.linalg.norm(x_raw) + 1e-8)
    z_raw = neck - hip_c
    z_raw = z_raw - np.dot(z_raw, x_axis) * x_axis
    z_axis = z_raw / (np.linalg.norm(z_raw) + 1e-8)
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / (np.linalg.norm(y_axis) + 1e-8)

    # Project axis endpoints using the same 3/4 view
    axis_len = np.linalg.norm(neck - hip_c) * 0.6
    origin = hip_c
    axes_info = [
        (x_axis, (0, 0, 255),   'X'),  # red   = lateral
        (y_axis, (0, 255, 0),   'Y'),  # green = forward
        (z_axis, (255, 100, 0), 'Z'),  # blue  = vertical
    ]

    def project_pt(pt):
        px_ = pt[0] * cos_a + pt[2] * sin_a
        py_ = -pt[1]
        return (int(px_ * scale + cx), int(-py_ * scale + cy))

    o2d = project_pt(origin)
    for axis_vec, color, label in axes_info:
        tip = origin + axis_vec * axis_len
        t2d = project_pt(tip)
        cv2.arrowedLine(img, o2d, t2d, color, 2, tipLength=0.2)
        cv2.putText(img, label, (t2d[0] + 4, t2d[1] - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    return img


def main():
    parser = argparse.ArgumentParser(description='Batch video analysis with velocity plots')
    parser.add_argument('--server-ip', default=None, help='Batch server IP address')
    parser.add_argument('--port', type=int, default=9001)
    parser.add_argument(
        '--video', '--vid', dest='video', required=True, help='Path to video file'
    )
    parser.add_argument(
        '--speed', action='store_true',
        help='Plot scalar speed (magnitude) instead of X/Y/Z velocity components'
    )
    parser.add_argument('--punch-speed-factor', type=float, default=2.0,
                        help='Adaptive speed threshold factor (default: 2.0)')
    parser.add_argument('--punch-min-speed', type=float, default=3.0,
                        help='Minimum speed to trigger punch candidate (torso-L/s, default: 3.0)')
    parser.add_argument('--punch-min-elbow', type=float, default=120.0,
                        help='Minimum elbow angle peak to confirm punch (degrees, default: 120)')
    parser.add_argument('--uppercut-elev-ext', type=float, default=44.0,
                        help='Arm elevation threshold at extension for uppercut (deg, default: 44)')
    parser.add_argument('--hook-sh-yaw-range', type=float, default=11.0,
                        help='Shoulder yaw range threshold for hook (deg, default: 11)')
    args = parser.parse_args()

    video_path = os.path.abspath(os.path.expanduser(args.video))
    if not os.path.isfile(video_path):
        print(f'Error: video file not found: {video_path}')
        sys.exit(1)

    # Pose cache: hash video content to create a cache key
    cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pose_cache')
    os.makedirs(cache_dir, exist_ok=True)

    with open(video_path, 'rb') as f:
        video_bytes = f.read()
    video_hash = hashlib.sha256(video_bytes).hexdigest()[:16]
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    cache_path = os.path.join(cache_dir, f'{video_name}_{video_hash}.json')

    print(f'Video: {video_path} ({len(video_bytes) / 1024 / 1024:.1f} MB)')

    if os.path.isfile(cache_path):
        # Load cached pose data
        print(f'Loading cached pose data from: {cache_path}')
        with open(cache_path, 'r') as f:
            response = json.load(f)
    else:
        # Need server to process
        if args.server_ip is None:
            print('Error: no cached data found and --server-ip not provided.')
            sys.exit(1)

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((args.server_ip, args.port))
        print(f'Connected to server {args.server_ip}:{args.port}')
        print('Sending video file...')
        send_msg(sock, video_bytes)
        print('Sent. Waiting for server to process (this may take a while)...')

        data = recv_msg(sock)
        sock.close()

        if data is None:
            print('Error: server disconnected without sending results.')
            sys.exit(1)

        response = json.loads(data.decode('utf-8'))
        if 'error' in response and response['error']:
            print(f'Server error: {response["error"]}')
            sys.exit(1)

        # Save to cache
        with open(cache_path, 'w') as f:
            json.dump(response, f)
        print(f'Pose data cached to: {cache_path}')

    video_fps = response['fps']
    all_joints = response['results']  # list of (17x3) or None
    print(f'{len(all_joints)} pose frames at {video_fps:.1f} FPS')


    # Body-frame motion signals (normalization in preprocessing.py)
    sig = extract_motion_signals(all_joints, video_fps)
    left_vel = sig['left_vel']
    right_vel = sig['right_vel']
    left_elev = sig['left_elev']
    right_elev = sig['right_elev']
    left_elbow = sig['left_elbow']
    right_elbow = sig['right_elbow']
    left_reach = sig['left_reach']
    right_reach = sig['right_reach']
    left_forearm = sig['left_forearm']
    right_forearm = sig['right_forearm']
    sh_yaw = sig['sh_yaw']
    left_speed = sig['left_speed']
    right_speed = sig['right_speed']

    t_arr = np.arange(len(left_vel)) / video_fps

    # Detect punches
    punch_kwargs = dict(
        speed_thresh_factor=args.punch_speed_factor,
        min_speed=args.punch_min_speed,
        min_elbow_peak=args.punch_min_elbow,
    )
    left_punches = detect_punches(left_speed, left_elbow, left_reach, video_fps, **punch_kwargs)
    right_punches = detect_punches(right_speed, right_elbow, right_reach, video_fps, **punch_kwargs)

    # Extract features for all punches
    left_feat_list = []
    for p in left_punches:
        feats = extract_punch_features(p, left_speed, left_elbow, left_elev,
                                       left_reach, left_forearm, left_vel, video_fps)
        left_feat_list.append(feats)
    right_feat_list = []
    for p in right_punches:
        feats = extract_punch_features(p, right_speed, right_elbow, right_elev,
                                       right_reach, right_forearm, right_vel, video_fps)
        right_feat_list.append(feats)

    # Classify each punch
    print('  Left hand classification:')
    for p in left_punches:
        p['type'] = classify_punch(left_elev, left_elbow, sh_yaw,
                                   p['start'], p['end'], video_fps,
                                   uppercut_elev_ext=args.uppercut_elev_ext,
                                   hook_sh_yaw_range=args.hook_sh_yaw_range)
    print('  Right hand classification:')
    for p in right_punches:
        p['type'] = classify_punch(right_elev, right_elbow, sh_yaw,
                                   p['start'], p['end'], video_fps,
                                   uppercut_elev_ext=args.uppercut_elev_ext,
                                   hook_sh_yaw_range=args.hook_sh_yaw_range)

    # ── Print feature analysis table ───────────────────────────────────
    def print_feature_table(label, feat_list):
        if not feat_list:
            print(f'\n  {label}: no punches detected\n')
            return

        keys = list(feat_list[0].keys())
        # Collect all values per feature
        arrays = {k: np.array([f[k] for f in feat_list]) for k in keys}

        print(f'\n{"=" * 80}')
        print(f'  {label}  —  {len(feat_list)} punches')
        print(f'{"=" * 80}')

        # Per-punch table
        header = f'  {"#":>3}  {"t_ext":>6}'
        feat_cols = [k for k in keys if k not in ('t_start', 't_ext')]
        for k in feat_cols:
            header += f'  {k:>12}'
        print(header)
        print('  ' + '-' * (len(header) - 2))

        for i, f in enumerate(feat_list):
            row = f'  {i+1:>3}  {f["t_ext"]:>6.2f}'
            for k in feat_cols:
                row += f'  {f[k]:>12.3f}'
            print(row)

        # Summary: mean, std, cv (coefficient of variation)
        print('  ' + '-' * (len(header) - 2))
        mean_row = f'  {"avg":>3}  {"":>6}'
        std_row  = f'  {"std":>3}  {"":>6}'
        cv_row   = f'  {" cv":>3}  {"":>6}'
        for k in feat_cols:
            m = np.mean(arrays[k])
            s = np.std(arrays[k])
            cv = s / (abs(m) + 1e-8)
            mean_row += f'  {m:>12.3f}'
            std_row  += f'  {s:>12.3f}'
            cv_row   += f'  {cv:>12.2f}'
        print(mean_row)
        print(std_row)
        print(cv_row)

        # Highlight most consistent features (lowest CV)
        cvs = {}
        for k in feat_cols:
            m = np.mean(arrays[k])
            s = np.std(arrays[k])
            cvs[k] = s / (abs(m) + 1e-8)
        sorted_feats = sorted(cvs.items(), key=lambda x: x[1])

        print(f'\n  Most consistent features (lowest coefficient of variation):')
        for rank, (k, cv) in enumerate(sorted_feats[:8], 1):
            m = np.mean(arrays[k])
            s = np.std(arrays[k])
            print(f'    {rank}. {k:<18}  mean={m:>8.3f}  std={s:>7.3f}  cv={cv:.3f}')

        print(f'\n  Least consistent features (highest CV):')
        for rank, (k, cv) in enumerate(reversed(sorted_feats[-5:]), 1):
            m = np.mean(arrays[k])
            s = np.std(arrays[k])
            print(f'    {rank}. {k:<18}  mean={m:>8.3f}  std={s:>7.3f}  cv={cv:.3f}')
        print()

    print_feature_table('LEFT HAND', left_feat_list)
    print_feature_table('RIGHT HAND', right_feat_list)

    # Print punch summary
    print(f'\n--- Punch Detection ---')
    print(f'Left hand: {len(left_punches)} punches detected')
    for p in left_punches:
        t = p['peak'] / video_fps
        print(f'  {p["type"].upper()} at t={t:.2f}s (peak speed: {p["peak_speed"]:.1f} torso-L/s, '
              f'elbow peak: {p["peak_elbow"]:.0f}\u00b0)')
    print(f'Right hand: {len(right_punches)} punches detected')
    for p in right_punches:
        t = p['peak'] / video_fps
        print(f'  {p["type"].upper()} at t={t:.2f}s (peak speed: {p["peak_speed"]:.1f} torso-L/s, '
              f'elbow peak: {p["peak_elbow"]:.0f}\u00b0)')
    print()

    # Set up live matplotlib plots
    plt.ion()
    lines_left = []
    lines_right = []
    markers_left = []
    markers_right = []

    if args.speed:
        # 3 rows: speed, elevation, elbow
        NUM_ROWS = 3
        fig, axes = plt.subplots(NUM_ROWS, 2, figsize=(12, 8))
        fig.suptitle('Wrist Speed & Arm Angles in Body Frame', fontsize=14)

        # Row 0: speed
        axes[0, 0].set_ylabel('Speed\n(torso-L/s)')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 0].plot(t_arr, left_speed, linewidth=0.6, color='tab:blue', alpha=0.3)
        axes[0, 1].plot(t_arr, right_speed, linewidth=0.6, color='tab:orange', alpha=0.3)
        speed_max = max(np.max(left_speed), np.max(right_speed)) * 1.05
        axes[0, 0].set_ylim(0, speed_max)
        axes[0, 1].set_ylim(0, speed_max)

        # Row 1: arm elevation
        axes[1, 0].set_ylabel('Elevation\n(deg)')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 0].plot(t_arr, left_elev, linewidth=0.6, color='tab:blue', alpha=0.3)
        axes[1, 1].plot(t_arr, right_elev, linewidth=0.6, color='tab:orange', alpha=0.3)
        axes[1, 0].set_ylim(0, 180)
        axes[1, 1].set_ylim(0, 180)

        # Row 2: elbow angle
        axes[2, 0].set_ylabel('Elbow\n(deg)')
        axes[2, 0].grid(True, alpha=0.3)
        axes[2, 1].grid(True, alpha=0.3)
        axes[2, 0].plot(t_arr, left_elbow, linewidth=0.6, color='tab:blue', alpha=0.3)
        axes[2, 1].plot(t_arr, right_elbow, linewidth=0.6, color='tab:orange', alpha=0.3)
        axes[2, 0].set_ylim(0, 180)
        axes[2, 1].set_ylim(0, 180)

    else:
        # 5 rows: X/Y/Z velocity, elevation, elbow
        NUM_ROWS = 5
        fig, axes = plt.subplots(NUM_ROWS, 2, figsize=(12, 12))
        fig.suptitle('Wrist Velocity & Arm Angles in Body Frame', fontsize=14)
        vel_labels = ['X (lateral)', 'Y (forward)', 'Z (vertical)']

        for row in range(3):
            axes[row, 0].set_ylabel(f'{vel_labels[row]}\nvel (torso-L/s)')
            axes[row, 0].grid(True, alpha=0.3)
            axes[row, 1].grid(True, alpha=0.3)
            axes[row, 0].plot(t_arr, left_vel[:, row], linewidth=0.6, color='tab:blue', alpha=0.3)
            axes[row, 1].plot(t_arr, right_vel[:, row], linewidth=0.6, color='tab:orange', alpha=0.3)

        # Share Y axis limits across velocity subplots
        all_vel = np.concatenate([left_vel, right_vel], axis=0)
        vel_max = np.max(np.abs(all_vel)) * 1.05
        for row in range(3):
            axes[row, 0].set_ylim(-vel_max, vel_max)
            axes[row, 1].set_ylim(-vel_max, vel_max)

        # Row 3: arm elevation
        axes[3, 0].set_ylabel('Elevation\n(deg)')
        axes[3, 0].grid(True, alpha=0.3)
        axes[3, 1].grid(True, alpha=0.3)
        axes[3, 0].plot(t_arr, left_elev, linewidth=0.6, color='tab:blue', alpha=0.3)
        axes[3, 1].plot(t_arr, right_elev, linewidth=0.6, color='tab:orange', alpha=0.3)
        axes[3, 0].set_ylim(0, 180)
        axes[3, 1].set_ylim(0, 180)

        # Row 4: elbow angle
        axes[4, 0].set_ylabel('Elbow\n(deg)')
        axes[4, 0].grid(True, alpha=0.3)
        axes[4, 1].grid(True, alpha=0.3)
        axes[4, 0].plot(t_arr, left_elbow, linewidth=0.6, color='tab:blue', alpha=0.3)
        axes[4, 1].plot(t_arr, right_elbow, linewidth=0.6, color='tab:orange', alpha=0.3)
        axes[4, 0].set_ylim(0, 180)
        axes[4, 1].set_ylim(0, 180)

    # Shade detected punch regions on all rows, colored by punch type
    for p in left_punches:
        t_start = p['start'] / video_fps
        t_end = p['end'] / video_fps
        color = PUNCH_COLORS[p['type']]
        for row in range(NUM_ROWS):
            axes[row, 0].axvspan(t_start, t_end, alpha=0.15, color=color)
    for p in right_punches:
        t_start = p['start'] / video_fps
        t_end = p['end'] / video_fps
        color = PUNCH_COLORS[p['type']]
        for row in range(NUM_ROWS):
            axes[row, 1].axvspan(t_start, t_end, alpha=0.15, color=color)

    # Progress lines and markers for all rows
    for row in range(NUM_ROWS):
        ln_l, = axes[row, 0].plot([], [], linewidth=1.2, color='tab:blue')
        ln_r, = axes[row, 1].plot([], [], linewidth=1.2, color='tab:orange')
        lines_left.append(ln_l)
        lines_right.append(ln_r)

        mk_l = axes[row, 0].axvline(0, color='red', linewidth=0.8, alpha=0.7)
        mk_r = axes[row, 1].axvline(0, color='red', linewidth=0.8, alpha=0.7)
        markers_left.append(mk_l)
        markers_right.append(mk_r)

    axes[0, 0].set_title('Left')
    axes[0, 1].set_title('Right')
    axes[NUM_ROWS - 1, 0].set_xlabel('Time (s)')
    axes[NUM_ROWS - 1, 1].set_xlabel('Time (s)')

    # Add punch-type legend to top-right subplot
    from matplotlib.patches import Patch
    legend_patches = [Patch(facecolor=c, alpha=0.3, label=t.capitalize())
                      for t, c in PUNCH_COLORS.items()]
    axes[0, 1].legend(handles=legend_patches, loc='upper right', fontsize=8)

    fig.tight_layout()
    fig.show()

    # Playback: open video again and step through with pose overlay
    cap = cv2.VideoCapture(video_path)
    frame_delay = int(1000.0 / video_fps)
    frame_idx = 0
    PLOT_UPDATE_EVERY = 3  # update plots every N frames to keep playback smooth

    print('Playing back with pose overlay... Press q to stop.')

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Draw skeleton overlay
            skeleton_img = None
            if frame_idx < len(all_joints) and all_joints[frame_idx] is not None:
                j3d = np.array(all_joints[frame_idx], dtype=np.float32)
                skeleton_img = render_3d_cv(j3d)

            cv2.putText(frame, f'Frame: {frame_idx}', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow('Video Playback', frame)
            if skeleton_img is not None:
                cv2.imshow('3D Skeleton', skeleton_img)

            # Update plot progress
            if frame_idx % PLOT_UPDATE_EVERY == 0 and frame_idx < len(left_vel):
                current_t = frame_idx / video_fps
                n = frame_idx + 1
                if args.speed:
                    lines_left[0].set_data(t_arr[:n], left_speed[:n])
                    lines_right[0].set_data(t_arr[:n], right_speed[:n])
                    lines_left[1].set_data(t_arr[:n], left_elev[:n])
                    lines_right[1].set_data(t_arr[:n], right_elev[:n])
                    lines_left[2].set_data(t_arr[:n], left_elbow[:n])
                    lines_right[2].set_data(t_arr[:n], right_elbow[:n])
                else:
                    for row in range(3):
                        lines_left[row].set_data(t_arr[:n], left_vel[:n, row])
                        lines_right[row].set_data(t_arr[:n], right_vel[:n, row])
                    lines_left[3].set_data(t_arr[:n], left_elev[:n])
                    lines_right[3].set_data(t_arr[:n], right_elev[:n])
                    lines_left[4].set_data(t_arr[:n], left_elbow[:n])
                    lines_right[4].set_data(t_arr[:n], right_elbow[:n])
                for row in range(NUM_ROWS):
                    markers_left[row].set_xdata([current_t])
                    markers_right[row].set_xdata([current_t])
                fig.canvas.draw_idle()
                fig.canvas.flush_events()
                plt.pause(0.001)

            frame_idx += 1

            if cv2.waitKey(frame_delay) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

    # Final plot state: show all data, remove markers
    if args.speed:
        lines_left[0].set_data(t_arr, left_speed)
        lines_right[0].set_data(t_arr, right_speed)
        lines_left[1].set_data(t_arr, left_elev)
        lines_right[1].set_data(t_arr, right_elev)
        lines_left[2].set_data(t_arr, left_elbow)
        lines_right[2].set_data(t_arr, right_elbow)
    else:
        for row in range(3):
            lines_left[row].set_data(t_arr, left_vel[:, row])
            lines_right[row].set_data(t_arr, right_vel[:, row])
        lines_left[3].set_data(t_arr, left_elev)
        lines_right[3].set_data(t_arr, right_elev)
        lines_left[4].set_data(t_arr, left_elbow)
        lines_right[4].set_data(t_arr, right_elbow)
    for row in range(NUM_ROWS):
        markers_left[row].set_visible(False)
        markers_right[row].set_visible(False)

    title = 'Wrist Speed & Arm Angles' if args.speed else 'Wrist Velocity & Arm Angles'
    fig.suptitle(f'{title} (complete)', fontsize=14)
    fig.canvas.draw_idle()
    fig.canvas.flush_events()

    video_base = os.path.splitext(video_path)[0]
    plot_path = video_base + '_plot.png'
    fig.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f'Plot saved to: {plot_path}')

    # Keep plots open
    plt.ioff()
    plt.show()


if __name__ == '__main__':
    main()
