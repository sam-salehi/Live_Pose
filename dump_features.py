#!/usr/bin/env python3
"""
Dump per-punch feature profiles from cached pose data. No GUI.

Usage:
    python dump_features.py --video vids/1.mov
"""
import sys
import os
import json
import hashlib
import argparse

import numpy as np
from scipy.signal import butter, filtfilt

from live_pose3d import (
    normalize_to_body_frame,
    arm_elevation_angle,
    elbow_included_angle_deg,
)
from analyze_video import detect_punches, extract_punch_features, classify_punch


def main():
    parser = argparse.ArgumentParser(description='Dump punch features from cached data')
    parser.add_argument('--video', '--vid', dest='video', required=True)
    parser.add_argument('--punch-speed-factor', type=float, default=2.0)
    parser.add_argument('--punch-min-speed', type=float, default=3.0)
    parser.add_argument('--punch-min-elbow', type=float, default=120.0)
    parser.add_argument('--punch-cooldown', type=float, default=200.0)
    args = parser.parse_args()

    video_path = os.path.abspath(os.path.expanduser(args.video))
    if not os.path.isfile(video_path):
        print(f'Error: video not found: {video_path}')
        sys.exit(1)

    # Find cache
    cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pose_cache')
    with open(video_path, 'rb') as f:
        video_bytes = f.read()
    video_hash = hashlib.sha256(video_bytes).hexdigest()[:16]
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    cache_path = os.path.join(cache_dir, f'{video_name}_{video_hash}.json')

    if not os.path.isfile(cache_path):
        print(f'Error: no cached data found at {cache_path}')
        print('Run analyze_video.py with --server-ip first to generate cache.')
        sys.exit(1)

    with open(cache_path, 'r') as f:
        response = json.load(f)

    video_fps = response['fps']
    all_joints = response['results']
    print(f'Video: {video_path}')
    print(f'{len(all_joints)} frames at {video_fps:.1f} FPS')

    # ── Replicate signal extraction from analyze_video.py ──
    dt = 1.0 / video_fps
    left_vel = []
    right_vel = []
    left_elev = []
    right_elev = []
    left_elbow = []
    right_elbow = []
    left_reach = []
    right_reach = []
    left_forearm = []
    right_forearm = []
    prev_lw = None
    prev_rw = None

    for joints in all_joints:
        if joints is not None:
            j3d = np.array(joints, dtype=np.float32)
            body = normalize_to_body_frame(j3d, scale_by_torso=True)
            lw = body[13].copy()
            rw = body[16].copy()

            left_vel.append((lw - prev_lw) / dt if prev_lw is not None else np.zeros(3))
            right_vel.append((rw - prev_rw) / dt if prev_rw is not None else np.zeros(3))
            prev_lw = lw
            prev_rw = rw

            left_elev.append(arm_elevation_angle(body, side='left', already_normalized=True))
            right_elev.append(arm_elevation_angle(body, side='right', already_normalized=True))
            left_elbow.append(elbow_included_angle_deg(j3d, side='left'))
            right_elbow.append(elbow_included_angle_deg(j3d, side='right'))
            left_reach.append(np.linalg.norm(body[13] - body[11]))
            right_reach.append(np.linalg.norm(body[16] - body[14]))
            left_forearm.append(body[13] - body[12])
            right_forearm.append(body[16] - body[15])
        else:
            left_vel.append(np.zeros(3))
            right_vel.append(np.zeros(3))
            left_elev.append(0.0)
            right_elev.append(0.0)
            left_elbow.append(0.0)
            right_elbow.append(0.0)
            left_reach.append(0.0)
            right_reach.append(0.0)
            left_forearm.append(np.zeros(3))
            right_forearm.append(np.zeros(3))
            prev_lw = None
            prev_rw = None

    left_vel_raw = np.array(left_vel)
    right_vel_raw = np.array(right_vel)
    left_elev_raw = np.array(left_elev)
    right_elev_raw = np.array(right_elev)
    left_elbow_raw = np.array(left_elbow)
    right_elbow_raw = np.array(right_elbow)
    left_reach_raw = np.array(left_reach)
    right_reach_raw = np.array(right_reach)
    left_forearm = np.array(left_forearm)
    right_forearm = np.array(right_forearm)

    cutoff_hz = 6.0
    b, a = butter(2, cutoff_hz, btype='low', fs=video_fps)
    if len(left_vel_raw) > 3 * max(len(b), len(a)):
        left_vel = filtfilt(b, a, left_vel_raw, axis=0)
        right_vel = filtfilt(b, a, right_vel_raw, axis=0)
        left_elev = filtfilt(b, a, left_elev_raw)
        right_elev = filtfilt(b, a, right_elev_raw)
        left_elbow = filtfilt(b, a, left_elbow_raw)
        right_elbow = filtfilt(b, a, right_elbow_raw)
        left_reach_f = filtfilt(b, a, left_reach_raw)
        right_reach_f = filtfilt(b, a, right_reach_raw)
    else:
        left_vel = left_vel_raw
        right_vel = right_vel_raw
        left_elev = left_elev_raw
        right_elev = right_elev_raw
        left_elbow = left_elbow_raw
        right_elbow = right_elbow_raw
        left_reach_f = left_reach_raw
        right_reach_f = right_reach_raw

    left_speed = np.linalg.norm(left_vel, axis=1)
    right_speed = np.linalg.norm(right_vel, axis=1)

    # Detect punches
    punch_kwargs = dict(
        speed_thresh_factor=args.punch_speed_factor,
        min_speed=args.punch_min_speed,
        min_elbow_peak=args.punch_min_elbow,
        cooldown_ms=args.punch_cooldown,
    )
    left_punches = detect_punches(left_speed, left_elbow, left_reach_f, video_fps, **punch_kwargs)
    right_punches = detect_punches(right_speed, right_elbow, right_reach_f, video_fps, **punch_kwargs)

    # Extract features and classify
    left_feat_list = []
    print('  Left hand classification:')
    for p in left_punches:
        feats = extract_punch_features(p, left_speed, left_elbow, left_elev,
                                       left_reach_f, left_forearm, left_vel, video_fps)
        p['type'] = classify_punch(left_elev, left_reach_f,
                                   p['start'], p['end'], video_fps)
        feats['classified_as'] = p['type']
        left_feat_list.append(feats)
    right_feat_list = []
    print('  Right hand classification:')
    for p in right_punches:
        feats = extract_punch_features(p, right_speed, right_elbow, right_elev,
                                       right_reach_f, right_forearm, right_vel, video_fps)
        p['type'] = classify_punch(right_elev, right_reach_f,
                                   p['start'], p['end'], video_fps)
        feats['classified_as'] = p['type']
        right_feat_list.append(feats)

    # ── Print feature tables ──
    def print_feature_table(label, feat_list):
        if not feat_list:
            print(f'\n  {label}: no punches detected\n')
            return

        keys = list(feat_list[0].keys())
        arrays = {k: np.array([f[k] for f in feat_list]) for k in keys}

        print(f'\n{"=" * 80}')
        print(f'  {label}  —  {len(feat_list)} punches')
        print(f'{"=" * 80}')

        # Per-punch rows
        feat_cols = [k for k in keys if k not in ('t_start', 't_ext')]
        header = f'  {"#":>3}  {"t_ext":>6}'
        for k in feat_cols:
            header += f'  {k:>12}'
        print(header)
        print('  ' + '-' * (len(header) - 2))

        for i, f in enumerate(feat_list):
            row = f'  {i+1:>3}  {f["t_ext"]:>6.2f}'
            for k in feat_cols:
                row += f'  {f[k]:>12.3f}'
            print(row)

        # Summary
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

        # Rank by consistency
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


if __name__ == '__main__':
    main()
