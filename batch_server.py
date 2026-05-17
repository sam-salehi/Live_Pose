#!/usr/bin/env python3
"""
Batch pose inference server.
Receives an entire video file over TCP, runs YOLOv8-pose + MotionBERT on all
frames, then sends back the full array of 3D pose estimates as JSON.

Protocol:
  Client -> Server: send_msg(video_file_bytes)
  Server -> Client: send_msg(json_bytes)
    JSON: {"fps": float, "results": [ [17x3] or null, ... ]}

Usage:
    python batch_server.py --host 0.0.0.0 --port 9001
"""

import sys
import os
import json
import socket
import argparse
import tempfile
from collections import deque
from functools import partial

import numpy as np
import cv2
import torch
import torch.nn as nn
import yaml
from easydict import EasyDict as edict
from ultralytics import YOLO

from live_pose3d import (
    coco_to_h36m,
    normalize_keypoints,
    load_motionbert,
    CLIP_LEN,
    MOTIONBERT_ROOT,
)
from protocol import send_msg, recv_msg

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def process_video(video_path, yolo, motionbert):
    """Run full inference on a video file. Returns (fps, list_of_joints)."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, []

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f'  Video: {frame_w}x{frame_h}, {fps:.1f} FPS, ~{total_frames} frames')

    # First pass: extract all 2D keypoints
    all_h36m = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = yolo(frame, verbose=False)
        result = results[0]

        coco_kpts = None
        if result.keypoints is not None and len(result.keypoints) > 0:
            kp_data = result.keypoints.data
            if kp_data.shape[0] > 0:
                coco_kpts = kp_data[0].cpu().numpy()

        if coco_kpts is not None:
            h36m_kpts = coco_to_h36m(coco_kpts)
            all_h36m.append(h36m_kpts)
        else:
            all_h36m.append(None)

        frame_idx += 1
        if frame_idx % 50 == 0:
            print(f'  2D detection: {frame_idx}/{total_frames}')

    cap.release()
    print(f'  2D detection complete: {len(all_h36m)} frames')

    # Second pass: run MotionBERT with sliding window
    # Fill None gaps with nearest valid frame for the buffer
    all_joints_3d = []
    kpts_buffer = deque(maxlen=CLIP_LEN)

    for i, h36m_kpts in enumerate(all_h36m):
        if h36m_kpts is not None:
            kpts_buffer.append(h36m_kpts)

            # Build padded input
            buf = list(kpts_buffer)
            while len(buf) < CLIP_LEN:
                buf.insert(0, buf[0])
            buf = np.array(buf, dtype=np.float32)

            buf_norm = normalize_keypoints(buf, frame_w, frame_h)
            input_tensor = torch.from_numpy(buf_norm).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                pred_3d = motionbert(input_tensor)

            center_idx = min(len(kpts_buffer) - 1, CLIP_LEN // 2)
            pad_count = CLIP_LEN - len(kpts_buffer)
            fidx = pad_count + center_idx
            joints_3d = pred_3d[0, fidx].cpu().numpy()
            joints_3d = joints_3d - joints_3d[0:1]  # root-relative
            all_joints_3d.append(joints_3d.tolist())
        else:
            all_joints_3d.append(None)

        if (i + 1) % 50 == 0:
            print(f'  3D lifting: {i + 1}/{len(all_h36m)}')

    print(f'  3D lifting complete.')
    return fps, all_joints_3d


def handle_client(conn, addr, yolo, motionbert):
    """Handle a single client: receive video, process, send results."""
    print(f'Client connected: {addr}')

    try:
        # Receive the video file bytes
        data = recv_msg(conn)
        if data is None:
            print(f'Client {addr} disconnected before sending data.')
            conn.close()
            return

        print(f'  Received {len(data) / 1024 / 1024:.1f} MB video file')

        # Write to temp file for OpenCV
        with tempfile.NamedTemporaryFile(suffix='.mov', delete=False) as tmp:
            tmp.write(data)
            tmp_path = tmp.name

        try:
            fps, results = process_video(tmp_path, yolo, motionbert)
        finally:
            os.unlink(tmp_path)

        if fps is None:
            response = {'error': 'Could not open video', 'fps': 0, 'results': []}
        else:
            response = {'fps': fps, 'results': results}

        payload = json.dumps(response).encode('utf-8')
        print(f'  Sending {len(payload) / 1024 / 1024:.1f} MB response ({len(results)} frames)')
        send_msg(conn, payload)

    except (ConnectionResetError, BrokenPipeError, OSError) as e:
        print(f'  Connection error: {e}')
    finally:
        conn.close()
        print(f'Client disconnected: {addr}')


def main():
    parser = argparse.ArgumentParser(description='Batch pose inference server')
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--port', type=int, default=9001)
    args = parser.parse_args()

    print('Loading YOLOv8-pose...')
    yolo = YOLO('yolov8m-pose.pt')

    config_path = os.path.join(
        MOTIONBERT_ROOT, 'configs', 'pose3d', 'MB_ft_h36m_global_lite.yaml'
    )
    ckpt_path = os.path.join(
        MOTIONBERT_ROOT, 'checkpoint', 'pose3d',
        'FT_MB_lite_MB_ft_h36m_global_lite', 'best_epoch.bin',
    )
    print('Loading MotionBERT-Lite...')
    motionbert, mb_args = load_motionbert(config_path, ckpt_path, DEVICE)

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((args.host, args.port))
    srv.listen(1)
    print(f'Listening on {args.host}:{args.port}')
    print('Waiting for video files...')

    try:
        while True:
            conn, addr = srv.accept()
            handle_client(conn, addr, yolo, motionbert)
    except KeyboardInterrupt:
        print('\nShutting down.')
    finally:
        srv.close()


if __name__ == '__main__':
    main()
