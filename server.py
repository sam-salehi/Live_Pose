#!/usr/bin/env python3
"""
Remote pose inference server.
Receives JPEG frames over TCP, runs YOLOv8-pose + MotionBERT on GPU,
sends back 3D joints and 2D keypoints as JSON.
Also displays the received video stream and 3D skeleton locally.

Usage:
    python server.py --host 0.0.0.0 --port 9000
    python server.py --host 0.0.0.0 --port 9000 --no-display
"""

import sys
import os
import json
import socket
import argparse
import threading
from collections import deque
from functools import partial

import numpy as np
import cv2
import torch
import torch.nn as nn
import yaml
from easydict import EasyDict as edict
from ultralytics import YOLO

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Reuse helpers from live_pose3d
from live_pose3d import (
    coco_to_h36m,
    normalize_keypoints,
    load_motionbert,
    draw_2d_skeleton,
    update_3d_plot,
    CLIP_LEN,
    MOTIONBERT_ROOT,
)
from protocol import send_msg, recv_msg, unpack_frame

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Shared display state — written by client handler, read by main thread
display_lock = threading.Lock()
display_state = {
    'frame': None,        # latest raw decoded frame (updated immediately)
    'overlay_frame': None, # frame with 2D skeleton drawn on it
    'skeleton_img': None,  # rendered 3D skeleton
}


def render_3d_to_image(fig, ax, joints_3d):
    """Render 3D skeleton to a BGR numpy image via the Agg backend."""
    update_3d_plot(ax, joints_3d)
    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    img = np.asarray(buf)
    return cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)


def handle_client(conn, addr, yolo, motionbert, mb_args, show_display):
    """Handle a single client connection."""
    print(f'Client connected: {addr}')
    kpts_buffer = deque(maxlen=CLIP_LEN)

    # Per-client matplotlib figure (Agg, thread-safe)
    fig = plt.figure(figsize=(5, 5), dpi=100)
    ax = fig.add_subplot(111, projection='3d')

    try:
        while True:
            # Drain to most recent frame (drop stale ones)
            data = recv_msg(conn)
            if data is None:
                break

            conn.setblocking(False)
            while True:
                try:
                    newer = recv_msg(conn)
                    if newer is None:
                        data = None
                        break
                    data = newer
                except BlockingIOError:
                    break
            conn.setblocking(True)

            if data is None:
                break

            # Decode frame
            frame_w, frame_h, jpeg_bytes = unpack_frame(data)
            frame = cv2.imdecode(
                np.frombuffer(jpeg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR
            )
            if frame is None:
                continue

            # Immediately update display with the raw frame so the video
            # stream stays smooth regardless of inference speed.
            if show_display:
                with display_lock:
                    display_state['frame'] = frame.copy()

            # YOLOv8-pose detection
            results = yolo(frame, verbose=False)
            result = results[0]

            coco_kpts = None
            if result.keypoints is not None and len(result.keypoints) > 0:
                kp_data = result.keypoints.data  # (N, 17, 3)
                if kp_data.shape[0] > 0:
                    coco_kpts = kp_data[0].cpu().numpy()  # (17, 3)

            response = {}
            joints_3d = None

            if coco_kpts is not None:
                response['coco_keypoints'] = coco_kpts.tolist()

                # Draw 2D overlay on frame for display
                if show_display:
                    draw_2d_skeleton(frame, coco_kpts)

                # Convert to H36M and buffer
                h36m_kpts = coco_to_h36m(coco_kpts)
                kpts_buffer.append(h36m_kpts)

                # Build padded input
                buf = list(kpts_buffer)
                while len(buf) < CLIP_LEN:
                    buf.insert(0, buf[0])
                buf = np.array(buf, dtype=np.float32)

                # Normalize
                buf_norm = normalize_keypoints(buf, frame_w, frame_h)

                # MotionBERT inference
                input_tensor = torch.from_numpy(buf_norm).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    pred_3d = motionbert(input_tensor)

                center_idx = min(len(kpts_buffer) - 1, CLIP_LEN // 2)
                pad_count = CLIP_LEN - len(kpts_buffer)
                frame_idx = pad_count + center_idx
                joints_3d = pred_3d[0, frame_idx].cpu().numpy()

                # Root-relative
                joints_3d = joints_3d - joints_3d[0:1]
                response['joints_3d'] = joints_3d.tolist()
            else:
                response['coco_keypoints'] = None
                response['joints_3d'] = None

            # Update display with inference results (overlay + 3D)
            if show_display:
                skeleton_img = None
                if joints_3d is not None:
                    skeleton_img = render_3d_to_image(fig, ax, joints_3d)
                with display_lock:
                    display_state['overlay_frame'] = frame.copy()
                    display_state['skeleton_img'] = skeleton_img

            # Send JSON response
            payload = json.dumps(response).encode('utf-8')
            send_msg(conn, payload)

    except (ConnectionResetError, BrokenPipeError):
        pass
    finally:
        plt.close(fig)
        conn.close()
        # Clear display on disconnect
        if show_display:
            with display_lock:
                display_state['frame'] = None
                display_state['overlay_frame'] = None
                display_state['skeleton_img'] = None
        print(f'Client disconnected: {addr}')


def main():
    parser = argparse.ArgumentParser(description='Remote pose inference server')
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--port', type=int, default=9000)
    parser.add_argument('--no-display', action='store_true',
                        help='Disable local display windows')
    args = parser.parse_args()

    show_display = not args.no_display

    # Load models
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

    # Start TCP server
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((args.host, args.port))
    srv.listen(1)
    print(f'Listening on {args.host}:{args.port}')

    if show_display:
        print('Display enabled. Press q in any window to quit.')

    try:
        # Accept clients in a background thread so main thread can run display
        def accept_loop():
            try:
                while True:
                    conn, addr = srv.accept()
                    t = threading.Thread(
                        target=handle_client,
                        args=(conn, addr, yolo, motionbert, mb_args, show_display),
                        daemon=True,
                    )
                    t.start()
            except OSError:
                pass  # socket closed

        accept_thread = threading.Thread(target=accept_loop, daemon=True)
        accept_thread.start()

        if show_display:
            # Main thread drives OpenCV display (required on macOS/Linux)
            while True:
                with display_lock:
                    frame = display_state['frame']
                    overlay = display_state['overlay_frame']
                    skeleton_img = display_state['skeleton_img']

                if frame is not None:
                    cv2.imshow('Server - Received Stream', frame)
                if overlay is not None:
                    cv2.imshow('Server - 2D Overlay', overlay)
                if skeleton_img is not None:
                    cv2.imshow('Server - 3D Skeleton', skeleton_img)

                key = cv2.waitKey(10) & 0xFF
                if key == ord('q'):
                    break
            cv2.destroyAllWindows()
        else:
            # No display — just block on accept thread
            accept_thread.join()

    except KeyboardInterrupt:
        print('\nShutting down.')
    finally:
        srv.close()


if __name__ == '__main__':
    main()
