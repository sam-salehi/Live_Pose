#!/usr/bin/env python3
"""
Batch video analysis client.
Sends an entire video file to the batch server, receives all 3D pose estimates,
then plays back the video with skeleton overlay and live wrist velocity plots.
Plots remain open after playback ends.

Usage:
    python analyze_video.py --server-ip <PC_IP> --video /path/to/video.mov
"""

import sys
import os
import json
import socket
import argparse

import numpy as np
import cv2
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from live_pose3d import (
    draw_2d_skeleton,
    H36M_BONES,
    normalize_to_body_frame,
)
from protocol import send_msg, recv_msg


def render_3d_cv(joints_3d, img_size=400):
    """Render 3D skeleton with OpenCV (~1ms)."""
    img = np.zeros((img_size, img_size, 3), dtype=np.uint8)

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

    for (i, j) in H36M_BONES:
        cv2.line(img, (sx[i], sy[i]), (sx[j], sy[j]), (255, 200, 0), 2)
    for k in range(17):
        cv2.circle(img, (sx[k], sy[k]), 4, (0, 0, 255), -1)

    return img


def main():
    parser = argparse.ArgumentParser(description='Batch video analysis with velocity plots')
    parser.add_argument('--server-ip', required=True, help='Batch server IP address')
    parser.add_argument('--port', type=int, default=9001)
    parser.add_argument('--video', required=True, help='Path to video file')
    args = parser.parse_args()

    video_path = os.path.abspath(os.path.expanduser(args.video))
    if not os.path.isfile(video_path):
        print(f'Error: video file not found: {video_path}')
        sys.exit(1)

    # Read the entire video file
    with open(video_path, 'rb') as f:
        video_bytes = f.read()
    print(f'Video: {video_path} ({len(video_bytes) / 1024 / 1024:.1f} MB)')

    # Connect and send
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((args.server_ip, args.port))
    print(f'Connected to server {args.server_ip}:{args.port}')
    print('Sending video file...')
    send_msg(sock, video_bytes)
    print('Sent. Waiting for server to process (this may take a while)...')

    # Receive results
    data = recv_msg(sock)
    sock.close()

    if data is None:
        print('Error: server disconnected without sending results.')
        sys.exit(1)

    response = json.loads(data.decode('utf-8'))
    if 'error' in response and response['error']:
        print(f'Server error: {response["error"]}')
        sys.exit(1)

    video_fps = response['fps']
    all_joints = response['results']  # list of (17x3) or None
    print(f'Received {len(all_joints)} pose frames at {video_fps:.1f} FPS')

    # Pre-compute body-frame wrist velocities
    dt = 1.0 / video_fps
    left_vel = []
    right_vel = []
    prev_lw = None
    prev_rw = None

    for joints in all_joints:
        if joints is not None:
            j3d = np.array(joints, dtype=np.float32)
            body = normalize_to_body_frame(j3d)
            lw = body[13].copy()
            rw = body[16].copy()

            if prev_lw is not None:
                left_vel.append((lw - prev_lw) / dt)
            else:
                left_vel.append(np.array([0.0, 0.0, 0.0]))

            if prev_rw is not None:
                right_vel.append((rw - prev_rw) / dt)
            else:
                right_vel.append(np.array([0.0, 0.0, 0.0]))

            prev_lw = lw
            prev_rw = rw
        else:
            left_vel.append(np.array([0.0, 0.0, 0.0]))
            right_vel.append(np.array([0.0, 0.0, 0.0]))
            prev_lw = None
            prev_rw = None

    left_vel = np.array(left_vel)
    right_vel = np.array(right_vel)
    t_arr = np.arange(len(left_vel)) / video_fps

    # Set up live matplotlib plots
    plt.ion()
    fig, axes = plt.subplots(3, 2, figsize=(12, 8))
    fig.suptitle('Wrist Velocity in Body Frame', fontsize=14)
    axis_labels = ['X (lateral)', 'Y (forward)', 'Z (vertical)']

    # Plot full traces in light color, with a moving marker for current time
    lines_left = []
    lines_right = []
    markers_left = []
    markers_right = []

    for row in range(3):
        axes[row, 0].set_ylabel(f'{axis_labels[row]}\nvel (units/s)')
        axes[row, 0].grid(True, alpha=0.3)
        axes[row, 1].grid(True, alpha=0.3)

        # Full trace (light)
        axes[row, 0].plot(t_arr, left_vel[:, row], linewidth=0.6, color='tab:blue', alpha=0.3)
        axes[row, 1].plot(t_arr, right_vel[:, row], linewidth=0.6, color='tab:orange', alpha=0.3)

        # Progress line (bold, grows with playback)
        ln_l, = axes[row, 0].plot([], [], linewidth=1.2, color='tab:blue')
        ln_r, = axes[row, 1].plot([], [], linewidth=1.2, color='tab:orange')
        lines_left.append(ln_l)
        lines_right.append(ln_r)

        # Vertical time marker
        mk_l = axes[row, 0].axvline(0, color='red', linewidth=0.8, alpha=0.7)
        mk_r = axes[row, 1].axvline(0, color='red', linewidth=0.8, alpha=0.7)
        markers_left.append(mk_l)
        markers_right.append(mk_r)

        if row == 0:
            axes[row, 0].set_title('Left Wrist')
            axes[row, 1].set_title('Right Wrist')

    axes[2, 0].set_xlabel('Time (s)')
    axes[2, 1].set_xlabel('Time (s)')
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

            cv2.imshow('Video Playback', frame)
            if skeleton_img is not None:
                cv2.imshow('3D Skeleton', skeleton_img)

            # Update plot progress
            if frame_idx % PLOT_UPDATE_EVERY == 0 and frame_idx < len(left_vel):
                current_t = frame_idx / video_fps
                for row in range(3):
                    lines_left[row].set_data(t_arr[:frame_idx + 1], left_vel[:frame_idx + 1, row])
                    lines_right[row].set_data(t_arr[:frame_idx + 1], right_vel[:frame_idx + 1, row])
                    markers_left[row].set_xdata([current_t])
                    markers_right[row].set_xdata([current_t])
                fig.canvas.draw_idle()
                fig.canvas.flush_events()

            frame_idx += 1

            if cv2.waitKey(frame_delay) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

    # Final plot state: show all data, remove markers
    for row in range(3):
        lines_left[row].set_data(t_arr, left_vel[:, row])
        lines_right[row].set_data(t_arr, right_vel[:, row])
        markers_left[row].set_visible(False)
        markers_right[row].set_visible(False)

    fig.suptitle('Wrist Velocity in Body Frame (complete)', fontsize=14)
    fig.canvas.draw_idle()
    fig.canvas.flush_events()

    # Keep plots open
    plt.ioff()
    plt.show()


if __name__ == '__main__':
    main()
