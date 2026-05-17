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
import socket
import argparse

import numpy as np
import cv2
from scipy.signal import butter, filtfilt
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
    normalize_to_body_frame,
    arm_elevation_angle,
    elbow_included_angle_deg,
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
    parser.add_argument(
        '--video', '--vid', dest='video', required=True, help='Path to video file'
    )
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


    # Pre-compute body-frame wrist velocities and arm angles
    dt = 1.0 / video_fps
    left_vel = []
    right_vel = []
    left_elev = []
    right_elev = []
    left_elbow = []
    right_elbow = []
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

            # Arm elevation (0=hanging, 90=horizontal, 180=overhead)
            left_elev.append(arm_elevation_angle(body, side='left', already_normalized=True))
            right_elev.append(arm_elevation_angle(body, side='right', already_normalized=True))

            # Elbow included angle (180=straight, 90=right angle)
            left_elbow.append(elbow_included_angle_deg(j3d, side='left'))
            right_elbow.append(elbow_included_angle_deg(j3d, side='right'))
        else:
            left_vel.append(np.array([0.0, 0.0, 0.0]))
            right_vel.append(np.array([0.0, 0.0, 0.0]))
            left_elev.append(0.0)
            right_elev.append(0.0)
            left_elbow.append(0.0)
            right_elbow.append(0.0)
            prev_lw = None
            prev_rw = None

    left_vel_raw = np.array(left_vel)
    right_vel_raw = np.array(right_vel)
    left_elev_raw = np.array(left_elev)
    right_elev_raw = np.array(right_elev)
    left_elbow_raw = np.array(left_elbow)
    right_elbow_raw = np.array(right_elbow)

    # Butterworth low-pass filter: 2nd order, 6 Hz cutoff, zero-phase
    cutoff_hz = 6.0
    filter_order = 2
    b, a = butter(filter_order, cutoff_hz, btype='low', fs=video_fps)
    if len(left_vel_raw) > 3 * max(len(b), len(a)):
        left_vel = filtfilt(b, a, left_vel_raw, axis=0)
        right_vel = filtfilt(b, a, right_vel_raw, axis=0)
        left_elev = filtfilt(b, a, left_elev_raw)
        right_elev = filtfilt(b, a, right_elev_raw)
        left_elbow = filtfilt(b, a, left_elbow_raw)
        right_elbow = filtfilt(b, a, right_elbow_raw)
    else:
        left_vel = left_vel_raw
        right_vel = right_vel_raw
        left_elev = left_elev_raw
        right_elev = right_elev_raw
        left_elbow = left_elbow_raw
        right_elbow = right_elbow_raw

    t_arr = np.arange(len(left_vel)) / video_fps

    # Set up live matplotlib plots: 5 rows x 2 columns
    # Rows 0-2: wrist velocity X/Y/Z, Row 3: arm elevation, Row 4: elbow angle
    plt.ion()
    NUM_ROWS = 5
    fig, axes = plt.subplots(NUM_ROWS, 2, figsize=(12, 12))
    fig.suptitle('Wrist Velocity & Arm Angles in Body Frame', fontsize=14)
    vel_labels = ['X (lateral)', 'Y (forward)', 'Z (vertical)']

    # Plot full traces in light color, with a moving marker for current time
    lines_left = []
    lines_right = []
    markers_left = []
    markers_right = []

    # Rows 0-2: velocity
    for row in range(3):
        axes[row, 0].set_ylabel(f'{vel_labels[row]}\nvel (units/s)')
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

    # Progress lines and markers for all 5 rows
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
                # Pump the Matplotlib GUI without blocking playback (helps macOS + OpenCV).
                plt.pause(0.001)

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
    lines_left[3].set_data(t_arr, left_elev)
    lines_right[3].set_data(t_arr, right_elev)
    lines_left[4].set_data(t_arr, left_elbow)
    lines_right[4].set_data(t_arr, right_elbow)
    for row in range(NUM_ROWS):
        markers_left[row].set_visible(False)
        markers_right[row].set_visible(False)

    fig.suptitle('Wrist Velocity & Arm Angles (complete)', fontsize=14)
    fig.canvas.draw_idle()
    fig.canvas.flush_events()

    # Save filtered plot as PNG
    video_base = os.path.splitext(video_path)[0]
    plot_path = video_base + '_filtered.png'
    fig.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f'Filtered plot saved to: {plot_path}')

    # Save unfiltered plot as a separate PNG
    fig_raw, axes_raw = plt.subplots(NUM_ROWS, 2, figsize=(12, 12))
    fig_raw.suptitle('Wrist Velocity & Arm Angles (unfiltered)', fontsize=14)

    for row in range(3):
        axes_raw[row, 0].set_ylabel(f'{vel_labels[row]}\nvel (units/s)')
        axes_raw[row, 0].grid(True, alpha=0.3)
        axes_raw[row, 1].grid(True, alpha=0.3)
        axes_raw[row, 0].plot(t_arr, left_vel_raw[:, row], linewidth=0.8, color='tab:blue')
        axes_raw[row, 1].plot(t_arr, right_vel_raw[:, row], linewidth=0.8, color='tab:orange')
        axes_raw[row, 0].set_ylim(-vel_max, vel_max)
        axes_raw[row, 1].set_ylim(-vel_max, vel_max)

    axes_raw[3, 0].set_ylabel('Elevation\n(deg)')
    axes_raw[3, 0].grid(True, alpha=0.3)
    axes_raw[3, 1].grid(True, alpha=0.3)
    axes_raw[3, 0].plot(t_arr, left_elev_raw, linewidth=0.8, color='tab:blue')
    axes_raw[3, 1].plot(t_arr, right_elev_raw, linewidth=0.8, color='tab:orange')
    axes_raw[3, 0].set_ylim(0, 180)
    axes_raw[3, 1].set_ylim(0, 180)

    axes_raw[4, 0].set_ylabel('Elbow\n(deg)')
    axes_raw[4, 0].grid(True, alpha=0.3)
    axes_raw[4, 1].grid(True, alpha=0.3)
    axes_raw[4, 0].plot(t_arr, left_elbow_raw, linewidth=0.8, color='tab:blue')
    axes_raw[4, 1].plot(t_arr, right_elbow_raw, linewidth=0.8, color='tab:orange')
    axes_raw[4, 0].set_ylim(0, 180)
    axes_raw[4, 1].set_ylim(0, 180)

    axes_raw[0, 0].set_title('Left')
    axes_raw[0, 1].set_title('Right')
    axes_raw[NUM_ROWS - 1, 0].set_xlabel('Time (s)')
    axes_raw[NUM_ROWS - 1, 1].set_xlabel('Time (s)')
    fig_raw.tight_layout()

    plot_raw_path = video_base + '_unfiltered.png'
    fig_raw.savefig(plot_raw_path, dpi=150, bbox_inches='tight')
    print(f'Unfiltered plot saved to: {plot_raw_path}')

    # Keep plots open
    plt.ioff()
    plt.show()


if __name__ == '__main__':
    main()
