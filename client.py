#!/usr/bin/env python3
"""
Remote pose inference client.
Captures webcam, sends JPEG frames to the server, receives 3D joints + 2D
keypoints, and displays the results locally.

Usage:
    python client.py --server-ip <PC_IP> --port 9000
    python client.py --server-ip <PC_IP> --video /path/to/video.mp4
"""

import sys
import os
import json
import socket
import argparse
import threading
import time

import numpy as np
import cv2

from live_pose3d import (
    draw_2d_skeleton,
    H36M_BONES,
    normalize_to_body_frame,
    arm_elevation_angle,
    elbow_included_angle_deg,
)
from protocol import send_msg, recv_msg, pack_frame

# Smoothing factor for the displayed shoulder angles (0 = no update, 1 = no smoothing).
ANGLE_EMA_ALPHA = 0.25


def draw_corner_label(frame, text, corner='top-left', y_row=0,
                      color=(0, 255, 255), scale=0.8, thickness=2,
                      margin=10, row_height=35):
    """
    Draw `text` anchored to a corner of `frame`. For top corners, `y_row`
    stacks downward; for bottom corners, rows stack upward from the bottom.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    h, w = frame.shape[:2]
    y = margin + th + y_row * row_height
    if corner == 'top-left':
        x = margin
    elif corner == 'top-right':
        x = w - tw - margin
    elif corner == 'bottom-left':
        x = margin
        y = h - margin - y_row * row_height
    elif corner == 'bottom-right':
        x = w - tw - margin
        y = h - margin - y_row * row_height
    else:
        raise ValueError(f'Unsupported corner: {corner}')
    cv2.putText(frame, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)


def render_3d_cv(joints_3d, img_size=500):
    """
    Render 3D skeleton to a BGR image using OpenCV drawing (~1ms).
    Uses a simple rotated orthographic projection for a 3/4 view.
    """
    img = np.zeros((img_size, img_size, 3), dtype=np.uint8)

    # Remap axes for display: X=x, Y=z (depth), Z=-y (up) — same as the
    # old matplotlib view.
    x = joints_3d[:, 0]
    y = joints_3d[:, 2]
    z = -joints_3d[:, 1]

    # Simple 3/4 rotation around vertical for depth cue
    angle = np.radians(25)
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    px = x * cos_a + y * sin_a
    py = z  # vertical stays vertical

    # Scale and center
    scale = img_size * 0.35
    cx, cy = img_size // 2, img_size // 2
    sx = (px * scale + cx).astype(int)
    sy = (-py * scale + cy).astype(int)  # flip so +Z is up

    # Draw bones
    for (i, j) in H36M_BONES:
        cv2.line(img, (sx[i], sy[i]), (sx[j], sy[j]), (255, 200, 0), 2)

    # Draw joints
    for k in range(17):
        cv2.circle(img, (sx[k], sy[k]), 4, (0, 0, 255), -1)

    return img


def main():
    parser = argparse.ArgumentParser(description='Remote pose inference client')
    parser.add_argument('--server-ip', required=True, help='Server IP address')
    parser.add_argument('--port', type=int, default=9000)
    parser.add_argument('--jpeg-quality', type=int, default=80)
    parser.add_argument('--debug', action='store_true',
                        help='Show body-frame arm components for angle debugging')
    parser.add_argument(
        '--video', metavar='PATH', default=None,
        help='If set, run on this video file (.mp4, etc.) instead of the webcam',
    )
    args = parser.parse_args()

    # Connect to server
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((args.server_ip, args.port))
    print(f'Connected to server {args.server_ip}:{args.port}')

    if args.video:
        video_path = os.path.abspath(os.path.expanduser(args.video))
        if not os.path.isfile(video_path):
            print(f'Error: video file not found: {video_path}')
            sys.exit(1)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f'Error: cannot open video: {video_path}')
            sys.exit(1)
        ret, probe = cap.read()
        if not ret or probe is None:
            print(f'Error: cannot read frames from: {video_path}')
            cap.release()
            sys.exit(1)
        frame_h, frame_w = probe.shape[:2]
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        print(f'Video: {video_path}')
        print(f'Size: {frame_w}x{frame_h}')
    else:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print('Error: Cannot open webcam')
            sys.exit(1)
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f'Webcam: {frame_w}x{frame_h}')

    # Shared state between threads
    latest_result = {'coco_keypoints': None, 'joints_3d': None}
    result_lock = threading.Lock()
    running = True

    def receiver_thread():
        """Background thread that receives results from server."""
        nonlocal running
        try:
            while running:
                data = recv_msg(sock)
                if data is None:
                    print('Server disconnected.')
                    running = False
                    break
                response = json.loads(data.decode('utf-8'))
                with result_lock:
                    latest_result['coco_keypoints'] = (
                        np.array(response['coco_keypoints'], dtype=np.float32)
                        if response.get('coco_keypoints') is not None
                        else None
                    )
                    latest_result['joints_3d'] = (
                        np.array(response['joints_3d'], dtype=np.float32)
                        if response.get('joints_3d') is not None
                        else None
                    )
        except (ConnectionResetError, BrokenPipeError, OSError):
            running = False

    recv_t = threading.Thread(target=receiver_thread, daemon=True)
    recv_t.start()

    fps_counter = 0
    fps_time = time.time()
    fps_display = 0.0
    skeleton_img = None

    # Smoothed elevation angles (degrees).
    ema_left_angle = None
    ema_right_angle = None
    ema_left_elbow = None
    ema_right_elbow = None

    # Debug: latest body-frame upper-arm vectors.
    last_left_arm_body = None
    last_right_arm_body = None

    # Hip-frame wrist positions (body frame) for finite-difference velocity.
    prev_left_wrist_hip = None
    prev_right_wrist_hip = None
    delta_left_hip = None   # (dx, dy, dz) per frame, difference method
    delta_right_hip = None

    try:
        while running:
            ret, frame = cap.read()
            if not ret:
                if args.video:
                    print('End of video.')
                break

            # JPEG encode and send
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality]
            _, jpeg_buf = cv2.imencode('.jpg', frame, encode_params)
            jpeg_bytes = jpeg_buf.tobytes()

            try:
                payload = pack_frame(frame_w, frame_h, jpeg_bytes)
                send_msg(sock, payload)
            except (BrokenPipeError, OSError):
                print('Connection lost.')
                break

            # Draw latest result
            with result_lock:
                coco_kpts = latest_result['coco_keypoints']
                joints_3d = latest_result['joints_3d']

            if coco_kpts is not None:
                draw_2d_skeleton(frame, coco_kpts)

            if joints_3d is not None:
                skeleton_img = render_3d_cv(joints_3d)

                # Elevation angle: 0 = arm hanging, 90 = horizontal in any
                # direction, 180 = straight overhead. Uses body-frame Z, which
                # is mostly aligned with image-vertical (robust axis).
                body = normalize_to_body_frame(joints_3d)
                last_left_arm_body = body[12] - body[11]   # L elbow - L shoulder
                last_right_arm_body = body[15] - body[14]  # R elbow - R shoulder
                left_raw = arm_elevation_angle(
                    body, side='left', already_normalized=True
                )
                right_raw = arm_elevation_angle(
                    body, side='right', already_normalized=True
                )
                ema_left_angle = (
                    left_raw if ema_left_angle is None
                    else ANGLE_EMA_ALPHA * left_raw
                         + (1.0 - ANGLE_EMA_ALPHA) * ema_left_angle
                )
                ema_right_angle = (
                    right_raw if ema_right_angle is None
                    else ANGLE_EMA_ALPHA * right_raw
                         + (1.0 - ANGLE_EMA_ALPHA) * ema_right_angle
                )

                # Elbow bend: included angle upper arm vs forearm (180 = straight).
                le_raw = elbow_included_angle_deg(joints_3d, side='left')
                re_raw = elbow_included_angle_deg(joints_3d, side='right')
                ema_left_elbow = (
                    le_raw if ema_left_elbow is None
                    else ANGLE_EMA_ALPHA * le_raw
                         + (1.0 - ANGLE_EMA_ALPHA) * ema_left_elbow
                )
                ema_right_elbow = (
                    re_raw if ema_right_elbow is None
                    else ANGLE_EMA_ALPHA * re_raw
                         + (1.0 - ANGLE_EMA_ALPHA) * ema_right_elbow
                )

                # Hand motion in hip frame: Δ = wrist(t) − wrist(t−1) in body axes
                # (X lateral, Y forward, Z up — same as normalize_to_body_frame).
                lw = body[13].copy()
                rw = body[16].copy()
                if prev_left_wrist_hip is not None:
                    delta_left_hip = lw - prev_left_wrist_hip
                if prev_right_wrist_hip is not None:
                    delta_right_hip = rw - prev_right_wrist_hip
                prev_left_wrist_hip = lw
                prev_right_wrist_hip = rw
            else:
                prev_left_wrist_hip = None
                prev_right_wrist_hip = None
                delta_left_hip = None
                delta_right_hip = None

            # FPS counter
            fps_counter += 1
            elapsed = time.time() - fps_time
            if elapsed >= 1.0:
                fps_display = fps_counter / elapsed
                fps_counter = 0
                fps_time = time.time()

            # Overlay: FPS + left shoulder angle in top-left,
            # right shoulder angle in top-right.
            draw_corner_label(
                frame, f'FPS: {fps_display:.1f}',
                corner='top-left', y_row=0,
                color=(0, 255, 0), scale=1.0,
            )
            left_txt = (
                f'L Elev: {ema_left_angle:5.1f} deg'
                if ema_left_angle is not None else 'L Elev:   -- deg'
            )
            right_txt = (
                f'R Elev: {ema_right_angle:5.1f} deg'
                if ema_right_angle is not None else 'R Elev:   -- deg'
            )
            draw_corner_label(
                frame, left_txt, corner='top-left', y_row=1,
                scale=1.4, thickness=3, row_height=55,
            )
            draw_corner_label(
                frame, right_txt, corner='top-right', y_row=0,
                scale=1.4, thickness=3, row_height=55,
            )
            left_elbow_txt = (
                f'L Elbow: {ema_left_elbow:5.1f} deg'
                if ema_left_elbow is not None else 'L Elbow:   -- deg'
            )
            right_elbow_txt = (
                f'R Elbow: {ema_right_elbow:5.1f} deg'
                if ema_right_elbow is not None else 'R Elbow:   -- deg'
            )
            draw_corner_label(
                frame, left_elbow_txt, corner='top-left', y_row=2,
                scale=1.4, thickness=3, row_height=55,
            )
            draw_corner_label(
                frame, right_elbow_txt, corner='top-right', y_row=1,
                scale=1.4, thickness=3, row_height=55,
            )

            def _fmt_dxyz(d):
                if d is None:
                    return 'dX=---- dY=---- dZ=----'
                return (
                    f'dX={d[0]:+7.4f} dY={d[1]:+7.4f} dZ={d[2]:+7.4f}'
                )

            # Yellow, same size as elevation / elbow lines; under those rows.
            wrist_yellow = (0, 255, 255)  # BGR
            draw_corner_label(
                frame, 'L wrist Δ(hip): ' + _fmt_dxyz(delta_left_hip),
                corner='top-left', y_row=3,
                scale=1.4, thickness=3, row_height=55,
                color=wrist_yellow,
            )
            draw_corner_label(
                frame, 'R wrist Δ(hip): ' + _fmt_dxyz(delta_right_hip),
                corner='top-right', y_row=2,
                scale=1.4, thickness=3, row_height=55,
                color=wrist_yellow,
            )

            if args.debug:
                def _fmt(v):
                    if v is None:
                        return 'x=--   y=--   z=--'
                    return f'x={v[0]:+.2f} y={v[1]:+.2f} z={v[2]:+.2f}'

                draw_corner_label(
                    frame, 'L arm body: ' + _fmt(last_left_arm_body),
                    corner='top-left', y_row=4,
                    scale=0.6, thickness=2, row_height=55,
                    color=(255, 255, 255),
                )
                draw_corner_label(
                    frame, 'R arm body: ' + _fmt(last_right_arm_body),
                    corner='top-right', y_row=3,
                    scale=0.6, thickness=2, row_height=55,
                    color=(255, 255, 255),
                )

            cv2.imshow('Remote Pose (press q to quit)', frame)
            if skeleton_img is not None:
                cv2.imshow('3D Skeleton', skeleton_img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        running = False
        cap.release()
        cv2.destroyAllWindows()
        sock.close()
        print('Done.')


if __name__ == '__main__':
    main()
