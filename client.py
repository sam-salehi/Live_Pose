#!/usr/bin/env python3
"""
Remote pose inference client.
Captures webcam, sends JPEG frames to the server, receives 3D joints + 2D
keypoints, and displays the results locally.

Usage:
    python client.py --server-ip <PC_IP> --port 9000
"""

import sys
import json
import socket
import argparse
import threading
import time

import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend — no Tk, no GIL issues
import matplotlib.pyplot as plt

from live_pose3d import (
    draw_2d_skeleton,
    update_3d_plot,
)
from protocol import send_msg, recv_msg, pack_frame


def render_3d_to_image(fig, ax, joints_3d):
    """Render 3D skeleton to a BGR numpy image via the Agg backend."""
    update_3d_plot(ax, joints_3d)
    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    img = np.asarray(buf)                     # RGBA
    return cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)


def main():
    parser = argparse.ArgumentParser(description='Remote pose inference client')
    parser.add_argument('--server-ip', required=True, help='Server IP address')
    parser.add_argument('--port', type=int, default=9000)
    parser.add_argument('--jpeg-quality', type=int, default=80)
    args = parser.parse_args()

    # Connect to server
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((args.server_ip, args.port))
    print(f'Connected to server {args.server_ip}:{args.port}')

    # Webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print('Error: Cannot open webcam')
        sys.exit(1)

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f'Webcam: {frame_w}x{frame_h}')

    # Matplotlib 3D figure (offscreen via Agg)
    fig = plt.figure(figsize=(5, 5), dpi=100)
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title('Waiting for pose...')

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

    try:
        while running:
            ret, frame = cap.read()
            if not ret:
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
                skeleton_img = render_3d_to_image(fig, ax, joints_3d)

            # FPS counter
            fps_counter += 1
            elapsed = time.time() - fps_time
            if elapsed >= 1.0:
                fps_display = fps_counter / elapsed
                fps_counter = 0
                fps_time = time.time()

            cv2.putText(
                frame, f'FPS: {fps_display:.1f}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2,
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
        plt.close('all')
        sock.close()
        print('Done.')


if __name__ == '__main__':
    main()
