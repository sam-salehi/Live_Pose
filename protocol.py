"""
Shared message protocol for remote pose inference.
Length-prefixed TCP messages: 4-byte big-endian uint32 length + payload.
"""

import struct


def send_msg(sock, data: bytes):
    """Send a length-prefixed message."""
    length = len(data)
    sock.sendall(struct.pack('>I', length) + data)


def recv_msg(sock) -> bytes:
    """Receive a length-prefixed message. Returns None on disconnect."""
    raw_len = _recv_exact(sock, 4)
    if raw_len is None:
        return None
    length = struct.unpack('>I', raw_len)[0]
    return _recv_exact(sock, length)


def pack_frame(frame_w: int, frame_h: int, jpeg_bytes: bytes) -> bytes:
    """Pack a frame message: 8-byte header (w, h as uint32) + JPEG bytes."""
    header = struct.pack('>II', frame_w, frame_h)
    return header + jpeg_bytes


def unpack_frame(data: bytes):
    """Unpack a frame message. Returns (frame_w, frame_h, jpeg_bytes)."""
    frame_w, frame_h = struct.unpack('>II', data[:8])
    jpeg_bytes = data[8:]
    return frame_w, frame_h, jpeg_bytes


def _recv_exact(sock, n: int) -> bytes:
    """Receive exactly n bytes from socket."""
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            return None
        buf.extend(chunk)
    return bytes(buf)
