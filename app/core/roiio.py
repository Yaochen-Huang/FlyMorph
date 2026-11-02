"""ImageJ .roi writer (v228, polygon type).

Layout: 64-byte header + planar int16 coordinates relative to (left, top)
+ 82-byte tail. Coordinates are stored relative to the ROI bounding box, which
is what ImageJ expects.
"""
import struct


def roi_bytes(points):
    xs = [int(round(p[0])) for p in points]
    ys = [int(round(p[1])) for p in points]
    n = len(xs)
    left, right = min(xs), max(xs)
    top, bottom = min(ys), max(ys)
    h = bytearray(64)
    h[0:4] = b"Iout"
    struct.pack_into(">h", h, 4, 228)
    struct.pack_into(">hhhh", h, 8, top, left, bottom, right)
    struct.pack_into(">h", h, 16, n)
    body = b"".join(struct.pack(">h", x - left) for x in xs)
    body += b"".join(struct.pack(">h", y - top) for y in ys)
    return bytes(h) + body + b"\x00" * 82


def write_roi(path, points):
    if not points or len(points) < 3:
        return False
    with open(path, "wb") as fh:
        fh.write(roi_bytes(points))
    return True


def read_roi(path):
    with open(path, "rb") as fh:
        d = fh.read()
    top = struct.unpack(">h", d[8:10])[0]
    left = struct.unpack(">h", d[10:12])[0]
    n = struct.unpack(">h", d[16:18])[0]
    xs = [x + left for x in struct.unpack(">" + "h" * n, d[64:64 + 2 * n])]
    ys = [y + top for y in struct.unpack(">" + "h" * n, d[64 + 2 * n:64 + 4 * n])]
    return list(zip(xs, ys))
