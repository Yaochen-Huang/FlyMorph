"""Contour tracing (Moore neighbourhood) for binary masks -> ordered polygon.

Radial "farthest point per angle" cannot represent concavities: any waist or
notch becomes a bulge. Tracing the actual boundary keeps the true shape, which is
what the elongated L3-L4 band needs.
"""
import numpy as np


def _boundary(mask):
    m = mask.astype(np.uint8)
    pad = np.pad(m, 1)
    up = pad[:-2, 1:-1]
    down = pad[2:, 1:-1]
    left = pad[1:-1, :-2]
    right = pad[1:-1, 2:]
    interior = m & up & down & left & right
    return (m == 1) & (~interior.astype(bool))


def trace(mask, max_steps=400000):
    """Moore-neighbour contour tracing. Returns (N,2) float array or None."""
    b = _boundary(mask)
    ys, xs = np.nonzero(b)
    if len(xs) < 20:
        return None
    # start at the topmost, then leftmost boundary pixel
    start = int(np.lexsort((xs, ys))[0])
    sy, sx = ys[start], xs[start]
    dirs = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]
    h, w = mask.shape

    def get(y, x):
        return 0 <= y < h and 0 <= x < w and b[y, x]

    contour = [(sy, sx)]
    prev_dir = 7          # came from the left
    cy, cx = sy, sx
    for _ in range(max_steps):
        found = False
        for k in range(8):
            d = (prev_dir + 1 + k) % 8
            ny, nx = cy + dirs[d][0], cx + dirs[d][1]
            if get(ny, nx):
                cy, cx = ny, nx
                prev_dir = (d + 4) % 8
                contour.append((cy, cx))
                found = True
                break
        if not found:
            break
        if (cy, cx) == (sy, sx) and len(contour) > 10:
            break
    if len(contour) < 10:
        return None
    return np.array(contour, dtype=np.float32)


def simplify(pts, tol=1.5):
    """Douglas-Peucker."""
    if len(pts) < 3:
        return pts

    def rec(a, b):
        if b <= a + 1:
            return []
        p0, p1 = pts[a], pts[b]
        d = p1 - p0
        n = np.hypot(d[0], d[1])
        if n < 1e-6:
            dist = np.hypot(pts[a + 1:b, 0] - p0[0], pts[a + 1:b, 1] - p0[1])
        else:
            dist = np.abs(np.cross(np.ones((b - a - 1, 1)) * d, pts[a + 1:b] - p0)) / n
        if len(dist) == 0:
            return []
        i = int(np.argmax(dist))
        if dist[i] > tol:
            return rec(a, a + 1 + i) + [a + 1 + i] + rec(a + 1 + i, b)
        return []

    idx = [0] + rec(0, len(pts) - 1) + [len(pts) - 1]
    return pts[sorted(set(idx))]
