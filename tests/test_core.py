"""Smoke tests for the pure-python geometry helpers (no model files needed).

Run with:  python -m pytest tests -q      (or simply  python tests/test_core.py)
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))

import numpy as np

from core import morph
from core.contour import simplify, trace
from core.infer import shoelace
from core.roiio import read_roi, roi_bytes, write_roi


def test_shoelace_square():
    square = [(0, 0), (10, 0), (10, 10), (0, 10)]
    assert abs(shoelace(square)) == 100


def test_roi_roundtrip(tmp_path=None):
    pts = [(12.4, 30.6), (44.0, 31.0), (45.2, 80.9), (13.7, 79.1)]
    data = roi_bytes(pts)
    assert data[:4] == b"Iout"
    path = os.path.join(os.path.dirname(__file__), "_tmp_roundtrip.roi")
    write_roi(path, pts)
    got = read_roi(path)
    os.remove(path)
    assert len(got) == len(pts)
    for (x0, y0), (x1, y1) in zip(pts, got):
        assert abs(x0 - x1) <= 1 and abs(y0 - y1) <= 1


def test_trace_rectangle_with_notch():
    m = np.zeros((120, 200), bool)
    m[20:100, 20:180] = True
    m[80:100, 80:120] = False          # notch
    pts = trace(m)
    assert pts is not None and len(pts) > 100
    xs, ys = pts[:, 1], pts[:, 0]
    assert (xs.min(), ys.min()) == (20, 20)
    assert (xs.max(), ys.max()) == (179, 99)
    area = abs(0.5 * (np.dot(xs, np.roll(ys, 1)) - np.dot(ys, np.roll(xs, 1))))
    assert 0.9 * m.sum() < area < m.sum()


def test_simplify_keeps_corners():
    m = np.zeros((60, 60), bool)
    m[10:50, 10:50] = True
    pts = simplify(trace(m), tol=1.0)
    assert 4 <= len(pts) <= 12


def test_open_close_removes_thin_spur():
    m = np.zeros((80, 80), bool)
    m[30:50, 10:60] = True             # body
    m[36:40, 60:78] = True             # thin spur sticking out
    cleaned = morph.open_close(m, 9)
    assert cleaned[35:41, 70:75].sum() == 0          # spur gone
    assert cleaned[35:45, 20:55].sum() > m[35:45, 20:55].sum() * 0.9   # body kept


def test_open_close_keeps_elongated_shape():
    m = np.zeros((120, 260), bool)
    m[45:75, 10:250] = True            # long band, aspect 8:1
    cleaned = morph.open_close(m, 9)
    assert cleaned.sum() > 0.9 * m.sum()


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
    print("all tests passed")
