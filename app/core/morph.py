"""Fast binary morphology with numpy (separable shifts).

PIL's MinFilter/MaxFilter are rank filters: O(k^2) per pixel and unusably slow on
megapixel masks (2.7 s for 4 passes at 1280). Separable min/max needs O(k) numpy
passes and is ~50-100x faster.
"""
import numpy as np


def _shift(m, d, axis):
    if d == 0:
        return m
    out = np.zeros_like(m)
    if axis == 0:
        if d > 0:
            out[d:] = m[:-d]
        else:
            out[:d] = m[-d:]
    else:
        if d > 0:
            out[:, d:] = m[:, :-d]
        else:
            out[:, :d] = m[:, -d:]
    return out


def _filter(m, k, fn):
    """Separable min(=AND)/max(=OR) over a k x k square."""
    h = k // 2
    vert = m.copy()
    for d in range(1, h + 1):
        vert = fn(vert, _shift(m, d, 0))
        vert = fn(vert, _shift(m, -d, 0))
    horiz = vert.copy()
    for d in range(1, h + 1):
        horiz = fn(horiz, _shift(vert, d, 1))
        horiz = fn(horiz, _shift(vert, -d, 1))
    return horiz


def erode(m, k):
    return _filter(m.astype(bool), int(k), np.logical_and)


def dilate(m, k):
    return _filter(m.astype(bool), int(k), np.logical_or)


def open_close(m, k):
    """Opening removes narrow protrusions, closing bridges the resulting cut."""
    if k <= 1:
        return m.astype(bool)
    opened = dilate(erode(m, k), k)
    return erode(dilate(opened, k), k)
