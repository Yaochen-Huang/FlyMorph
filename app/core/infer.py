"""ONNX inference for the eye task (no PyTorch needed)."""
import json
import os

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageFilter

SIZE = 768
THR_BIG, THR_SMALL = 0.35, 0.40
SMALL_FRAC = 0.02


def _polygon(mask, spike_limit=1.8):
    """Radial outline of the mask, with thin spikes clipped.

    A filament touching the eye (hair, antenna, scratch) inflates the mask and
    produces a long spike. Radii beyond spike_limit x median are treated as
    outliers and interpolated from neighbours; the ratio is returned so QC can
    flag the image.
    """
    ys, xs = np.nonzero(mask)
    if len(xs) < 60:
        return None, 0.0
    cx, cy = float(xs.mean()), float(ys.mean())
    r = np.hypot(xs - cx, ys - cy)
    ang = (np.arctan2(ys - cy, xs - cx) + 2 * np.pi) % (2 * np.pi)
    b = (ang / (2 * np.pi) * 360).astype(np.int32) % 360
    rad = np.zeros(360)
    np.maximum.at(rad, b, r)
    ok = rad > 0
    if ok.mean() < 0.5:
        return None, 0.0
    rad[~ok] = np.interp(np.nonzero(~ok)[0], np.nonzero(ok)[0], rad[ok], period=360)
    med = float(np.median(rad))
    spike = float(rad.max() / max(1e-6, med))
    # Robust spike removal: compare each ray with the angular median of its
    # neighbourhood. A thin spike is a narrow protrusion (far above the local
    # median) and gets clipped; a genuinely elongated target (e.g. the L3-L4
    # intervein band) varies smoothly and is preserved.
    win = 11
    ext = np.concatenate([rad[-win:], rad, rad[:win]])
    local = np.array([np.median(ext[i:i + 2 * win + 1]) for i in range(len(rad))])
    out = rad > spike_limit * np.maximum(local, 1e-6)
    if out.any() and (~out).sum() >= 36:
        rad[out] = np.interp(np.nonzero(out)[0], np.nonzero(~out)[0], rad[~out], period=360)
    k = np.array([0.25, 0.5, 0.25])
    ext = np.concatenate([rad[-2:], rad, rad[:2]])
    rad = np.convolve(ext, k, mode="same")[2:-2]
    th = np.arange(360) * 2 * np.pi / 360
    return [(float(cx + rad[i] * np.cos(th[i])), float(cy + rad[i] * np.sin(th[i])))
            for i in range(360)], spike


class EyeModel:
    def __init__(self, onnx_path, thr_big=THR_BIG, thr_small=THR_SMALL, small_frac=SMALL_FRAC,
                 size=SIZE, spike_limit=1.8, outline="radial", open_kernel=0):
        so = ort.SessionOptions()
        so.intra_op_num_threads = 0
        self.sess = ort.InferenceSession(onnx_path, so, providers=["CPUExecutionProvider"])
        self.input_name = self.sess.get_inputs()[0].name
        self.thr_big = thr_big
        self.thr_small = thr_small
        self.small_frac = small_frac
        self.size = size
        self.spike_limit = spike_limit
        self.outline = outline          # "radial" (default) or "contour"
        # Morphological opening removes narrow protrusions (a spur is cut off and the
        # two flanking points get connected), morphological closing fills tears.
        self.open_kernel = open_kernel

    def predict(self, path):
        return self.predict_image(Image.open(path).convert("RGB"))

    def predict_image(self, im):
        W, H = im.size
        x = np.asarray(im.resize((self.size, self.size), Image.BILINEAR), np.float32) / 255.0
        x = x.transpose(2, 0, 1)[None]
        logits = self.sess.run(None, {self.input_name: x})[0][0, 0]
        p = 1.0 / (1.0 + np.exp(-logits))
        m = p > self.thr_big
        thr = self.thr_big
        if m.sum() / m.size < self.small_frac:
            m = p > self.thr_small
            thr = self.thr_small
        if self.open_kernel > 1:
            # Spur removal on the small (network-resolution) mask; the kernel is
            # scaled with the input resolution so its physical size stays constant.
            k = int(round(self.open_kernel * self.size / 1280.0))
            if k % 2 == 0:
                k += 1
            if k > 1:
                from .morph import open_close
                m = open_close(m, k)
        mfull = np.asarray(Image.fromarray((m * 255).astype(np.uint8))
                           .resize((W, H), Image.NEAREST)) > 127
        if self.outline == "contour":
            from .contour import trace, simplify
            pts = trace(mfull)
            poly_c = None
            if pts is not None and len(pts) >= 10:
                pts = simplify(pts, tol=max(1.5, min(W, H) / 800.0))
                # trace() returns (row, col) = (y, x); polygons expect (x, y)
                cand = [(float(p[1]), float(p[0])) for p in pts]
                a_poly = abs(shoelace(cand))
                # a broken trace collapses the area -> fall back to the radial outline
                if a_poly > 0.5 * float(mfull.sum()):
                    poly_c = cand
            if poly_c is not None:
                poly, spike_raw = poly_c, 1.0
            else:
                poly, spike_raw = _polygon(mfull, self.spike_limit)
        else:
            poly, spike_raw = _polygon(mfull, self.spike_limit)
        area = shoelace(poly) if poly else 0.0
        conf = float(p[m].mean()) if m.sum() else 0.0
        # spike AFTER clipping: measures residual shape abnormality (the raw value
        # is large on every image of some devices and is useless as a QC flag)
        spike = 0.0
        if poly:
            px = np.array([q[0] for q in poly])
            py = np.array([q[1] for q in poly])
            rr = np.hypot(px - px.mean(), py - py.mean())
            med = float(np.median(rr))
            spike = float(rr.max() / max(1e-6, med))
        return dict(poly=poly, area=area, conf=conf, thr=thr, spike=spike,
                    raw_spike=spike_raw, size=(W, H))


def shoelace(pts):
    s = 0.0
    for i in range(len(pts)):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % len(pts)]
        s += x1 * y2 - x2 * y1
    return abs(s) / 2


def run_batch(model, folder, out_dir, progress=None, cancel=None):
    files = sorted(f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".jpeg", ".png")))
    os.makedirs(os.path.join(out_dir, "rois"), exist_ok=True)
    csv_path = os.path.join(out_dir, "prefill.csv")
    with open(csv_path, "w", encoding="utf-8") as fh:
        fh.write("file,area_px,conf,thr,spike\n")
    rows = []
    for i, name in enumerate(files):
        if cancel and cancel():
            break
        try:
            r = model.predict(os.path.join(folder, name))
        except Exception as e:
            rows.append(dict(file=name, area=0.0, conf=0.0, thr=0.0, err=str(e)))
            if progress:
                progress(i + 1, len(files), name)
            continue
        base = os.path.splitext(name)[0]
        if r["poly"]:
            json.dump(r["poly"], open(os.path.join(out_dir, "rois", base + ".json"), "w"))
            # ImageJ-compatible ROI so the result can be opened/edited in ImageJ
            # and reused as training material later.
            try:
                from .roiio import write_roi
                write_roi(os.path.join(out_dir, "rois", base + ".roi"), r["poly"])
            except Exception:
                pass
        with open(csv_path, "a", encoding="utf-8") as fh:
            fh.write("%s,%.1f,%.3f,%.2f,%.2f\n" % (base, r["area"], r["conf"], r["thr"],
                                                   r.get("spike", 0.0)))
        rows.append(dict(file=name, area=r["area"], conf=r["conf"], thr=r["thr"],
                         err="" if r["poly"] else "no_polygon"))
        if progress:
            progress(i + 1, len(files), name)
    # batch-local outliers: robust z with a 2.5 threshold. The within-batch CV of
    # real measurements is around 16%, so 2.5 sigma is tight enough to catch a bad
    # segmentation without flooding the review queue.
    areas = sorted(r["area"] for r in rows if r["area"] > 0)
    flagged = set()
    if len(areas) >= 5:
        m = areas[len(areas) // 2]
        d = sorted(abs(a - m) for a in areas)
        mad = max(1e-6, d[len(d) // 2])
        for r in rows:
            if r["area"] > 0 and abs(0.6745 * (r["area"] - m) / mad) > 2.5:
                flagged.add(r["file"])
    with open(os.path.join(out_dir, "review.csv"), "w", newline="", encoding="utf-8") as fh:
        fh.write("file,area_px,conf,reason\n")
        for r in rows:
            rs = []
            if r["file"] in flagged:
                rs.append("outlier")
            if r["conf"] and r["conf"] < 0.75:
                rs.append("low_conf")
            # thin protrusion (hair / antenna / scratch touching the eye)
            sp = float(r.get("spike") or 0.0)
            if sp > 2.0:
                rs.append("spike_%.1f" % sp)
            if r["err"]:
                rs.append(r["err"])
            if rs:
                fh.write("%s,%.1f,%.3f,%s\n" % (r["file"], r["area"], r["conf"], ";".join(rs)))
    # Always materialise final.csv so every run has the same output format, whether
    # or not the user goes through manual review afterwards.
    try:
        from .tables import rebuild_final
        rebuild_final(out_dir)
    except Exception:
        pass
    return rows, sorted(flagged)
