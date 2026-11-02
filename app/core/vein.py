"""Two-stage vein measurement: wing model locates the wing -> crop -> vein model.

The vein model was trained on wing crops (margin 1.18, resized to 1280), so the
same crop geometry must be used at inference or the scale will not match.
"""
import numpy as np
from PIL import Image

from .infer import EyeModel, shoelace

CROP_SIZE = 1280
MARGIN = 1.18


class VeinPipeline:
    def __init__(self, wing_onnx, vein_onnx, vein_thr=0.40, vein_size=CROP_SIZE,
                 wing_thr=0.50, margin=MARGIN, outline="contour", open_kernel=13):
        self.wing = EyeModel(wing_onnx, thr_big=wing_thr, thr_small=wing_thr, small_frac=0.0)
        self.vein = EyeModel(vein_onnx, thr_big=vein_thr, thr_small=vein_thr,
                             small_frac=0.0, size=vein_size, outline=outline,
                             open_kernel=open_kernel)
        self.margin = margin
        self.size = vein_size

    def predict(self, path):
        im = Image.open(path).convert("RGB")
        W, H = im.size
        w = self.wing.predict(path)
        if not w["poly"]:
            return dict(ok=False, reason="wing not found", poly=None, area=0.0, conf=0.0,
                        thr=0.0, spike=0.0, box=None, size=(W, H), wing_area=0.0)

        xs = [p[0] for p in w["poly"]]
        ys = [p[1] for p in w["poly"]]
        cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
        half_w = (max(xs) - min(xs)) * self.margin / 2
        half_h = (max(ys) - min(ys)) * self.margin / 2
        box = (int(max(0, cx - half_w)), int(max(0, cy - half_h)),
               int(min(W, cx + half_w)), int(min(H, cy + half_h)))
        if box[2] - box[0] < 50 or box[3] - box[1] < 50:
            return dict(ok=False, reason="bad crop", poly=None, area=0.0, conf=0.0, thr=0.0,
                        spike=0.0, box=box, size=(W, H), wing_area=w["area"])

        crop = im.crop(box)
        r = self.vein.predict_image(crop)
        if not r["poly"]:
            return dict(ok=False, reason="vein not found", poly=None, area=0.0, conf=0.0,
                        thr=r["thr"], spike=0.0, box=box, size=(W, H), wing_area=w["area"])

        # EyeModel already returns the polygon in the *crop's* pixel coordinates
        # (it maps the mask back to the input image size), so only the crop origin
        # needs to be added here.
        cw, chh = box[2] - box[0], box[3] - box[1]
        poly = [(box[0] + p[0], box[1] + p[1]) for p in r["poly"]]
        # recompute geometry in original pixels (a crop-coordinate polygon is affine
        # scaled, so area scales by sx*sy; recomputing keeps it exact)
        area = shoelace(poly)
        perim = float(np.sum(np.hypot(np.diff(np.r_[[p[0] for p in poly], poly[0][0]]),
                                     np.diff(np.r_[[p[1] for p in poly], poly[0][1]]))))
        circ = float(np.clip(4 * np.pi * area / perim ** 2, 0, 1)) if perim > 0 else 0.0
        px = np.array([p[0] for p in poly])
        py = np.array([p[1] for p in poly])
        rr = np.hypot(px - px.mean(), py - py.mean())
        spike = float(rr.max() / max(1e-6, float(np.median(rr))))
        return dict(ok=True, poly=poly, area=area, conf=r["conf"], thr=r["thr"],
                    spike=spike, box=box, size=(W, H), wing_area=w["area"],
                    circularity=circ, raw_spike=r.get("raw_spike", 0.0))
