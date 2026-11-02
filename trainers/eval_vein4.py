import csv
import os
import sys

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.environ.get("FLYMORPH_DATA", "."))
from train_unet import UNetSmall

ROOT = os.environ.get("FLYMORPH_DATA", ".")
DS = os.path.join(ROOT, "trainset_vein_crop")
CKPT = os.path.join(ROOT, "runs", "vein4", "best.pt")
OUTDIR = os.path.join(ROOT, "runs", "vein4")
SIZE = 1280
THRS = [0.40, 0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90]

splits = list(csv.DictReader(open(os.path.join(DS, "splits.csv"), encoding="utf-8-sig")))
ann = {r["id"]: r for r in csv.DictReader(open(os.path.join(DS, "annotation.csv"), encoding="utf-8-sig"))}
test = [r for r in splits if r["split"] == "test"]

model = UNetSmall()
model.load_state_dict(torch.load(CKPT, map_location="cpu"))
model.eval()

cache = []
for r in test:
    iid = r["id"]
    ip = os.path.join(DS, "images", ann[iid]["file"])
    mp = os.path.join(DS, "masks", iid + ".png")
    im = Image.open(ip).convert("RGB")
    x = np.asarray(im.resize((SIZE, SIZE), Image.BILINEAR), np.float32) / 255.0
    y = (np.asarray(Image.open(mp).convert("L").resize((SIZE, SIZE), Image.NEAREST),
                    np.float32) > 127).astype(np.float32)
    with torch.no_grad():
        pr = torch.sigmoid(model(torch.from_numpy(x.transpose(2, 0, 1))[None]))[0, 0].numpy()
    cache.append((iid, pr, y))

print("test n =", len(cache))
print("%5s %8s %9s %9s %9s" % ("thr", "dice", "dev_med", "|dev|med", "|dev|max"))
best = (999, 0.5)
for thr in THRS:
    dices, devs = [], []
    for iid, pr, y in cache:
        m = (pr > thr).astype(np.float32)
        inter = float((m * y).sum())
        dices.append(2 * inter / (m.sum() + y.sum() + 1e-6))
        devs.append((m.sum() - y.sum()) / max(1e-6, y.sum()) * 100)
    dv = np.array(devs)
    md = float(np.median(np.abs(dv)))
    print("%5.2f %8.4f %9.1f %9.2f %9.2f" %
          (thr, np.mean(dices), np.median(dv), md, np.abs(dv).max()))
    if md < best[0]:
        best = (md, thr)
print("best thr:", best[1], " |dev| median:", round(best[0], 2))


def font(sz=14):
    p = "/usr/share/fonts/dejavu/DejaVuSans.ttf"
    return ImageFont.truetype(p, sz) if os.path.exists(p) else ImageFont.load_default()


def edge(a):
    return a & ~(np.roll(a, 1, 0) & np.roll(a, 1, 1))


thr = best[1]
cells = []
for iid, pr, y in cache[:8]:
    im = Image.open(os.path.join(DS, "images", ann[iid]["file"])).convert("RGB")
    im = im.resize((SIZE, SIZE), Image.BILINEAR)
    pm = (pr > thr).astype(np.uint8)
    arr = np.array(im)
    et = edge(y.astype(bool))
    ep = edge(pm.astype(bool))
    arr[et] = (0, 220, 0)
    arr[ep] = (255, 40, 40)
    ov = Image.fromarray(arr)
    ys, xs = np.nonzero(y > 0)
    pad = 120
    box = (max(0, xs.min() - pad), max(0, ys.min() - pad),
           min(SIZE, xs.max() + pad), min(SIZE, ys.max() + pad))
    ov = ov.crop(box)
    ov.thumbnail((520, 380))
    dev = (pm.sum() - y.sum()) / max(1, y.sum()) * 100
    cells.append((ov, "%s dev=%+.1f%%" % (iid, dev)))

cols, cw, ch = 4, 540, 400
rn = (len(cells) + cols - 1) // cols
sheet = Image.new("RGB", (cols * cw, rn * (ch + 20)), "white")
dr = ImageDraw.Draw(sheet)
f = font()
for i, (c, lab) in enumerate(cells):
    x, y2 = (i % cols) * cw, (i // cols) * (ch + 20)
    sheet.paste(c, (x + (cw - c.width) // 2, y2 + 20))
    dr.text((x + 5, y2 + 3), lab, fill="black", font=f)
p = os.path.join(OUTDIR, "vein4_sheet.png")
sheet.save(p)
print("saved", p)
