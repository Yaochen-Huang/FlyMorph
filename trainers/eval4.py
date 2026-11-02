"""Evaluate unet4 on the test split, separated by device domain (old vs shandong)."""
import csv
import os
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.environ.get("FLYMORPH_DATA", "."))
from train_unet import UNetSmall

ROOT = os.environ.get("FLYMORPH_DATA", ".")
DS = os.path.join(ROOT, "trainset_v2")
CKPT = os.path.join(ROOT, "runs", "unet4", "best.pt")
SIZE = 768
THR_BIG, THR_SMALL = 0.35, 0.40

splits = list(csv.DictReader(open(os.path.join(DS, "splits.csv"), encoding="utf-8-sig")))
ann = {r["id"]: r for r in csv.DictReader(open(os.path.join(DS, "annotation.csv"), encoding="utf-8-sig"))}
test = [r for r in splits if r["split"] == "test"]

model = UNetSmall()
model.load_state_dict(torch.load(CKPT, map_location="cpu"))
model.eval()

res = []
for r in test:
    iid = r["id"]
    f = ann[iid]["file"]
    ip = os.path.join(DS, "images", f)
    mp = os.path.join(DS, "masks", iid + ".png")
    if not (os.path.exists(ip) and os.path.exists(mp)):
        continue
    im = Image.open(ip).convert("RGB")
    W, H = im.size
    ms = Image.open(mp).convert("L")
    x = np.asarray(im.resize((SIZE, SIZE), Image.BILINEAR), np.float32) / 255.0
    y = (np.asarray(ms.resize((SIZE, SIZE), Image.NEAREST), np.float32) > 127).astype(np.float32)
    with torch.no_grad():
        pr = torch.sigmoid(model(torch.from_numpy(x.transpose(2, 0, 1))[None]))[0, 0].numpy()
    m = pr > THR_BIG
    if m.sum() / m.size < 0.02:
        m = pr > THR_SMALL
    m = m.astype(np.float32)
    inter = float((m * y).sum())
    dice = 2 * inter / (m.sum() + y.sum() + 1e-6)
    a_pred = float(m.sum()) * (W / SIZE) * (H / SIZE)
    a_true = float(y.sum()) * (W / SIZE) * (H / SIZE)
    dev = (a_pred - a_true) / a_true * 100 if a_true else float("nan")
    dom = "shandong" if iid.startswith("W") else "old"
    res.append((dom, r.get("phenotype", ""), dice, dev))

import statistics as st
print("%-10s %-10s %5s %8s %8s %8s" % ("domain", "pheno", "n", "dice", "|dev|med", "|dev|p90"))
for dom in ("old", "shandong"):
    for ph in sorted({x[1] for x in res if x[0] == dom}):
        g = [x for x in res if x[0] == dom and x[1] == ph]
        if not g:
            continue
        d = [x[2] for x in g]
        dv = sorted(abs(x[3]) for x in g)
        print("%-10s %-10s %5d %8.4f %8.2f %8.2f" %
              (dom, ph, len(g), st.mean(d), st.median(dv),
               dv[min(len(dv) - 1, int(len(dv) * 0.9))]))
    g = [x for x in res if x[0] == dom]
    if g:
        d = [x[2] for x in g]
        dv = sorted(abs(x[3]) for x in g)
        print("%-10s %-10s %5d %8.4f %8.2f %8.2f" %
              (dom, "ALL", len(g), st.mean(d), st.median(dv),
               dv[min(len(dv) - 1, int(len(dv) * 0.9))]))
print("worst:", sorted([(round(x[2], 3), x[0], x[1], round(x[3], 1)) for x in res])[:5])
