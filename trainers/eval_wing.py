import csv
import os
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.environ.get("FLYMORPH_DATA", "."))
from train_unet import UNetSmall

ROOT = os.environ.get("FLYMORPH_DATA", ".")
DS = os.path.join(ROOT, "trainset_wing")
CKPT = os.path.join(ROOT, "runs", "wing1", "best.pt")
SIZE = 768

splits = list(csv.DictReader(open(os.path.join(DS, "splits.csv"), encoding="utf-8-sig")))
ann = {r["id"]: r for r in csv.DictReader(open(os.path.join(DS, "annotation.csv"), encoding="utf-8-sig"))}
model = UNetSmall()
model.load_state_dict(torch.load(CKPT, map_location="cpu"))
model.eval()

res = []
for r in splits:
    if r["split"] != "test":
        continue
    iid = r["id"]
    ip = os.path.join(DS, "images", ann[iid]["file"])
    mp = os.path.join(DS, "masks", iid + ".png")
    im = Image.open(ip).convert("RGB")
    W, H = im.size
    x = np.asarray(im.resize((SIZE, SIZE), Image.BILINEAR), np.float32) / 255.0
    y = (np.asarray(Image.open(mp).convert("L").resize((SIZE, SIZE), Image.NEAREST),
                    np.float32) > 127).astype(np.float32)
    with torch.no_grad():
        pr = torch.sigmoid(model(torch.from_numpy(x.transpose(2, 0, 1))[None]))[0, 0].numpy()
    m = (pr > 0.5).astype(np.float32)
    inter = float((m * y).sum())
    dice = 2 * inter / (m.sum() + y.sum() + 1e-6)
    ap = float(m.sum()) * (W / SIZE) * (H / SIZE)
    at = float(y.sum()) * (W / SIZE) * (H / SIZE)
    res.append((r.get("phenotype", "?"), dice, (ap - at) / at * 100))

import statistics as st
print("%-8s %4s %8s %9s %9s" % ("source", "n", "dice", "|dev|med", "|dev|max"))
for src in sorted({x[0] for x in res}):
    g = [x for x in res if x[0] == src]
    d = [x[1] for x in g]
    dv = sorted(abs(x[2]) for x in g)
    print("%-8s %4d %8.4f %9.2f %9.2f" %
          (src, len(g), st.mean(d), st.median(dv), dv[-1]))
d = [x[1] for x in res]
dv = sorted(abs(x[2]) for x in res)
print("%-8s %4d %8.4f %9.2f %9.2f" % ("ALL", len(res), st.mean(d), st.median(dv), dv[-1]))
print("worst:", sorted(res, key=lambda x: x[1])[:3])
