import csv
import os
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.environ.get("FLYMORPH_DATA", "."))
from train_unet import UNetSmall

ROOT = os.environ.get("FLYMORPH_DATA", ".")
DS = os.path.join(ROOT, "trainset_pupa")
CKPT = os.path.join(ROOT, "runs", "pupa1", "best.pt")
SIZE = 768
THRS = [0.4, 0.5, 0.6, 0.7, 0.8]

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
    W, H = im.size
    x = np.asarray(im.resize((SIZE, SIZE), Image.BILINEAR), np.float32) / 255.0
    y = (np.asarray(Image.open(mp).convert("L").resize((SIZE, SIZE), Image.NEAREST),
                    np.float32) > 127).astype(np.float32)
    with torch.no_grad():
        pr = torch.sigmoid(model(torch.from_numpy(x.transpose(2, 0, 1))[None]))[0, 0].numpy()
    cache.append((pr, y))

print("n test =", len(cache))
print("%5s %8s %9s %9s %9s" % ("thr", "dice", "dev_med", "|dev|med", "|dev|max"))
for thr in THRS:
    dices, devs = [], []
    for pr, y in cache:
        m = (pr > thr).astype(np.float32)
        inter = float((m * y).sum())
        dices.append(2 * inter / (m.sum() + y.sum() + 1e-6))
        devs.append((m.sum() - y.sum()) / max(1e-6, y.sum()) * 100)
    d = np.mean(dices)
    dv = np.array(devs)
    print("%5.2f %8.4f %9.1f %9.2f %9.2f" % (thr, d, np.median(dv), np.median(np.abs(dv)), np.abs(dv).max()))
