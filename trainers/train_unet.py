"""Train a small U-Net (MobileNetV3-Small encoder) for fly-eye segmentation.

Dataset layout (produced locally, uploaded as-is):
    dataset/images/A001.jpg
    dataset/masks/A001.png      255 = eye
    dataset/splits.csv          id, phenotype, genotype, split

Usage:
    python train_unet.py --data ./dataset --out ./runs/unet1 --epochs 60

Outputs: best.pt, last.pt, metrics.csv, val/*.png previews, model.onnx (if --export)
"""
import argparse
import csv
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset


# ---------------------------------------------------------------- model
class ConvBN(nn.Module):
    def __init__(self, a, b):
        super().__init__()
        self.c = nn.Conv2d(a, b, 3, padding=1, bias=False)
        self.b = nn.BatchNorm2d(b)

    def forward(self, x):
        return F.relu(self.b(self.c(x)), inplace=True)


class Up(nn.Module):
    def __init__(self, cin, cskip, cout):
        super().__init__()
        self.u = nn.ConvTranspose2d(cin, cout, 2, stride=2)
        self.c1 = ConvBN(cout + cskip, cout)
        self.c2 = ConvBN(cout, cout)

    def forward(self, x, skip):
        x = self.u(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="nearest")
        return self.c2(self.c1(torch.cat([x, skip], 1)))


class UNetSmall(nn.Module):
    def __init__(self):
        super().__init__()
        from torchvision.models import mobilenet_v3_small
        m = mobilenet_v3_small(weights="IMAGENET1K_V1").features
        self.s1 = m[0:2]      # 16ch, /4
        self.s2 = m[2:4]      # 24ch, /8
        self.s3 = m[4:7]      # 40ch, /16
        self.s4 = m[7:12]     # 96ch, /32
        self.u3 = Up(96, 40, 48)
        self.u2 = Up(48, 24, 24)
        self.u1 = Up(24, 16, 16)
        self.head = nn.Sequential(ConvBN(16, 16), nn.Conv2d(16, 1, 1))

    def forward(self, x):
        x1 = self.s1(x)
        x2 = self.s2(x1)
        x3 = self.s3(x2)
        x4 = self.s4(x3)
        y = self.u3(x4, x3)
        y = self.u2(y, x2)
        y = self.u1(y, x1)
        y = F.interpolate(y, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return self.head(y)


# ---------------------------------------------------------------- data
class EyeDS(Dataset):
    def __init__(self, root, items, size=512, train=False):
        self.root, self.items, self.size, self.train = root, items, size, train

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        r = self.items[i]
        img = Image.open(os.path.join(self.root, "images", r["file"])).convert("RGB")
        msk = Image.open(os.path.join(self.root, "masks", r["id"] + ".png")).convert("L")
        img = img.resize((self.size, self.size), Image.BILINEAR)
        msk = msk.resize((self.size, self.size), Image.NEAREST)
        a = np.asarray(img, np.float32) / 255.0
        m = (np.asarray(msk, np.float32) > 127).astype(np.float32)
        if self.train:
            if np.random.rand() < 0.5:
                a, m = a[:, ::-1], m[:, ::-1]
            if np.random.rand() < 0.5:
                a, m = a[::-1], m[::-1]
            k = np.random.randint(4)
            a, m = np.rot90(a, k), np.rot90(m, k)
            # scale jitter: simulate eyes of different apparent size (fixes small-phenotype bias)
            s = np.random.uniform(0.7, 1.5)
            if abs(s - 1.0) > 0.05:
                img_t = Image.fromarray((a * 255).astype(np.uint8))
                msk_t = Image.fromarray((m * 255).astype(np.uint8))
                nw = max(64, int(self.size * s))
                a = np.asarray(img_t.resize((nw, nw), Image.BILINEAR), np.float32) / 255.0
                m = (np.asarray(msk_t.resize((nw, nw), Image.NEAREST), np.float32) > 127).astype(np.float32)
                # random crop or pad back to size
                if nw >= self.size:
                    ox = np.random.randint(0, nw - self.size + 1)
                    oy = np.random.randint(0, nw - self.size + 1)
                    a = a[oy:oy + self.size, ox:ox + self.size]
                    m = m[oy:oy + self.size, ox:ox + self.size]
                else:
                    pad = self.size - nw
                    a = np.pad(a, ((0, pad), (0, pad), (0, 0)), mode="edge")
                    m = np.pad(m, ((0, pad), (0, pad)), mode="edge")
            a = a * np.random.uniform(0.8, 1.2) + np.random.uniform(-0.08, 0.08)
            a = np.clip(a, 0, 1)
        a = np.ascontiguousarray(a.transpose(2, 0, 1))
        m = np.ascontiguousarray(m[None])
        return torch.from_numpy(a), torch.from_numpy(m)


def dice_loss(p, t, eps=1e-6):
    p = torch.sigmoid(p)
    num = 2 * (p * t).sum((1, 2, 3)) + eps
    den = p.sum((1, 2, 3)) + t.sum((1, 2, 3)) + eps
    return 1 - (num / den).mean()


def dice_metric(p, t, eps=1e-6):
    p = (torch.sigmoid(p) > 0.5).float()
    num = 2 * (p * t).sum((1, 2, 3)) + eps
    den = p.sum((1, 2, 3)) + t.sum((1, 2, 3)) + eps
    return (num / den).mean().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="./dataset")
    ap.add_argument("--out", default="./runs/unet1")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--export", action="store_true")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    rows = list(csv.DictReader(open(os.path.join(a.data, "splits.csv"), encoding="utf-8-sig")))
    ann = {r["id"]: r for r in csv.DictReader(open(os.path.join(a.data, "annotation.csv"), encoding="utf-8-sig"))}
    for r in rows:
        r["file"] = ann[r["id"]]["file"]
    # defensive: drop rows whose image/mask is missing (e.g. a corrupt ROI never rasterised)
    keep = [r for r in rows
            if os.path.exists(os.path.join(a.data, "masks", r["id"] + ".png"))
            and os.path.exists(os.path.join(a.data, "images", r["file"]))]
    if len(keep) != len(rows):
        print("dropped %d rows with missing files: %s" % (
            len(rows) - len(keep), sorted(set(r["id"] for r in rows if r not in keep))))
        rows = keep
    tr = [r for r in rows if r["split"] == "train"]
    va = [r for r in rows if r["split"] == "val"]
    print("train %d  val %d" % (len(tr), len(va)))

    dl_tr = DataLoader(EyeDS(a.data, tr, a.size, True), batch_size=a.bs, shuffle=True,
                       num_workers=a.workers, drop_last=True)
    dl_va = DataLoader(EyeDS(a.data, va, a.size, False), batch_size=a.bs, num_workers=a.workers)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = UNetSmall().to(dev)
    opt = torch.optim.AdamW(
        [{"params": [p for n, p in model.named_parameters() if n.startswith(("s1", "s2", "s3", "s4"))], "lr": a.lr * 0.15},
         {"params": [p for n, p in model.named_parameters() if not n.startswith(("s1", "s2", "s3", "s4"))], "lr": a.lr}],
        weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)

    best = 0.0
    log = open(os.path.join(a.out, "metrics.csv"), "w", newline="")
    log.write("epoch,train_loss,val_loss,val_dice,lr,sec\n")
    for ep in range(1, a.epochs + 1):
        t0 = time.time()
        model.train()
        tl = 0.0
        for x, y in dl_tr:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad()
            o = model(x)
            loss = F.binary_cross_entropy_with_logits(o, y) + dice_loss(o, y)
            loss.backward()
            opt.step()
            tl += loss.item()
        tl /= max(1, len(dl_tr))
        model.eval()
        vl, vd = 0.0, 0.0
        with torch.no_grad():
            for x, y in dl_va:
                x, y = x.to(dev), y.to(dev)
                o = model(x)
                vl += (F.binary_cross_entropy_with_logits(o, y) + dice_loss(o, y)).item()
                vd += dice_metric(o, y)
        vl /= max(1, len(dl_va))
        vd /= max(1, len(dl_va))
        sched.step()
        sec = time.time() - t0
        print("ep %3d  train %.4f  val %.4f  dice %.4f  %.0fs" % (ep, tl, vl, vd, sec), flush=True)
        log.write("%d,%.5f,%.5f,%.5f,%.2e,%.0f\n" % (ep, tl, vl, vd, sched.get_last_lr()[-1], sec))
        log.flush()
        torch.save(model.state_dict(), os.path.join(a.out, "last.pt"))
        if vd > best:
            best = vd
            torch.save(model.state_dict(), os.path.join(a.out, "best.pt"))
    print("best val dice %.4f" % best)

    if a.export:
        model.load_state_dict(torch.load(os.path.join(a.out, "best.pt"), map_location=dev))
        model.eval()
        dummy = torch.randn(1, 3, a.size, a.size, device=dev)
        torch.onnx.export(model, dummy, os.path.join(a.out, "model.onnx"),
                          input_names=["input"], output_names=["logits"],
                          dynamic_axes={"input": {0: "b"}, "logits": {0: "b"}}, opset_version=13)
        print("exported", os.path.join(a.out, "model.onnx"))


if __name__ == "__main__":
    main()
