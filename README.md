# FlyMorph

Batch area measurement for *Drosophila* phenotypes — eyes, whole wings, the L3–L4
intervein band, and pupae — from ordinary photographs.

FlyMorph replaces the manual *ImageJ → draw ROI → Measure* loop with a small
desktop application: pick a folder, pick a task, press start, and get a CSV table
plus an optional review pass where you only correct the few images the model got
wrong. No command line, no ImageJ, no Python installation needed by the end user.

## What it measures

| Module | Target | Output folder | Dice | Median area error |
|---|---|---|---|---|
| Eye area | the eye, including contiguous red/orange tissue | `eye_output/` | 0.966 | 2.5 % |
| Whole wing | wing membrane, veins included | `wing_output/` | 0.978 | 1.5 % |
| Intervein area | membrane between veins L3 and L4, veins excluded | `vein_output/` | 0.950 | 2.4 % |
| Pupa area | single pupa on a dark background | `pupa_output/` | 0.973 | 1.0 % |

"Median area error" is the relative difference from a human ImageJ annotation on
held-out images. Units are **original image pixels**, identical to ImageJ's
*Area*, so results are directly comparable with previous manual measurements.

## How it works

- **Segmentation**: a small U-Net with a MobileNetV3 encoder (ImageNet init),
  BCE + Dice loss, AdamW with a lower learning rate on the encoder, cosine
  schedule, scale jitter 0.7–1.5, flips and 90° rotations.
- **Inference**: ONNX Runtime on CPU — no PyTorch in the shipped app.
- **Wing-vein module is two-stage**: the whole-wing model locates the wing, the
  crop is upscaled to 1280 px, and a second model segments the L3–L4 band inside
  that crop (4–5× more pixels on target than running on the full frame).
- **Outline extraction** traces the mask boundary (Moore neighbourhood +
  Douglas–Peucker) instead of sampling rays from the centroid, so concave and
  elongated shapes keep their true form.
- **Spur removal** applies a morphological opening/closing with a resolution-
  scaled kernel: thin protrusions (hairs, scratches) are eroded away and the two
  flanking points are reconnected. This cut the 90th-percentile area error of the
  vein module from 15 % to 7 %.
- **Review queue**: robust z-score (median ± 2.5 MAD) inside the folder, low
  confidence, and abnormal outline shape.

## Install and run

### Users

1. Download `FlyMorph_v1.0.6.zip` from the *Releases* page and extract it anywhere.
2. Run `FlyMorph.exe`. Windows asks for elevation (the app writes next to your
   images) — accept it.
3. If Windows SmartScreen appears, choose *More info → Run anyway*. The build is
   signed with a self-signed certificate, so SmartScreen has no reputation for
   it; this is expected and harmless.

### Developers

```powershell
conda create -n flymorph python=3.11 -y
conda activate flymorph
pip install PySide6 onnxruntime numpy pillow pyinstaller

cd app
python main.py                      # run from source
powershell -ExecutionPolicy Bypass -File build_exe.ps1   # build the exe
```

`build_exe.ps1` writes `FlyMorph.exe` into `build/dist/FlyMorph/`. The four model
files in `models/` are bundled automatically.

## Output format

Each run creates one folder next to the images:

```
<image folder>/<module>_output/
├── prefill.csv     raw model output (area, confidence, threshold, shape metric)
├── review.csv      images flagged for review, with the reason
├── final.csv       final table — review edits are merged here (use this one)
└── rois/           per-image outlines as .json and as ImageJ-compatible .roi
```

`final.csv` columns: `file, area_px, conf, thr, edited`.

Because the outlines are exported as real ImageJ `.roi` files, a reviewed run
doubles as new training material: the `rois/` folder can be turned back into
masks with `trainers/` scripts.

## Review shortcuts

| Key | Action | Key | Action |
|---|---|---|---|
| `S` | save and go to the next image | `N` | skip to next |
| `B` | previous image | `R` | redraw the outline |
| `C` | clear the outline | `Esc` | leave review |
| wheel | zoom | middle-drag | pan |

## Retraining

Training data and the interactive annotation workflow are kept in a separate
private repository (raw photographs are unpublished experimental material);
the format is one image + one binary mask per sample.

```bash
export FLYMORPH_DATA=/path/to/dataset      # contains images/, masks/, splits.csv, annotation.csv
python trainers/train_unet.py --data $FLYMORPH_DATA --out ./runs/run1 \
       --epochs 60 --bs 3 --size 1280 --export
python trainers/eval_vein4.py              # per-threshold Dice and area error
```

`train_unet.py` exports `model.onnx` next to the checkpoints; drop it into
`models/` and update the corresponding entry in `TASKS` in `app/main.py`.

## Notes and limitations

- Absolute areas are only comparable within one imaging setup. Across devices or
  magnifications use ratios (e.g. intervein area ÷ wing area) or within-group
  relative values.
- The intervein module is the hardest target: roughly 5–10 % of images need a
  manual correction. Use the default *review all* mode for it.
- Biological variation between individuals of the same group (CV ≈ 16 %) is an
  order of magnitude larger than the measurement error (1–2.5 %); plan group
  sizes accordingly (6–15 individuals for a 20–40 % effect).

## Citation

If FlyMorph contributes to a publication, please cite this repository. A formal
method description is in preparation.

## License

MIT — see [LICENSE](LICENSE). Bundled third-party components: PySide6/Qt
(LGPL-3.0), onnxruntime (MIT), PyTorch (BSD-3-Clause, training only).
