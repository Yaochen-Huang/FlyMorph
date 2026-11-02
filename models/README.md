# Models

Four ONNX models, all with input shape `[b, 3, S, S]` and a single-channel logit
output. They are run by ONNX Runtime on CPU; no PyTorch is needed at inference.

| File | Input size | Threshold | Task | Training data | Dice | Median area error |
|---|---|---|---|---|---|---|
| `eye_unet_v4.onnx` | 768 | 0.35 (0.40 when the predicted area is < 2 % of frame) | eye area | 499 eyes, two imaging setups | 0.966 | 2.5 % |
| `wing_unet_v1.onnx` | 768 | 0.50 | whole wing | 113 wings, two setups | 0.978 | 1.5 % |
| `vein_unet_v6.onnx` | 1280 | 0.40 | L3–L4 intervein band **inside a wing crop** | 298 regions | 0.950 | 2.4 % |
| `pupa_unet_v1.onnx` | 768 | 0.90 | pupa | 99 pupae | 0.973 | 1.0 % |

Notes

- The vein model must be applied to a wing crop, not to the full frame: the crop
  is produced by `wing_unet_v1` with an 18 % margin around the wing bounding box.
  `app/core/vein.py` implements this two-stage pipeline.
- Thresholds were chosen by sweeping 0.3–0.9 on held-out images and minimising
  the median absolute area error, not by maximising Dice.
- Area is computed from the traced outline in **original image pixels**, so it is
  directly comparable to ImageJ's *Area* measurement.

## Retraining

`trainers/train_unet.py` reproduces the training recipe (BCE + Dice, AdamW with a
0.15× learning rate on the encoder, cosine schedule, scale jitter, flips and 90°
rotations) and exports `model.onnx` alongside the checkpoints.

```bash
python trainers/train_unet.py --data /path/to/dataset --out ./runs/eye_v5 \
       --epochs 60 --bs 3 --size 768 --export
```

Dataset layout expected by the script:

```
dataset/
├── images/        <id>.jpg
├── masks/         <id>.png   (255 = target)
├── splits.csv     id,phenotype,genotype,split      split ∈ {train, val, test}
└── annotation.csv id,file
```
