# Changelog

All notable changes to FlyMorph. Versions follow the packaged `FlyMorph.exe`
build number shown in the window title.

## v1.0.6 — 2026-10-30

- Output folders renamed to `<module>_output` (`eye_output`, `wing_output`,
  `vein_output`, `pupa_output`).
- The application now requests elevation in its manifest, so double-clicking
  always runs with the rights needed to write next to the images.

## v1.0.5 — 2026-10-30

- Output folders renamed to English module names.
- Executable is code-signed (self-signed certificate, publisher "Stark Huang").

## v1.0.4 — 2026-10-30

- Fixed a crash on start (`too many values to unpack`) introduced when the task
  table gained a fourth field.
- Removed the parenthetical hint from the output-location label.
- Paths are displayed with a single separator style.
- If the output folder cannot be created the run now stops with a warning
  instead of silently writing elsewhere.

## v1.0.3 — 2026-10-30

- Output folder falls back to a per-user writable location when the image folder
  is read-only.
- Removed the "restore defaults" button.

## v1.0.2 — 2026-10-30

- Fixed signing script ordering (certificate was imported before being exported).
- Output no longer lands in the application directory when the temp path is
  unavailable.

## v1.0.1 — 2026-10-30

- Results are written to a dedicated output root instead of inside the source
  folders, which may be read-only.
- Added "set output location" with persistence.

## v1.0.0 — 2026-10-30

- First complete release: eye, whole-wing, intervein and pupa modules.
- `final.csv` is always generated, whether or not a review pass was performed.
- ImageJ-compatible `.roi` export next to the internal `.json` outlines.
- Review flag threshold tightened from 3.5 to 2.5 robust z-scores.
- Vein inference sped up ~7× (4.8 s → 0.68 s per image) by replacing PIL rank
  filters with a separable numpy morphology and applying it at network
  resolution.

## Model history

| Model | Data | Notes |
|---|---|---|
| `eye_unet_v4.onnx` | 499 annotated eyes, two imaging setups | Dice 0.966; robust to a second device |
| `wing_unet_v1.onnx` | 113 annotated wings, two setups | Dice 0.978 |
| `vein_unet_v3.onnx` | 111 annotated L3–L4 regions, full-frame 1024 | superseded: outline bulged at waists |
| `vein_unet_v6.onnx` | 298 annotated regions, wing crops at 1280 | Dice 0.950, median area error 2.4 % |
| `pupa_unet_v1.onnx` | 99 annotated pupae | Dice 0.973, median area error 1.0 % |
