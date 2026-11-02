# Experiments

## Scenes

| Scene | Setup | Question |
|---|---|---|
| Textured | Books / patterned objects at different distances | Does optical focus add useful evidence? |
| Textureless | Smooth box, wall, tabletop | Does neural depth help when focus confidence is weak? |
| Boundaries | Object in front of a wall | Does fusion preserve foreground/background separation? |
| Failure (optional) | Reflective or dim scene | When do both methods struggle? |

Keep **calibration scenes** (used to tune `confidence.gate_*` and `alignment.conf_threshold`)
separate from **evaluation scenes**.

## Metrics

- Pairwise depth-ordering accuracy over annotated regions (measured distances), per method.
- Mean optical confidence per region, split textured / textureless.
- Qualitative crops: RGB, optical, neural, fixed fusion, confidence fusion, confidence.
- Capture time, processing time per stage (`metrics.json -> timings_s`), number of frames.

## Ablations

- Fixed vs confidence weights (`--fusion all` always produces both).
- Texture gate on/off: `confidence.texture_gate: false`.
- Prominence vs entropy confidence: `confidence.measure`.
- Sparse stacks: subsample 3/5/8/12 frames from one capture (edit the manifest copy).

## Log

| Date | Scene | Notes |
|---|---|---|
