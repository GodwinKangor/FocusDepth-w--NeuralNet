# Preliminary ordering analysis

Recomputed from saved maps with `scripts/analyze_ordering.py`. No model inference or parameter tuning was performed. Labels are visual estimates, not measured ground truth.

| Capture | Method | Correct / eligible pairs |
|---|---|---|
| `scene_01_20261102_180021` | B1 optical | 9/9 |
| `scene_01_20261102_180021` | B2 neural | 9/9 |
| `scene_01_20261102_180021` | B3 fixed fusion | 9/9 |
| `scene_01_20261102_180021` | Confidence fusion | 9/9 |
| `scene_01_20261102_182153` | B1 optical | 4/6 |
| `scene_01_20261102_182153` | B2 neural | 6/6 |
| `scene_01_20261102_182153` | B3 fixed fusion | 6/6 |
| `scene_01_20261102_182153` | Confidence fusion | 6/6 |

## Near-focus failure case

Capture `scene_01_20261102_182153` contains a can annotated as never sharp. Optical estimation reverses its order relative to the thermometer label and pink box. Neural estimation and both fusion methods recover all six provisional region orderings.

The can has mean optical confidence 0.042, so confidence fusion gives approximately 95.8% mean weight to the neural estimate within that box. This supports the intended suppression of weak optical evidence. It does not show superiority over neural estimation or fixed fusion, which also score 6/6.

The aligned neural median for the can is approximately −0.984, outside the nominal lens-control range. The affine alignment extrapolates; this is a relative coordinate and cannot be interpreted as physical distance. Region medians are computed after pixelwise fusion, so they are not simply weighted sums of input medians.

## Next capture

Collect a scene containing both (1) an object too near to focus and (2) textured objects within the focus range for which neural ordering may fail. Measure object distances independently, keep lighting and camera position fixed, and repeat three sweeps. Keep the present thresholds fixed. Record failures as well as successes; do not select only examples where fusion wins. A benefit over neural and fixed fusion remains unproven.
