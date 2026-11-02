# FocusDepth

Confidence-guided fusion of focus-based (optical) depth and pretrained neural depth on iPhone.
CS73 Computational Photography, Fall 2026.

**Question:** when does combining depth-from-focus with a pretrained monocular model
(Depth Anything V2) beat either alone, especially on smooth surfaces and object boundaries?

Methods compared: **B1** optical only, **B2** neural only, **B3** fixed-weight fusion,
and **confidence fusion**, where the optical weight is a per-pixel focus-confidence map.

## Layout

```
ios/FocusDepth/   SwiftUI + AVFoundation focus-sweep capture app (writes frames + manifest.json)
python/           registration -> sharpness -> optical depth + confidence -> neural -> alignment -> fusion -> evaluation
configs/          default.yaml (all tunable parameters)
data/             raw captures, processed data, annotations (only annotations are tracked)
results/          per-scene outputs (not tracked)
tests/            unit tests for focus depth, confidence, registration, alignment, fusion
docs/             capture protocol, experiment plan, limitations
```

## Python setup

```bash
python3 -m venv ~/.venvs/focusdepth      # see note below
source ~/.venvs/focusdepth/bin/activate
pip install -r requirements.txt
pytest -q
```

> `python -m venv` refuses to create a venv inside a path containing `:`, and this repo's
> parent folder is `FocusDepth-w:-NeuralNet`. Either rename that folder or keep the venv
> outside the repo as above.

## Running

Smoke test without a phone or PyTorch (synthetic stack plus a stand-in neural map):

```bash
python python/synthetic.py --output data/raw/synthetic_01
python python/run_pipeline.py --input data/raw/synthetic_01 --output results/synthetic_01 \
    --neural-depth data/raw/synthetic_01/neural_depth.npy
```

Real capture (copy the capture folder from the phone into `data/raw/`):

```bash
python python/run_pipeline.py --input data/raw/desk_01 --output results/desk_01 \
    --model depth-anything-v2-small --fusion all
```

Outputs: `reference_rgb.png`, `reference_full.png`, `optical_depth.{npy,png}`,
`optical_confidence.{npy,png}`, `neural_depth.{npy,png}`, `neural_depth_raw.npy`,
`fixed_fusion.{npy,png}`, `confidence_fusion.{npy,png}`, `comparison.png`, `metrics.json`.

If `data/annotations/<scene_id>.json` exists, `metrics.json` also includes pairwise
depth-ordering accuracy for every method (format in `python/evaluate.py`).

## Conventions

- Depth maps use **larger = farther** throughout.
- The optical map is in **AVFoundation lensPosition units** (0 = nearest focus, 1 = farthest),
  not meters. Neural depth is mapped into the same units by a confidence-weighted robust affine fit
  (`python/depth_alignment.py`); see that module for why affine is a reasonable model.
- LiDAR, if used, is only an evaluation reference, never a fusion input.
- Thresholds are tuned on calibration scenes only, never on evaluation scenes.

## iOS app

The Swift sources are in `ios/FocusDepth/`. To build:

1. In Xcode, create a new iOS App project named FocusDepth (SwiftUI, iOS 17+) inside `ios/`,
   delete its generated `ContentView.swift` / app file, and add the folders from `ios/FocusDepth/`.
2. Info.plist keys: `NSCameraUsageDescription`, `UIFileSharingEnabled = YES`,
   `LSSupportsOpeningDocumentsInPlace = YES` (captures then appear in the Files app).
3. Use Swift 5 language mode (or fix strict-concurrency diagnostics under Swift 6).

Use a tripod. See `docs/capture_protocol.md`.

## Models and licensing

Depth Anything V2 **Small** is Apache-2.0; Base/Large are CC-BY-NC-4.0. Weights are downloaded
from Hugging Face on first run and are not stored in this repo.

## References

- Suwajanakorn, Hernández, Seitz. *Depth from Focus with Your Mobile Phone.* CVPR 2015.
- Yang et al. *Depth Anything V2.* NeurIPS 2024. arXiv:2406.09414.
- CS73/273 Assignment 3 (depth-from-focus baseline).
