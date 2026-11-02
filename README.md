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

Open the checked-in project:

```bash
open ios/FocusDepth.xcodeproj
```

1. Select the **FocusDepth** scheme. The app targets iPhone running iOS 17 or later.
2. For a physical iPhone, open **FocusDepth target > Signing & Capabilities**, select your
   development team, and change the bundle identifier if your team needs a unique one.
3. Select your connected iPhone as the run destination and press **Run** (Cmd-R). Allow
   camera access when prompted. A physical device is required for focus sweeps; the
   simulator can build the app but has no supported back camera.

Camera permission, Files app sharing, a launch screen, and Swift 5 language mode are
configured in the project. Captures appear under **On My iPhone > FocusDepth > Captures**.

The Xcode navigator also includes the Python pipeline, configs, tests, and documentation.
Depth estimation and evaluation still run on the Mac using the Python commands above;
these files are not compiled into the iOS app.

To check the simulator build from Terminal without setting up signing:

```bash
DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer xcodebuild \
    -project ios/FocusDepth.xcodeproj -scheme FocusDepth \
    -sdk iphonesimulator -configuration Debug \
    -derivedDataPath /tmp/focusdepth-derived CODE_SIGNING_ALLOWED=NO build
```

Use a tripod. See `docs/capture_protocol.md`.

## Models and licensing

Depth Anything V2 **Small** is Apache-2.0; Base/Large are CC-BY-NC-4.0. Weights are downloaded
from Hugging Face on first run and are not stored in this repo.

## References

- Suwajanakorn, Hernández, Seitz. *Depth from Focus with Your Mobile Phone.* CVPR 2015.
- Yang et al. *Depth Anything V2.* NeurIPS 2024. arXiv:2406.09414.
- CS73/273 Assignment 3 (depth-from-focus baseline).
