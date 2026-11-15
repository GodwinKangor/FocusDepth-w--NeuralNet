# Capture protocol

1. Tripod, static scene, stable indoor lighting, flash off. No people or moving objects.
2. Launch FocusDepth and let auto-exposure settle on the scene before pressing capture.
   The app then locks exposure (current duration/ISO) and white balance for the sweep.
3. Default sweep: 10 lens positions evenly spaced over [0, 1]. Narrow the range if the
   whole scene lies in part of it (check which frames look sharp).
4. After capture, open the capture in **Captures** and confirm that focus visibly changes
   across frames. Check `manifest.json`: `lens_position_actual` should track
   `lens_position_requested`. If it doesn't, increase the settle time.
5. Measure the distance from the phone to each object of interest with a tape measure and
   record them in `data/annotations/<scene_id>.json` (bounding boxes in `reference_full.png` pixels).
6. Copy the folder to `data/raw/<scene_id>/` via Files, AirDrop (Share capture .zip) or Finder.

Record per scene: lighting, distances, which surfaces are textured/textureless, anything unusual.

## Fallback

If discrete still capture is unreliable (focus not settling, inconsistent processing),
capture a slow focus-sweep video and extract frames with matching lensPosition metadata.

## Camera capability report

On launch, the app saves `camera_capabilities.json` in its Documents directory.
Use **Share camera capabilities** in the capture form to export it. A fresh copy
is also included with every focus sweep. Run this on the physical iPhone 15 Pro;
the simulator cannot establish the phone's sensor capabilities.

The report lists rear main, ultrawide, telephoto, and LiDAR devices discovered by
AVFoundation, manual lens-position support, exposure/white-balance controls,
advertised video/depth formats, and supported multicamera device sets. Device IDs
link each supported set to its device entries.

This is a capability inventory, not a successful simultaneous-capture test. Next,
inspect the exported report and validate selected formats, synchronized RGB/depth,
manual focus behavior, and hardware/system-pressure budgets on the phone.

For metric experiments, use separate calibration and evaluation targets. Record
measured target-plane distances, the phone reference point used for measurement,
and measurement uncertainty. Do not score calibration targets as held-out results.
