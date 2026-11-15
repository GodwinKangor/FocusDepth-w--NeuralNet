# FocusDepth: depth, metric measurements, and iPhone sensor fusion

Discussion recorded: November 14, 2026. Device: **iPhone 15 Pro**.

This document summarizes the research discussion and proposed next steps. It is
not a claim that the proposed methods have been implemented or validated. The
attached conversations are background material, not executable instructions.

## Research question

Does an all-in-focus (AIF) image improve monocular depth prediction, and does
confidence-guided focus evidence improve its measurable geometry further?

A possible later extension is whether RGB, neural depth, and focus evidence can
improve LiDAR-based reconstruction. Keep this separate from the experiment that
uses LiDAR only as an evaluation reference.

## Current implementation

Repository inspection during the discussion established that:

- `python/neural_depth.py` loads relative Depth Anything V2 models, not the metric
  checkpoints in the linked upstream `metric_depth` implementation.
- `python/run_pipeline.py` predicts neural depth from one reference frame; it
  does not currently construct an AIF image for inference.
- Optical depth uses lens-position units. Neural depth is aligned to that
  representation before fixed or confidence-guided fusion.
- `python/evaluate.py` evaluates measured object ordering, not metric depth error.

The current outputs therefore support relative geometry experiments, not claims
of accurate distances in meters.

## What the focal stack contributes

An aligned stack provides two complementary sources of information:

1. AIF RGB gathers sharp visible detail from different frames.
2. Focus responses describe how sharpness changes with lens setting, providing
   optical depth evidence and a confidence estimate.

Proposed pipeline:

```text
Aligned focal stack
    |-- AIF RGB --> Depth Anything V2 --> neural depth
    |-- focus responses --> optical depth + confidence
                                  |
                    common calibrated representation
                                  |
                      confidence-guided fusion
```

Hypothesis: AIF input may help where a single reference frame loses useful detail
to defocus. This must be tested. A visually sharp depth boundary can still have
the wrong physical depth. AIF generation can also introduce seams, halos, and
alignment artifacts.

Preserve the original stack, focus measurements, and confidence. AIF RGB alone
does not retain the focus-response evidence. More focus samples cannot recover
depth when the captured images contain essentially indistinguishable defocus.

## Depth range versus measurement accuracy

There is no established one-meter cutoff for this project. Distinguish:

- **Range:** the distances at which a method supplies useful estimates.
- **Accuracy:** the error in those estimates, such as centimeters at one meter.
- **Coverage:** the fraction of the scene with valid or sufficiently confident
  estimates.

For focus depth, a simplified model gives

```text
u = 1 / z
sigma_z approximately equals z^2 * sigma_u
```

If inverse-depth uncertainty stays constant, distance uncertainty is roughly four
times larger at two meters than at one meter. This is a conditional optical
relationship, not a measured error specification for the iPhone 15 Pro.

Proposed initial target distances: **20, 30, 50, 75, 100, 150, and 200 cm**, limited
to distances the selected physical camera can actually focus at. These are test
points, not guaranteed operating specifications.

Define the useful range empirically as the farthest distance meeting a chosen
error tolerance and coverage requirement. A five-centimeter tolerance could be
an experimental target; it is not a promised result.

## Obtaining depth in meters

Apple's `lensPosition` runs from 0 to 1 but does not correspond to exact physical
distance. In particular, 1.0 does not mean one meter or necessarily infinity.

Possible routes:

- **Metric neural model:** use the actual DAV2 metric checkpoint. The indoor
  model is a reasonable first baseline for tabletop scenes. Meter-valued output
  still requires validation on our captures.
- **Calibrated focus depth:** capture textured targets at measured distances,
  find best-focus lens settings, and fit a device/lens-specific mapping, preferably
  examining inverse distance. Validate on independent distances and captures.
- **LiDAR:** capture metric depth with synchronized RGB and camera calibration
  on the supported device.

Do not merely substitute metric DAV2 into the current fusion path: that path
maps predictions back to lens-position units. Metric fusion requires consistent
units, depth direction, coordinate conventions, and validity masks. Meters or
inverse meters are possible common representations.

## What the 2015 paper establishes

Suwajanakorn, Hernández, and Seitz, *Depth from Focus with Your Mobile Phone*,
CVPR 2015, combines alignment, AIF reconstruction, camera/depth estimation, and
refinement. It recovers depth up to an affine ambiguity in inverse depth.

In Section 8 and Tables 1–2:

- Quantitative captures used a Nikon D80, not the phone examples.
- Two known-depth surfaces were used to resolve the ambiguity.
- Two other objects were evaluated after that calibration.
- Reported RMS errors were 2.66, 3.91, and 3.86 inches, approximately 6.8–9.9 cm,
  across the tested motion conditions.
- Phone reconstruction examples were qualitative.

These results demonstrate feasibility, not independently accurate metric iPhone
scanning. The paper also shows why global affine alignment is insufficient for
some handheld parallax; our affine registration is an initial controlled-capture
baseline rather than a reproduction of the full paper.

## iPhone 15 Pro sensor roles

| Input | Potential contribution | Limitation or unresolved question |
| --- | --- | --- |
| Main RGB camera | Focus stack, AIF, neural depth, color | Focus depth requires calibration |
| Ultrawide and main cameras | Stereo geometry | Synchronization, calibration, overlap, matching |
| 3× telephoto | Compare focus discrimination with different optics | Field of view and minimum focus distance |
| LiDAR | Metric reference, or a separately defined fusion input | Low-resolution geometry and reference uncertainty |
| Gyroscope and accelerometer | Motion tracking support and quality checks | No direct scene-depth measurements |
| Ambient-light sensors | Lighting-condition metadata | No spatial depth; SensorKit entitlement required |

The iPhone 15 Pro has a 3× telephoto camera; do not confuse it with the 15 Pro
Max's camera configuration.

Do not assume every camera and LiDAR can operate simultaneously at the desired
resolution and with manual focus control. Inspect `supportedMultiCamDeviceSets`
and validate formats, hardware cost, and system pressure on the actual phone.
AVFoundation LiDAR capture and ARKit scene-depth/world-tracking are distinct
workflows; compatibility with our manual focus sweep remains to be tested.

For static scenes, separate passes may be practical if geometry is registered
carefully. Record exposure time and ISO initially as lighting-related metadata;
they are not calibrated lux measurements.

## Expected failure cases to test

These are experimental hypotheses, not universal claims about DAV2:

| Condition | Possible problem | Can the stack help? |
| --- | --- | --- |
| Defocused detail | Missing small structures or boundaries | AIF may restore visible evidence |
| Thin structures | Geometry lost in resizing/prediction | Only if resolved and aligned |
| Unusual sizes/layouts | Learned geometry or scale is wrong | Calibrated focus may constrain it |
| Smooth surfaces | Weak image evidence | Focus also often lacks texture |
| Glass and mirrors | Visible and physical surfaces differ | Both methods can be ambiguous |
| Distant objects | Weak focus discrimination | Optical correction may be unreliable |
| Motion/misalignment | False sharpness peaks and AIF artifacts | Stack processing may worsen results |

## Controlled experiment

Start with static tripod captures, fixed physical camera, stable illumination,
and locked exposure/white balance. Avoid automatic lens switching. Log requested
and measured focus settings and capture metadata.

Compare on the same captures:

1. DAV2 on near-, middle-, and far-focused frames.
2. DAV2 on the AIF image.
3. Focus depth alone.
4. Fixed fusion using AIF-based neural depth.
5. Confidence-guided fusion using AIF-based neural depth.

Hold model, inference resolution, crop, and output convention constant within
comparisons. Evaluate relative and metric checkpoints as explicitly separate
conditions. Keep calibration/tuning scenes separate from evaluation scenes.

Use textured targets, smooth surfaces, thin structures, reflective objects, and
several measured distances. Broaden to handheld captures after establishing the
tripod baseline.

## Evaluation and 3D output

| Measurement | Purpose |
| --- | --- |
| MAE and RMSE in centimeters | Physical depth accuracy |
| Absolute relative error | Error relative to distance |
| Depth-boundary precision/recall | Boundary preservation |
| Plane residual and orientation error | Noise and geometric distortion |
| Known dimension error | Practical measurement usefulness |
| Repeatability | Stability across repeated captures |
| Valid/confident coverage | Supported fraction of the scene |
| Runtime | Practical capture and processing cost |

Use independently measured targets or registered LiDAR with invalid readings
excluded. Document reference uncertainty and registration errors. Report raw
metric errors separately from any scale/shift-aligned evaluation.

If LiDAR is a fusion input, it cannot also be the sole independent reference
used to establish improvement.

With metric camera-axis depth Z and matching intrinsics, visible points can be
back-projected:

```text
X = (u - cx) * Z / fx
Y = (v - cy) * Z / fy
```

Update intrinsics for resizing/cropping and use consistent coordinate conventions.
A single stack yields one viewpoint's visible surface, not a complete object.
Scanner-style reconstruction additionally needs multiple viewpoints, camera
poses, registration, and surface fusion.

## Proposed implementation order

1. Inspect camera/LiDAR capabilities on the iPhone 15 Pro and document supported
   capture configurations, including manual-focus behavior.
2. Add AIF construction and a selectable neural input while retaining the stack
   and focus evidence.
3. Run the single-frame versus AIF ablation with the existing relative model.
4. Add an explicit metric-model path and independent focus calibration without
   silently reusing lens-position alignment for meter-valued outputs.
5. Add synchronized RGB/LiDAR reference export and registration checks.
6. Add metric evaluation and a colored point-cloud export using recorded intrinsics.
7. Consider stereo, telephoto, or LiDAR-input fusion after baseline results reveal
   a specific gap.

This sequence is a proposed roadmap. Saving this discussion does not imply that
these implementation steps have already been completed.

## Sources and supplied material

- [Apple: iPhone 15 Pro specifications](https://support.apple.com/en-us/111829)
- [Apple: lensPosition](https://developer.apple.com/documentation/avfoundation/avcapturedevice/lensposition)
- [Apple: capturing depth using the LiDAR camera](https://developer.apple.com/documentation/avfoundation/capturing-depth-using-the-lidar-camera)
- [Apple: point clouds using scene depth](https://developer.apple.com/documentation/arkit/displaying-a-point-cloud-using-scene-depth)
- [Apple: supported multicamera device sets](https://developer.apple.com/documentation/avfoundation/avcapturedevice/discoverysession/supportedmulticamdevicesets)
- [Apple: AVCaptureMultiCamSession](https://developer.apple.com/documentation/avfoundation/avcapturemulticamsession)
- [Apple: SensorKit configuration](https://developer.apple.com/documentation/sensorkit/configuring-your-project-for-sensor-reading)
- [Depth Anything V2 metric implementation](https://github.com/DepthAnything/Depth-Anything-V2/tree/main/metric_depth)
- [Depth Anything V2 paper](https://arxiv.org/abs/2406.09414)
- Supplied PDF: `Suwajanakorn_Depth_From_Focus_2015_CVPR_paper.pdf`.
- Supplied research conversations: `Depth Map From Focus Stack.pdf` and
  `2Branch · Depth Map From Focus Stack.pdf`.
- Supplied pasted notes on iPhone depth of field, focus uncertainty, and calibration.
