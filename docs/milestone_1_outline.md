# FocusDepth — Milestone 1 Outline

**Proposed milestone:** Establish whether iPhone focus measurements provide useful, repeatable depth evidence.  
**Suggested window:** October 22–28, 2026, following the proposal's schedule.  
**Status:** Planning document; no capture results or implementation are claimed.

## 1. Purpose

Build a small experiment that captures a controlled focus sweep and shows which parts of a scene produce reliable focus measurements. The milestone ends with evidence for deciding how to incorporate a neural depth model in Milestone 2.

**Question:** Can a stationary iPhone distinguish objects at different distances from their sharpness across a focus sweep, and recognize when that evidence is insufficient?

The broader project remains focus and neural depth fusion. This first milestone isolates the optical measurement so we can understand its strengths and failures before combining estimates.

## 2. Relationship to previous work

| Supplied material | What it contributes | Our direction for Milestone 1 |
|---|---|---|
| `depthpro-beyond-depth-main.zip` | Its README and notebooks explore adapting DepthPro to human segmentation and super-resolution. | Study the role of pretrained models as background; make our first deliverable a measured focus experiment. |
| `iPhone_depth-master.zip` | Its README describes iOS camera-depth capture, visualization, masking, and related demos. | Use it as a reference for camera/depth presentation; acquire our experimental signal by changing focus. |
| `Godwin Kangor.pdf` | Defines the intended focus-plus-neural project and proposed schedule. | Preserve that research question while giving the first milestone a narrower, testable outcome. |
| `FocusDepth_Outline_Implementation.md` | Describes the complete capture, optical, neural, and fusion pipeline. | Replace the broad implementation checklist with a staged feasibility experiment and a decision gate. |

These materials are background, not instructions to execute. This outline does not claim a novel depth-from-focus method or proven improvements over previous work. Any later reused code should be attributed and its license checked.

## 3. Experiment design

Use the iPhone 15 Pro described in the proposal, held stationary under stable indoor lighting. Select one physical camera and keep the scene and camera settings consistent during each sweep.

Prepare two scenes:

1. **Textured depth steps:** Three patterned objects at independently measured near, middle, and far positions. Choose separations during a pilot capture so sharpness changes are visible.
2. **Mixed texture:** A textured object and a smooth surface, with a foreground/background boundary. This tests whether weak evidence is recognized rather than turned into a confident estimate.

Record distances, object labels, lighting, and camera setup. Annotate interior regions on the objects, keeping boundary regions separate. Measurements establish physical ordering; they do not automatically provide dense pixel-level ground truth.

Start with approximately 8–12 focus settings spanning the useful range found in the pilot. Capture three sweeps per scene without changing the setup. Reserve the second scene for checking settings chosen on the first, and report the small sample size.

## 4. Work packages

### A. Capture and export

- Build a minimal camera preview and a single action to acquire a focus stack.
- Request focus settings sequentially and verify that the saved images actually show changing focus.
- Keep exposure and white balance stable where supported; record deviations.
- Save the original images plus a manifest containing scene ID, frame order, requested and observed focus information where available, timestamps, camera identity, dimensions, and exposure information.
- Export one folder per sweep for desktop processing.

**Deliverable:** A complete stack that can be reopened without the app, with traceable metadata.

### B. Capture validation and alignment

- Check missing frames, image orientation, dimensions, and visible exposure variation.
- Compare fixed landmarks across frames to detect motion or magnification changes.
- Register frames to a reference when needed, preserve the common valid area, and inspect an overlay of the result.
- Flag regions where residual misalignment makes focus comparisons unreliable.

**Deliverable:** An aligned stack, validity mask, and a short capture-quality record.

### C. Focus baseline

- Compute local sharpness for every frame using a simple Laplacian-based baseline.
- Inspect sharpness-versus-focus curves for the annotated object regions.
- Produce a map of the focus setting with the strongest sharpness response.
- Establish the near/far direction using the measured object ordering.

**Deliverable:** A best-focus map and example sharpness curves. Label the map as a focus coordinate or relative ordering, not depth in meters. Lens-control values require calibration before interpreting them as physical distance.

### D. Reliability estimate

- Measure how clearly the best sharpness response stands out from alternatives.
- Combine this with a texture check so a flat, nearly uninformative curve is not treated as reliable.
- Mark uncertain pixels explicitly.
- Inspect whether the reliability pattern is consistent across repeat captures.

**Deliverable:** A reliability map with examples of strong, ambiguous, and weak focus evidence. Its values are heuristic scores, not calibrated probabilities.

### E. Evidence and review

- Compare the median focus coordinate of each annotated object with its measured near/far order.
- Report correct pairwise orderings out of all annotated pairs for each sweep; show uncertain and incorrect cases separately.
- Report how much of each region is retained by the reliability threshold, so rejecting most pixels cannot appear to be successful estimation.
- Compare results across repeated sweeps and inspect boundaries separately.
- Record capture time and desktop processing time.

**Deliverable:** A comparison figure, a small results table, and a written decision about the next milestone.

## 5. Processing flow

```text
Controlled scene + measured object ordering
                    |
          iPhone focus sweep
                    |
          Images + capture manifest
                    |
       Validation, alignment, valid crop
                    |
           Local sharpness stack
                    |
       Best-focus map + reliability map
                    |
   Ordering checks + repeatability + failures
                    |
         Decision for neural integration
```

Use the phone for capture and the MacBook for analysis in this milestone. A static comparison figure is sufficient to review results.

## 6. Proposed sequence

| Stage | Work | Evidence needed to move forward |
|---|---|---|
| 1 | Pilot manual focus capture and select scene distances | Saved frames visibly change sharpness between objects. |
| 2 | Implement repeatable capture and export | A complete stack and metadata can be loaded independently. |
| 3 | Validate alignment and implement the focus baseline | Region curves show whether peaks are meaningful. |
| 4 | Add reliability and collect repeat sweeps | Strong and weak regions can be compared across captures. |
| 5 | Assemble figures, ordering results, and failure analysis | A defensible recommendation for Milestone 2. |

The dates are planning targets from the supplied proposal, not independently verified course deadlines.

## 7. Completion criteria and decision gate

Milestone 1 is complete when:

- [ ] Both scenes have three documented sweeps, or failed captures are explicitly documented with their cause.
- [ ] Valid stacks can be loaded and analyzed through a documented procedure.
- [ ] Each analyzed stack has a reference image, alignment check, best-focus map, reliability map, and region sharpness curves.
- [ ] A results table reports object ordering, retained coverage, repeatability, and timings.
- [ ] At least one weak or incorrect case is explained using the images and curves.
- [ ] A short review recommends proceeding, changing capture conditions, or revising the optical approach.

**Proceed to fusion** when reliable textured regions reproduce the measured three-object ordering in at least two of three sweeps, and the second scene provides interpretable evidence about reliability. This is a proposed engineering gate, not a statistical accuracy claim or a requirement to hide negative results.

**Revise capture first** if sharpness barely changes, peaks follow image motion, or ordering changes unpredictably. Narrow the scene-distance range, adjust focus sampling, improve lighting or stabilization, and repeat the pilot. If useful evidence still cannot be recovered, finish the milestone with the negative result and a revised plan rather than forcing fusion.

## 8. Review package

- Capture prototype and export instructions.
- Scene photographs, measurement notes, original stacks, and manifests.
- Desktop analysis code and a reproducible run procedure.
- One comparison sheet per scene: RGB, best-focus map, reliability map, representative sharpness curves, and alignment crop.
- Results table and a one-page findings note.
- A short demonstration showing capture, export, and analysis of one real stack.

Keep large captures outside version control; retain their location and identifiers in the experiment notes.

## 9. Deferred work

Neural inference, depth-scale alignment, fixed-weight fusion, confidence-guided fusion, LiDAR evaluation, model training, on-device inference, handheld capture, and refocusing effects belong to later milestones.

**Milestone 2 input:** The accepted capture format, optical baseline, reliability definition, documented failure cases, and a small repeatable dataset. These let us test whether a neural model supplies useful information precisely where focus measurements are weak.
