# Limitations

- **No metric depth from focus.** lensPosition is a normalized control value. Without
  per-device calibration the optical map is a focus coordinate, so results are ordinal / relative.
- **Affine alignment is approximate.** It assumes lensPosition and the model's output are both
  roughly affine in inverse distance. Check `alignment.residual_mad` and the fit's Spearman value.
- **Alignment uses the scene's own confident optical pixels.** If confidence is wrong (highlights,
  misregistration), the neural map inherits that error.
- **Sharpness is not confidence.** Specular highlights, noise and misalignment create false peaks.
- **Focus breathing / ISP variation.** Magnification changes with focus; registration is affine and
  per-frame processing may still vary.
- **Registration** chains ECC between neighbouring frames; failures fall back to identity and are
  reported in `metrics.json -> registration.failed`. Inspect those frames.
- **Monocular depth** is relative, can be confidently wrong, and is evaluated only on a few scenes.
