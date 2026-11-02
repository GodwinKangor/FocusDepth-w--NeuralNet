"""Per-pixel confidence in the optical focus estimate.

Two ingredients:
  * peak distinctness: how clearly one focus setting stands out in the
    pixel's sharpness profile (prominence or 1 - normalized entropy);
  * texture gate: whether the pixel has enough sharpness signal at all.

High sharpness alone is not high confidence: misalignment, specular highlights
and noise can all create strong but misleading peaks.
"""
from __future__ import annotations

import numpy as np

EPS = 1e-8


def peak_prominence(sharpness: np.ndarray) -> np.ndarray:
    """(max - median) / max of each pixel's sharpness profile; 0 for flat profiles."""
    top = sharpness.max(axis=0)
    med = np.median(sharpness, axis=0)
    return np.clip((top - med) / (top + EPS), 0.0, 1.0).astype(np.float32)


def entropy_confidence(sharpness: np.ndarray) -> np.ndarray:
    """1 - H(p) / log(N), where p is the sharpness profile normalized to sum to 1."""
    n = sharpness.shape[0]
    p = np.maximum(sharpness, 0.0)
    p = p / (p.sum(axis=0, keepdims=True) + EPS)
    h = -(p * np.log(p + EPS)).sum(axis=0)
    return np.clip(1.0 - h / np.log(n), 0.0, 1.0).astype(np.float32)


def texture_gate(sharpness: np.ndarray, low: float = 0.05, high: float = 0.25, ref_percentile: float = 99.0) -> np.ndarray:
    """Smoothstep on peak sharpness, relative to a high percentile of the image.

    Using a relative reference keeps thresholds stable across exposure and
    image scale; ``low``/``high`` should be tuned on calibration scenes only.
    """
    top = sharpness.max(axis=0)
    ref = np.percentile(top, ref_percentile) + EPS
    t = np.clip((top / ref - low) / max(high - low, EPS), 0.0, 1.0)
    return (t * t * (3.0 - 2.0 * t)).astype(np.float32)


def optical_confidence(
    sharpness: np.ndarray,
    measure: str = "prominence",
    use_texture_gate: bool = True,
    gate_low: float = 0.05,
    gate_high: float = 0.25,
) -> np.ndarray:
    if measure == "prominence":
        conf = peak_prominence(sharpness)
    elif measure == "entropy":
        conf = entropy_confidence(sharpness)
    else:
        raise ValueError(f"unknown confidence measure: {measure}")
    if use_texture_gate:
        conf = conf * texture_gate(sharpness, gate_low, gate_high)
    return conf.astype(np.float32)
