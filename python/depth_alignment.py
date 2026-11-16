"""Bring neural depth into the optical focus-coordinate frame.

Convention throughout the project: larger values mean farther.

Why an affine fit is reasonable: for a thin lens, the lens displacement needed
to focus at distance z is approximately linear in 1/z (diopters), and
Depth Anything V2's relative output is affine in 1/z as well. So
lens_position ~= a * neural + b, with a < 0 for raw (larger = nearer) output.
Both relationships are approximations; check residuals per scene.

The fit uses only high-confidence optical pixels. Thresholds must be chosen on
calibration scenes, never on the held-out evaluation scenes.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.stats import spearmanr


@dataclass
class AlignmentParams:
    scale: float
    shift: float
    n_pixels: int
    spearman: float
    residual_mad: float

    def to_dict(self) -> dict:
        return asdict(self)


def robust_normalize(d: np.ndarray, lo: float = 1.0, hi: float = 99.0) -> np.ndarray:
    """Map the [lo, hi] percentile range to [0, 1], clipping outliers (for display/comparison)."""
    p_lo, p_hi = np.percentile(d, [lo, hi])
    return np.clip((d - p_lo) / max(p_hi - p_lo, 1e-8), 0.0, 1.0).astype(np.float32)


def _sample(mask: np.ndarray, max_samples: int, seed: int = 0) -> np.ndarray:
    idx = np.flatnonzero(mask)
    if idx.size > max_samples:
        idx = np.random.default_rng(seed).choice(idx, max_samples, replace=False)
    return idx


def depth_direction(reference: np.ndarray, other: np.ndarray, mask: np.ndarray | None = None, max_samples: int = 50_000) -> float:
    """Spearman correlation between two maps; negative means opposite near/far conventions."""
    mask = np.ones(reference.shape, bool) if mask is None else mask
    idx = _sample(mask, max_samples)
    if idx.size < 10:
        return float("nan")
    return float(spearmanr(reference.ravel()[idx], other.ravel()[idx]).statistic)


def fit_affine(src: np.ndarray, dst: np.ndarray, weights: np.ndarray | None = None, huber_k: float = 1.345, iterations: int = 10) -> tuple[float, float]:
    """Weighted Huber-IRLS fit of dst ~= scale * src + shift."""
    w0 = np.ones_like(src) if weights is None else weights.astype(np.float64)
    A = np.stack([src, np.ones_like(src)], axis=1).astype(np.float64)
    w = w0.copy()
    coef = np.zeros(2)
    for _ in range(iterations):
        sw = np.sqrt(w)
        coef, *_ = np.linalg.lstsq(A * sw[:, None], dst * sw, rcond=None)
        r = dst - A @ coef
        s = 1.4826 * np.median(np.abs(r - np.median(r))) + 1e-12
        u = np.abs(r) / (huber_k * s)
        huber_weights = np.ones_like(u)
        np.divide(1.0, u, out=huber_weights, where=u > 1.0)
        w = w0 * huber_weights
    return float(coef[0]), float(coef[1])


def align_neural_to_optical(
    neural: np.ndarray,
    optical: np.ndarray,
    confidence: np.ndarray,
    conf_threshold: float = 0.5,
    min_pixels: int = 500,
    max_samples: int = 200_000,
    params: AlignmentParams | None = None,
) -> tuple[np.ndarray, AlignmentParams]:
    """Return neural depth expressed in optical focus coordinates, plus the fit.

    Pass ``params`` to reuse a mapping instead of fitting one on this scene.
    """
    if params is None:
        mask = confidence >= conf_threshold
        if mask.sum() < min_pixels:
            raise ValueError(
                f"only {int(mask.sum())} pixels have confidence >= {conf_threshold}; "
                "lower the threshold or check the capture"
            )
        idx = _sample(mask, max_samples)
        src, dst, w = neural.ravel()[idx], optical.ravel()[idx], confidence.ravel()[idx]
        scale, shift = fit_affine(src, dst, w)
        residual = dst - (scale * src + shift)
        params = AlignmentParams(
            scale=scale,
            shift=shift,
            n_pixels=int(mask.sum()),
            spearman=depth_direction(optical, neural, mask),
            residual_mad=float(np.median(np.abs(residual - np.median(residual)))),
        )
    aligned = (params.scale * neural + params.shift).astype(np.float32)
    return aligned, params
