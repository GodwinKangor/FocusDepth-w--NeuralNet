"""Depth from focus: per-frame sharpness and per-pixel focus coordinate.

The output is an *optical focus-coordinate map*: each pixel holds the
(interpolated) AVFoundation lensPosition at which it was sharpest. lensPosition
is a normalized control value, not metric distance. On iPhone, 0.0 is the
nearest focus and 1.0 the farthest, so larger values mean farther; verify this
on your device with objects at known distances.
"""
from __future__ import annotations

import cv2
import numpy as np


def sharpness_stack(stack_gray: np.ndarray, laplacian_ksize: int = 3, smooth_sigma: float = 3.0) -> np.ndarray:
    """Absolute Laplacian response, locally smoothed. stack_gray: (N, H, W) float in [0, 1]."""
    out = np.empty(stack_gray.shape, dtype=np.float32)
    for i, img in enumerate(stack_gray):
        lap = np.abs(cv2.Laplacian(img.astype(np.float32), cv2.CV_32F, ksize=laplacian_ksize))
        out[i] = cv2.GaussianBlur(lap, (0, 0), smooth_sigma) if smooth_sigma > 0 else lap
    return out


def peak_index(sharpness: np.ndarray, subpixel: bool = True) -> np.ndarray:
    """Per-pixel index of maximum sharpness, optionally refined with a parabola fit."""
    n = sharpness.shape[0]
    idx = np.argmax(sharpness, axis=0)
    if not subpixel or n < 3:
        return idx.astype(np.float32)

    inner = np.clip(idx, 1, n - 2)
    take = lambda k: np.take_along_axis(sharpness, k[None], axis=0)[0]
    y_l, y_c, y_r = take(inner - 1), take(inner), take(inner + 1)
    denom = y_l - 2.0 * y_c + y_r
    with np.errstate(divide="ignore", invalid="ignore"):
        offset = np.where(np.abs(denom) > 1e-12, 0.5 * (y_l - y_r) / denom, 0.0)
    offset = np.clip(offset, -0.5, 0.5)
    # Only refine true interior peaks; endpoints stay at their integer index.
    interior = (idx > 0) & (idx < n - 1)
    return np.where(interior, idx + offset, idx).astype(np.float32)


def weighted_index(sharpness: np.ndarray, power: float = 2.0) -> np.ndarray:
    """Sharpness-weighted mean frame index (Assignment 3 style alternative to argmax)."""
    w = np.power(np.maximum(sharpness, 0.0), power)
    w_sum = w.sum(axis=0)
    idx = np.arange(sharpness.shape[0], dtype=np.float32)[:, None, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        out = (w * idx).sum(axis=0) / w_sum
    return np.where(w_sum > 0, out, (sharpness.shape[0] - 1) / 2.0).astype(np.float32)


def index_to_focus(index: np.ndarray, lens_positions: np.ndarray) -> np.ndarray:
    """Map fractional frame indices to lens positions by linear interpolation."""
    lens_positions = np.asarray(lens_positions, dtype=np.float32)
    return np.interp(index, np.arange(len(lens_positions)), lens_positions).astype(np.float32)


def optical_depth(sharpness: np.ndarray, lens_positions, method: str = "peak", power: float = 2.0) -> np.ndarray:
    """Focus-coordinate map. ``lens_positions`` must be sorted ascending to match the stack order."""
    if np.any(np.diff(lens_positions) <= 0):
        raise ValueError("lens_positions must be strictly increasing; sort the stack first")
    if method == "peak":
        idx = peak_index(sharpness)
    elif method == "weighted":
        idx = weighted_index(sharpness, power)
    else:
        raise ValueError(f"unknown optical depth method: {method}")
    return index_to_focus(idx, lens_positions)
