"""Baseline AIF compositing from an already registered and cropped focal stack.

A median-filtered focus-label map removes isolated selections. Small feathered
label masks soften seams. The label map is an AIF compositing choice, not metric
depth, and feathering can still produce halos at occlusion boundaries.
"""
from __future__ import annotations

import cv2
import numpy as np


def compose(images: np.ndarray, sharpness: np.ndarray, reference_index: int,
            label_window: int = 5, feather_sigma: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    images = np.asarray(images)
    if images.ndim != 4 or images.shape[-1] != 3 or sharpness.shape != images.shape[:3]:
        raise ValueError('Expected NxHxWx3 images and NxHxW sharpness')
    if not 0 <= reference_index < len(images):
        raise ValueError('Reference index outside stack')
    if len(images) > 256 or label_window not in (1, 3, 5):
        raise ValueError('Use at most 256 frames and label window 1, 3 or 5')
    if not np.isfinite(sharpness).all() or np.any(sharpness < 0) or feather_sigma < 0:
        raise ValueError('Sharpness must be finite/nonnegative and feather sigma nonnegative')
    labels = np.argmax(sharpness, axis=0).astype(np.uint8)
    flat = np.ptp(sharpness, axis=0) <= 1e-12
    labels[flat] = reference_index
    if label_window > 1:
        labels = cv2.medianBlur(labels, label_window)
    # No focus evidence: use the reference, including after spatial label filtering.
    labels[flat] = reference_index
    composite = np.zeros(images.shape[1:], np.float32)
    weight_sum = np.zeros(images.shape[1:3], np.float32)
    for index, image in enumerate(images):
        mask = (labels == index).astype(np.float32)
        if feather_sigma > 0:
            mask = cv2.GaussianBlur(mask, (0, 0), feather_sigma)
        composite += image.astype(np.float32) * mask[..., None]
        weight_sum += mask
    composite /= np.maximum(weight_sum[..., None], 1e-12)
    composite[flat] = images[reference_index][flat]
    return np.clip(np.rint(composite), 0, 255).astype(np.uint8), labels
