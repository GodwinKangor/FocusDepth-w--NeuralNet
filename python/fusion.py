"""Fusion of aligned optical and neural depth maps.

Both inputs must already be registered, direction-checked and expressed in the
same relative-depth representation (see depth_alignment.py).
"""
from __future__ import annotations

import numpy as np


def fixed_fusion(d_opt: np.ndarray, d_nn: np.ndarray, weight: float = 0.5) -> np.ndarray:
    """B3: the same optical weight everywhere."""
    return (weight * d_opt + (1.0 - weight) * d_nn).astype(np.float32)


def confidence_fusion(d_opt: np.ndarray, d_nn: np.ndarray, optical_confidence: np.ndarray) -> np.ndarray:
    """Proposed: optical weight equals per-pixel optical confidence."""
    w = np.clip(optical_confidence, 0.0, 1.0)
    return (w * d_opt + (1.0 - w) * d_nn).astype(np.float32)
