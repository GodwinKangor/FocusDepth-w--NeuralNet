import numpy as np
import pytest

import depth_alignment
import evaluate


def _scene(seed=0):
    rng = np.random.default_rng(seed)
    optical = rng.uniform(0.1, 0.9, (60, 80)).astype(np.float32)  # larger = farther
    neural = (-2.5 * optical + 4.0).astype(np.float32)             # inverse-depth-like: larger = nearer
    return optical, neural


def test_detects_reversed_depth_convention():
    optical, neural = _scene()
    assert depth_alignment.depth_direction(optical, neural) < -0.99
    assert depth_alignment.depth_direction(optical, -neural) > 0.99


def test_alignment_recovers_affine_map_and_ignores_low_confidence_outliers():
    optical, neural = _scene()
    conf = np.ones_like(optical)
    corrupted = optical.copy()
    corrupted[:10] = 5.0          # garbage optical values ...
    conf[:10] = 0.0               # ... that confidence correctly flags
    aligned, params = depth_alignment.align_neural_to_optical(neural, corrupted, conf, conf_threshold=0.5, min_pixels=10)
    assert params.scale == pytest.approx(-0.4, abs=1e-3)
    assert params.shift == pytest.approx(1.6, abs=1e-3)
    assert np.allclose(aligned, optical, atol=1e-3)


def test_huber_fit_is_robust_to_confident_outliers():
    optical, neural = _scene(1)
    noisy = optical.copy()
    noisy.ravel()[::25] += 0.8    # 4% confidently wrong pixels (e.g. specular highlights)
    _, params = depth_alignment.align_neural_to_optical(neural, noisy, np.ones_like(optical), min_pixels=10)
    assert params.scale == pytest.approx(-0.4, abs=0.02)


def test_alignment_requires_enough_confident_pixels():
    optical, neural = _scene()
    with pytest.raises(ValueError):
        depth_alignment.align_neural_to_optical(neural, optical, np.zeros_like(optical))


def test_pairwise_ordering_accuracy():
    depth = np.zeros((10, 30), np.float32)
    depth[:, 10:20] = 1.0
    depth[:, 20:] = 2.0
    regions = [
        {"name": "near", "bbox": [0, 0, 10, 10], "distance_m": 0.3},
        {"name": "mid", "bbox": [10, 0, 20, 10], "distance_m": 0.6},
        {"name": "far", "bbox": [20, 0, 30, 10], "distance_m": 1.2},
    ]
    assert evaluate.pairwise_ordering(depth, regions)["accuracy"] == 1.0
    assert evaluate.pairwise_ordering(-depth, regions)["accuracy"] == 0.0
    # Crop offset shifts the bbox into cropped-image coordinates.
    assert evaluate.pairwise_ordering(depth[:, 5:], [dict(r, bbox=[r["bbox"][0] + 5, 0, r["bbox"][2], 10]) for r in regions], (5, 0))["accuracy"] == 1.0
