import cv2
import numpy as np
import pytest

import confidence
import focus_depth
import registration
import synthetic


def test_synthetic_peak_index_matches_known_focus():
    stack, true_idx, lens = synthetic.layered_scene(n_frames=9)
    sharp = focus_depth.sharpness_stack(stack, smooth_sigma=3.0)
    idx = focus_depth.peak_index(sharp)
    # Judge textured pixels away from layer boundaries, where the signal is unambiguous.
    textured = np.zeros_like(true_idx, bool)
    textured[:, : true_idx.shape[1] // 2] = True
    textured |= true_idx < len(lens) - 1
    interior = cv2.erode((textured).astype(np.uint8), np.ones((15, 15))).astype(bool)
    edges = cv2.Canny((true_idx * 20).astype(np.uint8), 1, 1) > 0
    interior &= ~cv2.dilate(edges.astype(np.uint8), np.ones((15, 15))).astype(bool)
    err = np.abs(idx - true_idx)[interior]
    assert np.median(err) < 0.5
    assert (err < 1.0).mean() > 0.9


def test_subpixel_refinement_recovers_peak_between_frames():
    # Gaussian profile peaking at 3.3 for every pixel.
    k = np.arange(8, dtype=np.float32)[:, None, None]
    sharp = np.exp(-((k - 3.3) ** 2) / 2.0) * np.ones((8, 4, 4), np.float32)
    idx = focus_depth.peak_index(sharp)
    assert np.allclose(idx, 3.3, atol=0.1)
    depth = focus_depth.optical_depth(sharp, np.linspace(0.0, 0.7, 8))
    assert np.allclose(depth, 0.33, atol=0.01)


def test_optical_depth_rejects_unsorted_lens_positions():
    with pytest.raises(ValueError):
        focus_depth.optical_depth(np.ones((3, 2, 2), np.float32), [0.1, 0.3, 0.2])


@pytest.mark.parametrize("measure", ["prominence", "entropy"])
def test_flat_sharpness_profile_has_low_confidence(measure):
    flat = np.ones((9, 10, 10), np.float32)
    k = np.arange(9, dtype=np.float32)[:, None, None]
    peaked = (0.05 + np.exp(-((k - 4) ** 2) / 0.5)) * np.ones((9, 10, 10), np.float32)
    c_flat = confidence.optical_confidence(flat, measure, use_texture_gate=False)
    c_peak = confidence.optical_confidence(peaked, measure, use_texture_gate=False)
    assert c_flat.max() < 0.05
    assert c_peak.min() > 0.3


def test_texture_gate_suppresses_weak_signal():
    k = np.arange(9, dtype=np.float32)[:, None, None]
    profile = np.exp(-((k - 4) ** 2) / 0.5) * np.ones((9, 10, 20), np.float32)
    profile[:, :, 10:] *= 1e-3  # same shape, almost no signal
    c = confidence.optical_confidence(profile, "prominence", use_texture_gate=True)
    assert c[:, :10].min() > 0.5
    assert c[:, 10:].max() < 0.05


def test_registration_recovers_translation():
    # Two crops of a larger scene, like a camera that moved by (4, -3) pixels.
    scene = (synthetic.textured_image(240, 300, seed=3) * 255).astype(np.uint8)
    scene = cv2.cvtColor(scene, cv2.COLOR_GRAY2BGR)
    base = scene[20:220, 20:280]
    moved = scene[23:223, 16:276]
    res = registration.register_stack([base, moved], ref_index=0, motion="translation")
    warp = res.warps[1]
    assert abs(warp[0, 2] - 4.0) < 0.1 and abs(warp[1, 2] + 3.0) < 0.1
    x0, y0, x1, y1 = res.crop
    assert (x1 - x0, y1 - y0) == (256, 197)
    diff = np.abs(res.images[0].astype(float) - res.images[1].astype(float))
    assert diff.mean() < 5.0
