import numpy as np
import pytest
from all_in_focus import compose


def test_composite_selects_different_planes_and_flat_falls_back():
    images = np.zeros((3, 12, 18, 3), np.uint8)
    images[0] = [20, 30, 40]
    images[1] = [90, 100, 110]
    images[2] = [180, 190, 200]
    sharp = np.zeros((3, 12, 18), np.float32)
    sharp[0, :, :6] = 10
    sharp[2, :, 12:] = 10
    result, labels = compose(images, sharp, 1, label_window=1, feather_sigma=0)
    np.testing.assert_array_equal(result[:, :6], images[0, :, :6])
    np.testing.assert_array_equal(result[:, 6:12], images[1, :, 6:12])
    np.testing.assert_array_equal(result[:, 12:], images[2, :, 12:])
    assert (labels[:, 6:12] == 1).all()


def test_feathering_preserves_constant_radiance_and_bounds():
    images = np.full((3, 12, 18, 3), 173, np.uint8)
    sharp = np.random.default_rng(0).random((3, 12, 18)).astype(np.float32)
    composite, _ = compose(images, sharp, 1)
    assert (composite == 173).all()
    with pytest.raises(ValueError):
        compose(images, sharp[:, :-1], 1)
