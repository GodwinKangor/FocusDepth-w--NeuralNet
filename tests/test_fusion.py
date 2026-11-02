import numpy as np

import fusion


def test_confidence_fusion_endpoints():
    rng = np.random.default_rng(0)
    d_opt, d_nn = rng.random((2, 8, 8)).astype(np.float32)
    assert np.allclose(fusion.confidence_fusion(d_opt, d_nn, np.zeros((8, 8))), d_nn)
    assert np.allclose(fusion.confidence_fusion(d_opt, d_nn, np.ones((8, 8))), d_opt)


def test_confidence_is_clipped():
    d_opt, d_nn = np.ones((2, 2)), np.zeros((2, 2))
    out = fusion.confidence_fusion(d_opt, d_nn, np.array([[-1.0, 2.0], [0.25, 0.75]]))
    assert np.allclose(out, [[0.0, 1.0], [0.25, 0.75]])


def test_fixed_fusion_is_weighted_average():
    assert np.allclose(fusion.fixed_fusion(np.full((2, 2), 4.0), np.zeros((2, 2)), 0.25), 1.0)
