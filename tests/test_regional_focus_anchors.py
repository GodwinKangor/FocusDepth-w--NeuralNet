import numpy as np
import pytest
from regional_focus_anchors import extract_anchors,scale_with_anchors


def test_regional_fit_recovers_inverse_affine_and_rejects_degenerate_support():
    source=np.linspace(1,3,12)
    anchors=[dict(neural_inverse=float(s),focus_m=float(1/(2*s+.5)),prominence=.8) for s in source]
    result,fit=scale_with_anchors(source,anchors)
    np.testing.assert_allclose(result,1/(2*source+.5),rtol=1e-5)
    assert fit['scale']==pytest.approx(2)
    with pytest.raises(ValueError,match='Too few'):scale_with_anchors(source,anchors[:2])
    with pytest.raises(ValueError,match='span'):scale_with_anchors(source,[anchors[0]]*8)
    reversed_anchors=[dict(neural_inverse=float(s),focus_m=float(1/(7-s)),prominence=.8) for s in source]
    with pytest.raises(ValueError,match='Invalid'):scale_with_anchors(source,reversed_anchors)


def test_flat_focus_grid_does_not_create_anchors():
    anchors,rejections=extract_anchors(np.ones((5,64,128),np.float32),np.linspace(0,1,5),np.ones((64,128)),{})
    assert anchors==[]
    assert sum(rejections.values())==2
