import numpy as np
import pytest

from anchor_metric_depth import anchor_samples, calibrate


def test_scale_uses_sensor_anchors_and_recovers_known_scale():
    source=np.array([1.,2.,3.])
    target=source*.4
    output,params=calibrate(np.array([[1.,2.,3.]],np.float32),source,target,'scale')
    np.testing.assert_allclose(output,[[.4,.8,1.2]])
    assert params['scale']==pytest.approx(.4)


def test_inverse_fit_preserves_invalid_extrapolation_instead_of_clipping():
    source=np.array([1.,2.,3.])
    target=1/(2/source-.5)
    output,params=calibrate(np.array([[1.,2.,3.,5.]],np.float32),source,target,'inverse_affine')
    np.testing.assert_allclose(output[0,:3],target,rtol=1e-5)
    assert np.isnan(output[0,3])
    assert params['invalid_fraction']==.25
    with pytest.raises(ValueError,match='variation'):
        calibrate(np.ones((2,2)),np.ones(100),np.ones(100),'inverse_affine')


def test_anchor_selection_excludes_evaluation_region_with_margin():
    prediction=np.ones((90,120),np.float32)
    depth=np.full((15,20),2.,np.float32)
    regions=[dict(bbox_native=[30,30,36,36])]
    source,target,mask=anchor_samples(prediction,depth,(120,90),(120,90),(0,0,120,90),regions,[0,0,120,90])
    yy,xx=np.indices(depth.shape)
    nx=(xx+.5)*6-.5;ny=(yy+.5)*6-.5
    exclusion=(nx>=6)&(nx<60)&(ny>=6)&(ny<60)
    assert not mask[exclusion].any()
    assert len(source)==len(target)==mask.sum()
    assert (source==1).all() and (target==2).all()
