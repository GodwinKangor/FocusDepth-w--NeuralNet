import numpy as np
import pytest
from metric_focus_fusion import scale_relative_with_focus, inverse_depth_fusion
from evaluate_metric_scenes import region_statistics


def test_focus_scaled_relative_depth_recovers_inverse_metric_mapping():
    raw=np.arange(1,13,dtype=np.float32).reshape(3,4)
    optical=1/(2*raw+1);confidence=np.ones(raw.shape,np.float32)
    result,fit=scale_relative_with_focus(raw,optical,confidence,min_pixels=4)
    np.testing.assert_allclose(result,optical,rtol=1e-5)
    assert fit['scale']==pytest.approx(2,rel=1e-5)
    assert fit['shift']==pytest.approx(1,rel=1e-5)
    with pytest.raises(ValueError,match='few'):
        scale_relative_with_focus(raw,np.full_like(raw,np.nan),confidence,min_pixels=4)


def test_fusion_handles_missing_sources_and_uses_inverse_depth():
    opt=np.array([[.5,np.nan],[.5,np.nan]],np.float32)
    nn=np.array([[1,1],[np.nan,np.nan]],np.float32)
    result=inverse_depth_fusion(opt,nn,np.full((2,2),.5,np.float32))
    assert result[0,0]==pytest.approx(2/3)
    assert result[0,1]==1 and result[1,0]==.5 and np.isnan(result[1,1])


def test_region_scoring_retains_coverage_and_excludes_unsupported_target():
    region=dict(name='target',bbox=[0,0,2,2],distance_m=.2,within_measured_calibration_span=True)
    values=np.array([[.2,np.nan],[np.nan,np.nan]],np.float32)
    result=region_statistics(values,region)
    assert result['coverage']==.25
    assert not result['score_eligible']
    assert result['median_cm']==pytest.approx(20)
    region['within_measured_calibration_span']=False
    result=region_statistics(np.full((2,2),.2,np.float32),region)
    assert result['status']=='outside_measured_calibration_span'
