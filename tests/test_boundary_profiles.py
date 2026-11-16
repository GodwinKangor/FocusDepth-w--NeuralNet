import numpy as np
import pytest
from compare_boundary_detail import score_profile, probe_map


def test_sigmoid_profile_recovers_offset_and_width():
    x=np.arange(-45,46,dtype=float);y=1/(1+np.exp(-(x-4)/4))
    result=score_profile(x,y,1)
    assert result['status']=='valid'
    assert result['signed_offset_native_px']==pytest.approx(4,abs=.05)
    assert result['transition_width_native_px']==pytest.approx(8*np.log(9),abs=.15)


def test_missing_crossing_and_wrong_depth_direction_are_not_filled():
    x=np.arange(-45,46,dtype=float);y=(x>0).astype(float);y[abs(x)<3]=np.nan
    assert score_profile(x,y,1)['status']=='missing_half_crossing'
    assert score_profile(x,1-(x>0).astype(float),1)['status']=='wrong_direction_or_weak_step'


def test_strongest_crossing_is_not_chosen_for_proximity_to_annotation():
    x=np.arange(-45,46,dtype=float);y=np.zeros_like(x)
    y[(x>=-2)&(x<5)]=.6;y[(x>=5)&(x<20)]=.1;y[x>=20]=1
    result=score_profile(x,y,1)
    assert result['status']=='valid'
    assert result['n_rising_half_crossings']==2
    assert result['signed_offset_native_px']>10


def test_sparse_map_reports_missing_support():
    meta=dict(native_size=[100,100],resized_size=[100,100],crop=[0,0,100,100])
    edge=dict(start=[50,20],end=[50,80],background_normal=[-1,0])
    result=probe_map(np.full((100,100),np.nan,np.float32),edge,meta)
    assert result['status']=='missing_anchor_support'
    assert result['sample_coverage']==0


def test_metric_profiles_use_inverse_depth_and_preserve_invalid_samples():
    from compare_boundary_detail import far_inverse_metric
    depth=np.array([[.5,1,0,-1,np.nan]],np.float32)
    result=far_inverse_metric(depth)
    np.testing.assert_array_equal(result[0,:2],[-2,-1])
    assert np.isnan(result[0,2:]).all()
