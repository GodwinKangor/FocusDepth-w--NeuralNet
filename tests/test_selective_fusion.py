import numpy as np
import pytest
from selective_focus_fusion import selective_fusion


def test_disagreement_rejects_optical_without_mutating_inputs():
    optical=np.full((15,15),.4,np.float32);neural=np.ones((15,15),np.float32);conf=np.ones((15,15),np.float32)
    original=optical.copy()
    for mode in ['agreement','agreement_edge','agreement_edge_support']:
        result,w,stats=selective_fusion(optical,neural,conf,mode)
        np.testing.assert_array_equal(result,neural)
        assert (w==0).all() and stats['accepted_optical_fraction']==0
    np.testing.assert_array_equal(optical,original)


def test_neural_step_is_preserved_while_supported_interiors_can_change():
    neural=np.full((25,30),.5,np.float32);neural[:,15:]=1
    optical=neural*.95;conf=np.ones_like(neural)
    result,w,_=selective_fusion(optical,neural,conf,'agreement_edge_support')
    assert (w[:,13:17]==0).all()
    np.testing.assert_array_equal(result[:,13:17],neural[:,13:17])
    assert w[12,5]>0 and result[12,5]!=neural[12,5]


def test_isolated_optical_point_is_rejected_by_local_support():
    optical=np.full((15,15),np.nan,np.float32);optical[7,7]=.95
    neural=np.ones_like(optical);conf=np.ones_like(optical)
    result,w,_=selective_fusion(optical,neural,conf,'agreement_edge_support')
    assert (w==0).all()
    np.testing.assert_array_equal(result,neural)
    with pytest.raises(ValueError,match='Unknown'):
        selective_fusion(optical,neural,conf,'unknown')
