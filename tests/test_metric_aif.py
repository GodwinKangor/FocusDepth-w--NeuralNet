import numpy as np
import pytest
from compare_metric_aif import prepare_input, restore_prediction, region_result, aggregate


def test_orientation_round_trip_keeps_spatial_locations():
    image=np.arange(24,dtype=np.float32).reshape(4,6)
    upright=prepare_input(image,'upright')
    assert upright.shape==(6,4)
    np.testing.assert_array_equal(restore_prediction(upright,'upright'),image)
    np.testing.assert_array_equal(prepare_input(image,'native'),image)


def test_metric_residual_is_unadjusted_and_uncertainty_is_not_fit():
    prediction=np.array([[.4,.4,np.nan],[.4,0,9]],dtype=np.float32)
    region=dict(name='package',bbox=[0,0,2,2],distance_m=.3,approximate_uncertainty_m=.01)
    result=region_result(prediction,region)
    assert result['predicted_cm']==pytest.approx(40)
    assert result['signed_residual_cm']==pytest.approx(10)
    assert result['absolute_residual_lower_bound_cm']==pytest.approx(9)
    assert result['absolute_residual_upper_bound_cm']==pytest.approx(11)
    assert result['valid_fraction']==.75


def test_aggregate_excludes_wall_and_weights_package_regions_equally():
    rows=[]
    for orientation in ['native','upright']:
        for name in ['reference','aif']:
            for obj,err in [('first',3),('second',-4),('wall',100)]:
                rows.append(dict(orientation=orientation,input=name,object=obj,signed_residual_cm=err,
                                 absolute_residual_cm=abs(err),nominal_cm=50))
    summary=aggregate(rows)
    for s in summary:
        assert s['mae_cm']==3.5
        assert s['rmse_cm']==pytest.approx(np.sqrt(12.5))
        assert s['n_package_regions']==2


def test_relative_pipeline_rejects_metric_checkpoint_before_loading_weights(monkeypatch):
    from types import SimpleNamespace
    from transformers import AutoConfig
    from neural_depth import DepthPredictor

    monkeypatch.setattr(AutoConfig, 'from_pretrained', lambda *args, **kwargs: SimpleNamespace(depth_estimation_type='metric'))
    with pytest.raises(ValueError, match='Expected relative depth'):
        DepthPredictor('test-checkpoint', device='cpu')
