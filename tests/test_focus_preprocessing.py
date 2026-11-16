import numpy as np
import pytest

from compare_focus_preprocessing import roi_box, score_validation


def test_roi_coordinates_scale_then_account_for_registration_crop():
    assert roi_box([.45,.45,.55,.55],(1920,1440),(1600,1200),(20,15,1580,1185)) == [700,525,860,645]
    with pytest.raises(ValueError,match='outside'):
        roi_box([0,0,.1,.1],(1920,1440),(1600,1200),(20,15,1580,1185))


def test_pixel_scoring_rejects_low_metric_coverage(tmp_path,monkeypatch):
    np.save(tmp_path/'roi_pixel_focus.npy',np.ones((2,2)))
    monkeypatch.setattr('compare_focus_preprocessing.depth_from_focus',lambda pixels,model:np.array([[.4,np.nan],[np.nan,np.nan]]))
    result=score_validation({'distance_m':.4},{},'pixel_median',tmp_path)
    assert result['status']=='insufficient_metric_coverage'
    assert result['coverage']==.25
    assert result['predicted_m'] is None


def test_pixel_scoring_maps_pixels_before_regional_median(tmp_path,monkeypatch):
    np.save(tmp_path/'roi_pixel_focus.npy',np.ones((2,2)))
    monkeypatch.setattr('compare_focus_preprocessing.depth_from_focus',lambda pixels,model:np.array([[.4,.6],[np.nan,np.nan]]))
    result=score_validation({'distance_m':.4},{},'pixel_median',tmp_path)
    assert result['status']=='valid'
    assert result['predicted_m']==pytest.approx(.5)
    assert result['signed_residual_cm']==pytest.approx(10)
