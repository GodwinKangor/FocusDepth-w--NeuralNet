import numpy as np
import pytest

import diagnose_focus_confidence as diagnostic


def test_confidence_groups_keep_rejected_and_unsupported_pixels_visible(monkeypatch):
    monkeypatch.setattr(diagnostic.confidence,'peak_prominence',lambda s:np.array([[.8,.4,.8]]))
    monkeypatch.setattr(diagnostic.confidence,'texture_gate',lambda s:np.array([[.8,1.,.4]]))
    monkeypatch.setattr(diagnostic.focus_depth,'optical_depth',lambda s,l:np.array([[.4,.5,.6]]))
    monkeypatch.setattr(diagnostic,'depth_from_focus',lambda f,m:np.array([[.4,.5,np.nan]]))
    result=diagnostic.summarize(np.ones((3,1,3)),[0,.5,1],(0,0,3,1),{},.4)
    groups=result['groups']
    assert groups['combined']['pixel_fraction']==pytest.approx(1/3)
    assert groups['rejected_combined']['pixel_fraction']==pytest.approx(2/3)
    assert groups['rejected_combined']['metric_pixel_fraction']==pytest.approx(1/3)
    assert groups['rejected_combined']['median_absolute_pixel_residual_cm']==pytest.approx(10)
