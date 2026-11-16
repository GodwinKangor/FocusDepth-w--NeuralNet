import pytest
from compare_aif import transform_regions


def test_native_regions_scale_and_crop_on_shared_reference_grid():
    region = dict(name='box', bbox=[600, 300, 1200, 900], distance_m=.4)
    converted = transform_regions([region], (1920,1440), (1600,1200), (10,20,1590,1190))
    assert converted[0]['bbox'] == [490,230,990,730]
    assert region['bbox'] == [600,300,1200,900]
    with pytest.raises(ValueError, match='outside'):
        transform_regions([region], (1920,1440), (1600,1200), (600,20,1590,1190))
