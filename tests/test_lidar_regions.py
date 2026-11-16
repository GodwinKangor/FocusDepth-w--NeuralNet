import numpy as np
import pytest

from evaluate_lidar_regions import depth_bbox, summarize_patch


def test_native_grid_mapping_keeps_complete_interior_cells():
    assert depth_bbox([61, 121, 121, 181], (1920, 1440), (320, 240)) == (11, 21, 20, 30)
    with pytest.raises(ValueError, match='aspect'):
        depth_bbox([60, 120, 120, 180], (1920, 1440), (320, 180))
    with pytest.raises(ValueError, match='outside'):
        depth_bbox([-1, 120, 120, 180], (1920, 1440), (320, 240))


def test_patch_median_excludes_invalid_values_and_other_surfaces():
    depth = np.array([[.4, np.nan, 10], [.6, 0, 10], [10, 10, 10]])
    stats = summarize_patch(depth, (0, 0, 2, 2))
    assert stats['median_m'] == pytest.approx(.5)
    assert stats['valid_fraction'] == .5
    with pytest.raises(ValueError, match='no valid'):
        summarize_patch(depth, (1, 0, 2, 2))
