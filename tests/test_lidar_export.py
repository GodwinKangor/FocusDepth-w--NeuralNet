import json

import numpy as np
import pytest

from inspect_lidar_capture import inspect_capture, load_depth


def test_depth_layout_and_invalid_samples(tmp_path):
    path = tmp_path / 'depth.f32'
    values = np.array([[0.3, np.nan, 0], [1.0, np.inf, -1]], dtype='<f4')
    path.write_bytes(values.tobytes())
    meta = dict(depth_width=3, depth_height=2, depth_units='meters',
                depth_encoding='float32_little_endian_row_major')
    loaded = load_depth(path, meta)
    np.testing.assert_equal(loaded, values)
    assert ((loaded > 0) & np.isfinite(loaded)).sum() == 2


def test_truncated_depth_is_rejected(tmp_path):
    path = tmp_path / 'depth.f32'
    path.write_bytes(b'\x00' * 7)
    meta = dict(depth_width=2, depth_height=1, depth_units='meters',
                depth_encoding='float32_little_endian_row_major')
    with pytest.raises(ValueError, match='byte count'):
        load_depth(path, meta)


def test_capture_inspection_masks_invalid_depth_and_preserves_focus(tmp_path):
    np.array([0.3, 0.6, np.nan, 0], dtype='<f4').tofile(tmp_path / 'depth_000.f32')
    (tmp_path / 'frame_000.jpg').write_bytes(b'RGB placeholder; inspector checks existence only')
    metadata = dict(
        depth_width=2, depth_height=2, depth_units='meters',
        depth_encoding='float32_little_endian_row_major',
        intrinsics_row_major=[[100, 0, 1], [0, 100, 1], [0, 0, 1]],
        timestamp_delta_s=0.001, dropped_video_while_waiting=1,
        dropped_depth_while_waiting=2, depth_accuracy='absolute',
    )
    (tmp_path / 'pair_000.json').write_text(json.dumps(metadata))
    frame = dict(file='frame_000.jpg', depth_file='depth_000.f32',
                 pair_metadata_file='pair_000.json',
                 lens_position_requested=0.2, lens_position_actual=0.21)
    (tmp_path / 'manifest.json').write_text(json.dumps(
        dict(camera='lidar_rgb', scene_id='test', frames=[frame])))
    pair = inspect_capture(tmp_path)['pairs'][0]
    assert pair['valid_fraction'] == 0.5
    assert pair['median_depth_m'] == pytest.approx(0.45)
    assert pair['lens_actual'] == 0.21
    assert pair['timestamp_delta_s'] == 0.001
