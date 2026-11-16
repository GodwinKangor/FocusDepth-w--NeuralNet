"""Validate and summarize an exported LiDAR sweep without rescaling its geometry.

Usage: python python/inspect_lidar_capture.py path/to/capture
Depth remains on its native sensor grid; this does not establish RGB registration.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load_depth(path: Path, metadata: dict) -> np.ndarray:
    if metadata['depth_encoding'] != 'float32_little_endian_row_major':
        raise ValueError('Unsupported depth encoding')
    if metadata['depth_units'] != 'meters':
        raise ValueError('Expected depth in meters')
    width, height = metadata['depth_width'], metadata['depth_height']
    if width <= 0 or height <= 0:
        raise ValueError('Depth dimensions must be positive')
    if path.stat().st_size != width * height * 4:
        raise ValueError(f'{path.name}: depth byte count does not match dimensions')
    return np.fromfile(path, dtype='<f4').reshape(height, width)


def inspect_capture(folder: Path) -> dict:
    manifest = json.loads((folder / 'manifest.json').read_text())
    if manifest['camera'] != 'lidar_rgb':
        raise ValueError('Expected a LiDAR RGB capture')
    if not manifest['frames']:
        raise ValueError('Capture contains no frames')
    pairs = []
    for frame in manifest['frames']:
        if not (folder / frame['file']).is_file():
            raise ValueError(f"Missing RGB frame: {frame['file']}")
        meta = json.loads((folder / frame['pair_metadata_file']).read_text())
        depth = load_depth(folder / frame['depth_file'], meta)
        valid = np.isfinite(depth) & (depth > 0)
        values = depth[valid]
        intrinsics = np.asarray(meta['intrinsics_row_major'])
        if intrinsics.shape != (3, 3) or not np.isfinite(intrinsics).all():
            raise ValueError('Invalid calibration matrix')
        pairs.append({
            'rgb': frame['file'],
            'depth_shape': list(depth.shape),
            'valid_fraction': float(valid.mean()),
            'median_depth_m': float(np.median(values)) if values.size else None,
            'timestamp_delta_s': meta['timestamp_delta_s'],
            'lens_requested': frame['lens_position_requested'],
            'lens_actual': frame['lens_position_actual'],
            'dropped_video_while_waiting': meta['dropped_video_while_waiting'],
            'dropped_depth_while_waiting': meta['dropped_depth_while_waiting'],
            'depth_accuracy': meta['depth_accuracy'],
        })
    return {'scene_id': manifest['scene_id'], 'pairs': pairs,
            'scope': 'File integrity and native depth statistics; not metric accuracy or registration validation.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('capture', type=Path)
    args = parser.parse_args()
    print(json.dumps(inspect_capture(args.capture), indent=2, allow_nan=False))
