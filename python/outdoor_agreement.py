"""Evaluate outdoor depth agreement with LiDAR without plotting or inference.

Reports method-specific coverage and identical-valid-pixel comparisons within
20-100 cm. LiDAR is a reference, not independent ground truth. Source manifests
and the frozen calibration must match the preprocessing provenance.
"""
import json
import hashlib
from pathlib import Path

import numpy as np

ROOT = Path('results/outdoor_dusk')
SPAN = (0.2, 1.0)
SCENES = [('outdoor_dusk_20261116_155223', 'Plant on mulch'),
          ('outdoor_dusk_20261116_155442', 'Post in grass'),
          ('outdoor_dusk_20261116_155426', 'Post at pavement edge')]
METHODS = ('calibrated_focus', 'focus_scaled_relative_aif',
           'focus_neural_fusion', 'unadjusted_metric_aif')


def valid(a):
    return np.isfinite(a) & (a > 0)


def agreement(values, lidar, mask):
    if values.shape != lidar.shape or mask.shape != lidar.shape:
        raise ValueError('Prediction, LiDAR and mask grids differ')
    both = valid(values) & valid(lidar) & mask
    count = int(both.sum());total = int(mask.sum())
    return dict(n_evaluation_pixels=total, n_valid_pixels=count,
                in_span_coverage=float(count / total) if total else None,
                median_abs_diff_cm=float(np.median(np.abs(values[both] - lidar[both])) * 100) if count else None,
                median_signed_diff_cm=float(np.median(values[both] - lidar[both]) * 100) if count else None)


def summarize():
    rows = []
    for scene, label in SCENES:
        d = ROOT / scene
        lidar = np.load(d / 'direct_lidar_meters.npy')
        span = valid(lidar) & (lidar >= SPAN[0]) & (lidar <= SPAN[1])
        meta = json.loads((d / 'preprocessing.json').read_text())
        manifest_path = Path('data/raw') / scene / 'manifest.json'
        calibration_path = Path('results/focus_calibration/model.json')
        if meta['registration_failed']:raise ValueError('Registration failed')
        if hashlib.sha256(manifest_path.read_bytes()).hexdigest() != meta['source_manifest_sha256']:
            raise ValueError('Source manifest changed')
        if hashlib.sha256(calibration_path.read_bytes()).hexdigest() != meta['calibration_model_sha256']:
            raise ValueError('Calibration changed')
        maps = {key: np.load(d / f'{key}_meters.npy') for key in METHODS}
        common = span & np.logical_and.reduce([valid(values) for values in maps.values()])
        optical = maps['calibrated_focus']
        row = dict(scene=scene, label=label, in_span_fraction=float(span.mean()), methods={}, common_support_methods={},
                   common_support_image_fraction=float(common.mean()),
                   common_support_in_span_fraction=float(common.sum() / span.sum()) if span.any() else None,
                   optical_support_image_fraction=float(valid(optical).mean()),
                   source_manifest_sha256=meta['source_manifest_sha256'],
                   calibration_model_sha256=meta['calibration_model_sha256'])
        for key in METHODS:
            row['methods'][key] = agreement(maps[key], lidar, span)
            row['common_support_methods'][key] = agreement(maps[key], lidar, common)
        rows.append(row)
    return rows


def main():
    rows = summarize()
    output = ROOT / 'summary_vs_lidar.json'
    output.write_text(json.dumps(dict(
        scope='Agreement with direct LiDAR on pixels LiDAR places at 0.2-1.0 m; LiDAR is not ground truth.',
        scenes=rows), indent=2, allow_nan=False) + '\n')
    print('Wrote', output)


if __name__ == '__main__':
    main()
