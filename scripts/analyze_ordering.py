#!/usr/bin/env python3
"""Evaluate saved depth maps without rerunning inference or changing outputs."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'python'))
import numpy as np
import evaluate


def main():
    results = {}
    files = {'B1 optical': 'optical_depth', 'B2 neural': 'neural_depth',
             'B3 fixed fusion': 'fixed_fusion', 'Confidence fusion': 'confidence_fusion'}
    for annotation in sorted((ROOT / 'data/annotations').glob('*.json')):
        folder = ROOT / 'results' / annotation.stem
        if not (folder / 'metrics.json').exists():
            continue
        metrics = json.loads((folder / 'metrics.json').read_text())
        ann = json.loads(annotation.read_text())
        maps = {name: np.load(folder / (file + '.npy')) for name, file in files.items()}
        confidence = np.load(folder / 'optical_confidence.npy')
        offset = tuple(metrics['registration']['crop_xyxy'][:2])
        threshold = metrics['config']['alignment']['conf_threshold']
        evaluation = evaluate.evaluate_scene(maps, confidence, ann, offset,
                         metrics['config']['evaluation']['min_separation_m'])
        for region in ann['regions']:
            values = evaluate.region_values(confidence, region['bbox'], offset)
            evaluation['regions'][region['name']]['coverage_at_threshold'] = float((values >= threshold).mean())
        results[annotation.stem] = {'distance_source': ann.get('distance_source', 'unspecified'),
                                   'confidence_threshold': threshold, 'evaluation': evaluation}
    destination = ROOT / 'docs/ordering_analysis.json'
    destination.write_text(json.dumps(results, indent=2) + '\n')
    lines = ['# Preliminary ordering analysis', '',
             'Recomputed from saved maps with `scripts/analyze_ordering.py`. No model inference or parameter tuning was performed. Labels are visual estimates, not measured ground truth.', '',
             '| Capture | Method | Correct / eligible pairs |', '|---|---|---|']
    for scene, data in results.items():
        for method, score in data['evaluation']['ordering'].items():
            correct = sum(pair['correct'] for pair in score['pairs'])
            lines.append(f'| `{scene}` | {method} | {correct}/{score["n_pairs"]} |')
    lines += ['', '## Near-focus failure case', '',
              'Capture `scene_01_20261102_182153` contains a can annotated as never sharp. Optical estimation reverses its order relative to the thermometer label and pink box. Neural estimation and both fusion methods recover all six provisional region orderings.', '',
              'The can has mean optical confidence 0.042, so confidence fusion gives approximately 95.8% mean weight to the neural estimate within that box. This supports the intended suppression of weak optical evidence. It does not show superiority over neural estimation or fixed fusion, which also score 6/6.', '',
              'The aligned neural median for the can is approximately −0.984, outside the nominal lens-control range. The affine alignment extrapolates; this is a relative coordinate and cannot be interpreted as physical distance. Region medians are computed after pixelwise fusion, so they are not simply weighted sums of input medians.', '',
              '## Next capture', '',
              'Collect a scene containing both (1) an object too near to focus and (2) textured objects within the focus range for which neural ordering may fail. Measure object distances independently, keep lighting and camera position fixed, and repeat three sweeps. Keep the present thresholds fixed. Record failures as well as successes; do not select only examples where fusion wins. A benefit over neural and fixed fusion remains unproven.', '']
    (ROOT / 'docs/ordering_analysis.md').write_text('\n'.join(lines))
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
