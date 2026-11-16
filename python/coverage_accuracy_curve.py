"""Accuracy-versus-coverage curve for calibrated focus on the multi-object scenes.

Sweeps the optical-confidence threshold on the existing calibrated focus maps
(optical_meters_unmasked.npy, i.e. pixels inside the calibrated lens range) and
reports, per threshold:

- full-image coverage (fraction of pixels retained), averaged across scenes;
- region MAE against the approximate measured labels, using the same scoring
  rule as evaluate_metric_scenes.py (in-span regions with >= 50% box coverage);
- the number of regions that remain scorable;
- a secondary dense check against direct LiDAR (an imperfect reference),
  restricted to pixels LiDAR places inside the calibrated span (0.2-1.0 m):
  coverage of those pixels and median absolute difference on retained ones;
- retained optical pixels that LiDAR places beyond 1.5 m (false in-range support).

Most of each image is store background beyond 1.5 m, outside the calibration
span, so unrestricted whole-image LiDAR differences are dominated by background
and are reported separately rather than mixed into the dense check.

This describes the trade-off; it does not select a threshold. The scenes are
development data. Other methods are plotted as fixed reference points read from
results/multiscene_metric/summary.json. No predictions are recomputed.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from compare_aif import transform_regions
from evaluate_metric_scenes import region_statistics

SPAN_M = (0.2, 1.0)
FAR_M = 1.5
DEFAULT_THRESHOLDS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9]
REFERENCE_METHODS = {
    'direct_lidar': 'Direct LiDAR',
    'focus_neural_fusion': 'Focus/neural fusion',
    'focus_scaled_relative_aif': 'Focus-scaled neural (AIF)',
}


def valid(values):
    return np.isfinite(values) & (values > 0)


def in_span(lidar):
    return valid(lidar) & (lidar >= SPAN_M[0]) & (lidar <= SPAN_M[1])


def curve(root, thresholds):
    summary = json.loads((root / 'summary.json').read_text())
    scenes = []
    for scene in summary['annotations']['scenes']:
        out = root / scene['scene']
        meta = json.loads((out / 'preprocessing.json').read_text())
        scenes.append(dict(
            name=scene['scene'],
            regions=transform_regions(scene['regions'], meta['native_size'], meta['resized_size'], meta['crop']),
            optical=np.load(out / 'optical_meters_unmasked.npy'),
            confidence=np.load(out / 'optical_confidence.npy'),
            lidar=np.load(out / 'direct_lidar_meters.npy'),
            maps={m: np.load(out / f'{m}_meters.npy') for m in REFERENCE_METHODS}))
    n_in_span = sum(r['within_measured_calibration_span'] for s in scenes for r in s['regions'])

    rows = []
    for t in thresholds:
        coverages, rows_t, lidar_diffs, span_kept, span_total, far_kept = [], [], [], 0, 0, 0
        for s in scenes:
            keep = valid(s['optical']) & np.isfinite(s['confidence']) & (s['confidence'] >= t)
            coverages.append(float(keep.mean()))
            masked = np.where(keep, s['optical'], np.nan)
            for region in s['regions']:
                stats = region_statistics(masked, region)
                if stats['score_eligible']:
                    rows_t.append(dict(scene=s['name'], **stats))
            span = in_span(s['lidar'])
            span_total += int(span.sum())
            span_kept += int((keep & span).sum())
            far_kept += int((keep & valid(s['lidar']) & (s['lidar'] > FAR_M)).sum())
            both = keep & span
            lidar_diffs.append(np.abs(s['optical'][both] - s['lidar'][both]) * 100)
        diffs = np.concatenate(lidar_diffs)
        rows.append(dict(
            threshold=t,
            mean_full_image_coverage=float(np.mean(coverages)),
            per_scene_coverage=coverages,
            n_scored_regions=len(rows_t),
            n_in_span_regions=n_in_span,
            region_mae_cm=float(np.mean([abs(r['signed_residual_cm']) for r in rows_t])) if rows_t else None,
            scored_regions=[f"{r['scene']}/{r['name']}" for r in rows_t],
            in_span_coverage=span_kept / span_total,
            median_abs_diff_from_lidar_in_span_cm=float(np.median(diffs)) if diffs.size else None,
            n_pixels_vs_lidar=int(diffs.size),
            retained_pixels_lidar_beyond_far=far_kept,
            fraction_retained_lidar_beyond_far=far_kept / max(1, sum(int(c * s['optical'].size) for c, s in zip(coverages, scenes)))))

    references = []
    by_method = {m['method']: m for m in summary['summary']}
    for method, label in REFERENCE_METHODS.items():
        cov = [float(valid(s['maps'][method]).mean()) for s in scenes]
        dense, whole, span_kept, span_total = [], [], 0, 0
        for s in scenes:
            m = valid(s['maps'][method])
            span = in_span(s['lidar'])
            span_total += int(span.sum())
            span_kept += int((m & span).sum())
            both = m & span
            dense.append(np.abs(s['maps'][method][both] - s['lidar'][both]) * 100)
            allv = m & valid(s['lidar'])
            whole.append(np.abs(s['maps'][method][allv] - s['lidar'][allv]) * 100)
        dense, whole = np.concatenate(dense), np.concatenate(whole)
        references.append(dict(
            method=method, label=label,
            mean_full_image_coverage=float(np.mean(cov)),
            region_mae_cm=by_method[method]['all_supported_region_mae_cm'],
            n_scored_regions=by_method[method]['n_scored_regions'],
            in_span_coverage=span_kept / span_total,
            median_abs_diff_from_lidar_in_span_cm=None if method == 'direct_lidar' else float(np.median(dense)),
            whole_image_median_abs_diff_from_lidar_cm=None if method == 'direct_lidar' else float(np.median(whole))))
    return rows, references


def plot(rows, references, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    style = {'direct_lidar': ('s', '#555555'), 'focus_neural_fusion': ('D', '#d95f02'),
             'focus_scaled_relative_aif': ('^', '#1b9e77')}
    focus_color = '#3b6fb6'
    shown = {0.0, 0.5, 0.7, 0.85, 0.9}
    n_all = rows[0]['n_in_span_regions']
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout='constrained')

    ax = axes[0]
    pts = [r for r in rows if r['region_mae_cm'] is not None]
    xs = [r['mean_full_image_coverage'] * 100 for r in pts]
    ys = [r['region_mae_cm'] for r in pts]
    ax.plot(xs, ys, '-', color=focus_color, alpha=.6)
    full = [r['n_scored_regions'] == n_all for r in pts]
    ax.plot([x for x, f in zip(xs, full) if f], [y for y, f in zip(ys, full) if f], 'o', color=focus_color,
            label=f'Calibrated focus, all {n_all} regions scored')
    ax.plot([x for x, f in zip(xs, full) if not f], [y for y, f in zip(ys, full) if not f], 'o',
            mfc='white', color=focus_color, label='Calibrated focus, fewer regions (not comparable)')
    for x, y, r in zip(xs, ys, pts):
        if r['threshold'] in shown:
            ax.annotate(f"t={r['threshold']:g} ({r['n_scored_regions']}/{n_all})", (x, y),
                        textcoords='offset points', xytext=(5, -12 if r['threshold'] == .9 else 5), fontsize=7)
    for ref in references:
        m, c = style[ref['method']]
        ax.plot(ref['mean_full_image_coverage'] * 100, ref['region_mae_cm'], m, color=c, markersize=8,
                label=ref['label'])
    ax.set_xlabel('Full-image coverage (%)')
    ax.set_ylabel('Region MAE vs approximate labels (cm)')
    ax.set_title(f'Accuracy on {n_all} measured in-span regions', fontsize=10)
    ax.set_xlim(0, 105)
    ax.set_ylim(0, 6.5)
    ax.grid(alpha=.3)
    ax.legend(fontsize=7.5, loc='upper left')

    ax = axes[1]
    xs = [r['in_span_coverage'] * 100 for r in rows]
    ys = [r['median_abs_diff_from_lidar_in_span_cm'] for r in rows]
    ax.plot(xs, ys, 'o-', color=focus_color, label='Calibrated focus (confidence sweep)')
    for x, y, r in zip(xs, ys, rows):
        if r['threshold'] in shown:
            ax.annotate(f"t={r['threshold']:g}", (x, y), textcoords='offset points', xytext=(5, 5), fontsize=7)
    for ref in references:
        if ref['median_abs_diff_from_lidar_in_span_cm'] is not None:
            m, c = style[ref['method']]
            ax.plot(ref['in_span_coverage'] * 100, ref['median_abs_diff_from_lidar_in_span_cm'], m, color=c,
                    markersize=8, label=ref['label'])
    ax.set_xlabel('Coverage of pixels LiDAR places at 20–100 cm (%)')
    ax.set_ylabel('Median |depth − LiDAR| on those pixels (cm)')
    ax.set_title('Dense agreement with LiDAR inside the calibrated span\n(LiDAR is an imperfect reference)',
                 fontsize=10)
    ax.set_xlim(0, 105)
    ax.set_ylim(0, 7.5)
    ax.grid(alpha=.3)
    ax.legend(fontsize=7.5, loc='lower left')

    fig.suptitle('Accuracy vs coverage: three multi-object development scenes, '
                 'confidence threshold t swept (default 0.5)', fontsize=11)
    fig.savefig(output / 'coverage_accuracy_curve.png', dpi=150)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--results', type=Path, default=Path('results/multiscene_metric'))
    parser.add_argument('--output', type=Path, default=Path('results/coverage_curve'))
    parser.add_argument('--thresholds', type=float, nargs='+', default=DEFAULT_THRESHOLDS)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows, references = curve(args.results, args.thresholds)
    (args.output / 'curve.json').write_text(json.dumps(dict(
        scope='Describes the coverage/accuracy trade-off on development scenes; does not select a threshold.',
        source=str(args.results), curve=rows, reference_points=references), indent=2) + '\n')
    plot(rows, references, args.output)
    for r in rows:
        mae = '   n/a' if r['region_mae_cm'] is None else f"{r['region_mae_cm']:6.2f}"
        print(f"t={r['threshold']:<5g} coverage={r['mean_full_image_coverage']*100:5.1f}%  "
              f"regions={r['n_scored_regions']:2d}/{r['n_in_span_regions']}  MAE={mae} cm  "
              f"span cov={r['in_span_coverage']*100:5.1f}%  |vs LiDAR| span median={r['median_abs_diff_from_lidar_in_span_cm']:.2f} cm  "
              f"far false={r['fraction_retained_lidar_beyond_far']*100:.2f}%")
    for ref in references:
        lid = ref['median_abs_diff_from_lidar_in_span_cm']
        whole = ref['whole_image_median_abs_diff_from_lidar_cm']
        print(f"{ref['label']:<28} coverage={ref['mean_full_image_coverage']*100:5.1f}%  span cov={ref['in_span_coverage']*100:5.1f}%  "
              f"MAE={ref['region_mae_cm']:.2f} cm  |vs LiDAR| span median={'—' if lid is None else f'{lid:.2f} cm'}  "
              f"whole-image={'—' if whole is None else f'{whole:.1f} cm'}")
