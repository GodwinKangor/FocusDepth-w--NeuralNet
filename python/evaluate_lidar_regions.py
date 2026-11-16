"""Preliminary native-grid LiDAR checks against manually measured package faces.

No scale/offset fitting. RGB and AVDepthData share lens distortion; same-aspect
native grids are scaled for 2D sampling only. This does not rectify geometry for
3D projection or establish metrology-grade registration.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

from inspect_lidar_capture import load_depth


def depth_bbox(bbox, rgb_size, depth_size):
    rw, rh = rgb_size
    dw, dh = depth_size
    if rw <= 0 or rh <= 0 or dw <= 0 or dh <= 0 or rw * dh != rh * dw:
        raise ValueError('Native RGB/depth aspect ratios must match')
    x0, y0, x1, y1 = bbox
    if not (0 <= x0 < x1 <= rw and 0 <= y0 < y1 <= rh):
        raise ValueError('ROI outside native RGB grid')
    # Keep only depth cells fully inside the RGB rectangle.
    result = (int(np.ceil(x0 * dw / rw)), int(np.ceil(y0 * dh / rh)),
              int(np.floor(x1 * dw / rw)), int(np.floor(y1 * dh / rh)))
    if result[0] >= result[2] or result[1] >= result[3]:
        raise ValueError('ROI contains no complete depth cells')
    return result


def summarize_patch(depth, bbox):
    x0, y0, x1, y1 = bbox
    patch = depth[y0:y1, x0:x1]
    values = patch[np.isfinite(patch) & (patch > 0)]
    if not values.size:
        raise ValueError('ROI has no valid depth')
    return dict(median_m=float(np.median(values)),
                valid_fraction=float(values.size / patch.size),
                p10_m=float(np.percentile(values, 10)),
                p90_m=float(np.percentile(values, 90)))


def run(measurements: Path, output: Path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    record = json.loads(measurements.read_text())
    output.mkdir(parents=True, exist_ok=True)
    rows, details = [], []
    fig, axes = plt.subplots(len(record['scenes']), 3, figsize=(18, 5 * len(record['scenes'])), squeeze=False)
    for scene_idx, scene in enumerate(record['scenes']):
        folder = Path(scene['local_capture_directory'])
        manifest = json.loads((folder / 'manifest.json').read_text())
        rgb_size = (manifest['image_width'], manifest['image_height'])
        label = Path(scene['archive']).stem
        all_stats = {r['name']: [] for r in scene['regions']}
        jitter = {r['name']: [] for r in scene['regions']}
        reference_index = len(manifest['frames']) // 2
        for index, frame in enumerate(manifest['frames']):
            meta = json.loads((folder / frame['pair_metadata_file']).read_text())
            if meta['orientation'] != 'native_sensor_unmirrored_rotation_0' or manifest['geometric_distortion_correction']:
                raise ValueError('This 2D mapping requires matching unrectified native buffers')
            if (meta['intrinsics_reference_width'], meta['intrinsics_reference_height']) != rgb_size:
                raise ValueError('Unexpected calibration reference dimensions')
            depth = load_depth(folder / frame['depth_file'], meta)
            bboxes = {}
            for region in scene['regions']:
                bbox = depth_bbox(region['bbox'], rgb_size, (depth.shape[1], depth.shape[0]))
                bboxes[region['name']] = bbox
                stats = summarize_patch(depth, bbox)
                stats.update(frame=index, lens_position=frame['lens_position_actual'], depth_bbox=list(bbox))
                all_stats[region['name']].append(stats)
                shifted = []
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    x0,y0,x1,y1 = bbox
                    if 0 <= x0+dx < x1+dx <= depth.shape[1] and 0 <= y0+dy < y1+dy <= depth.shape[0]:
                        shifted.append(abs(summarize_patch(depth, (x0+dx,y0+dy,x1+dx,y1+dy))['median_m'] - stats['median_m']))
                jitter[region['name']].extend(shifted)
            if index == reference_index:
                rgb = cv2.cvtColor(cv2.imread(str(folder / frame['file'])), cv2.COLOR_BGR2RGB)
                if (rgb.shape[1], rgb.shape[0]) != rgb_size:
                    raise ValueError('Unexpected RGB dimensions')
                display_depth = np.where(np.isfinite(depth) & (depth > 0), depth, np.nan)
                axes[scene_idx,0].imshow(rgb)
                axes[scene_idx,1].imshow(display_depth, vmin=.15, vmax=1.35, cmap='viridis')
                # Nearest-neighbor display only: does not increase depth resolution.
                up = cv2.resize(display_depth, rgb_size, interpolation=cv2.INTER_NEAREST)
                axes[scene_idx,2].imshow(rgb)
                axes[scene_idx,2].imshow(up, vmin=.15, vmax=1.35, cmap='viridis', alpha=.45)
                for number, region in enumerate(scene['regions'], 1):
                    for col, box in [(0, region['bbox']), (1, bboxes[region['name']]), (2, region['bbox'])]:
                        x0,y0,x1,y1 = box
                        axes[scene_idx,col].add_patch(Rectangle((x0,y0),x1-x0,y1-y0,fill=False,edgecolor='red',linewidth=1.5))
                        axes[scene_idx,col].text(x0,y0,str(number),color='white',bbox=dict(facecolor='black',alpha=.7))
                for col,title in enumerate(['Native RGB regions','Native LiDAR depth (m)','2D correspondence overlay']):
                    axes[scene_idx,col].set_title(label+'\n'+title)
                    axes[scene_idx,col].axis('off')
        for number,region in enumerate(scene['regions'],1):
            stats = all_stats[region['name']]
            medians = np.array([s['median_m'] for s in stats])
            estimate = float(np.median(medians))
            error_cm = (estimate - region['distance_m']) * 100
            uncertainty = region['approximate_uncertainty_m']
            rows.append(dict(scene=label, region_number=number, object=region['name'],
                             nominal_cm=region['distance_m']*100,
                             median_lidar_cm=estimate*100, signed_residual_cm=error_cm,
                             approximate_measurement_uncertainty_cm=None if uncertainty is None else uncertainty*100,
                             sweep_median_range_cm=float(np.ptp(medians)*100),
                             minimum_valid_fraction=min(s['valid_fraction'] for s in stats),
                             max_one_depth_pixel_shift_change_cm=max(jitter[region['name']], default=0)*100))
            details.append(dict(scene=label, region=region, per_frame=stats))
    fig.tight_layout()
    fig.savefig(output/'annotated_regions.png',dpi=130)
    plt.close(fig)
    (output/'region_results.json').write_text(json.dumps(dict(
        scope='Preliminary unrectified native-grid 2D ROI comparison; no fitted correction, not certified accuracy.',
        limitations=['Package identity assignments inferred visually from confirmed distance order.',
                     'Measurement origin and camera-axis versus slant distance remain unspecified.',
                     'Wall measurement uncertainty is unspecified.',
                     'Within-sweep variation is not independent-capture repeatability.',
                     'Approximately 35 ms RGB/depth timestamp offset.',
                     'One-pixel shift checks test local ROI sensitivity, not complete registration accuracy.'],
        summary=rows,details=details),indent=2,allow_nan=False)+'\n')
    with (output/'region_results.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    for row in rows:
        print(f"{row['scene']:20} {row['object']:15} nominal {row['nominal_cm']:5.1f} LiDAR {row['median_lidar_cm']:6.2f} residual {row['signed_residual_cm']:+5.2f} cm; shift {row['max_one_depth_pixel_shift_change_cm']:.2f} cm")
    return rows


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--measurements',type=Path,default=Path('data/annotations/lidar_capture_measurements.json'))
    parser.add_argument('--output',type=Path,default=Path('results/lidar_regions'))
    args=parser.parse_args()
    run(args.measurements,args.output)
