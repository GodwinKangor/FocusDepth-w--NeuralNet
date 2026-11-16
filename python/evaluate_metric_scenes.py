"""Fresh multi-object calibrated-focus, AIF neural, fusion, and LiDAR comparison.

LiDAR and scene distances never enter optical-assisted neural scaling or fusion.
Annotations use native RGB coordinates; each output uses the common reference crop.
"""
import argparse
import hashlib
import json
import logging
from pathlib import Path

import cv2
import numpy as np

from compare_aif import transform_regions
from compare_metric_aif import prepare_input,restore_prediction
from focus_calibration import signature
from inspect_lidar_capture import load_depth
from metric_focus_fusion import scale_relative_with_focus,inverse_depth_fusion
from neural_depth import DepthPredictor


def region_statistics(values,region,min_coverage=.5):
    x0,y0,x1,y1=region['bbox'];h,w=values.shape
    if not (0<=x0<x1<=w and 0<=y0<y1<=h):raise ValueError('Region outside prediction')
    patch=values[y0:y1,x0:x1];valid=np.isfinite(patch)&(patch>0);coverage=float(valid.mean())
    median=float(np.median(patch[valid])) if valid.any() else None
    spread=float((np.percentile(patch[valid],90)-np.percentile(patch[valid],10))*100) if valid.any() else None
    eligible=region['within_measured_calibration_span'] and coverage>=min_coverage
    return dict(name=region['name'],nominal_cm=region['distance_m']*100,
                coverage=coverage,median_cm=None if median is None else median*100,
                spatial_p10_to_p90_span_cm=spread,
                signed_residual_cm=None if median is None else (median-region['distance_m'])*100,
                score_eligible=eligible,
                status='outside_measured_calibration_span' if not region['within_measured_calibration_span'] else ('valid' if eligible else 'insufficient_coverage'))


def render_comparison(reference,aif,maps,label,output):
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,4,figsize=(17,11),layout='constrained');axes=axes.ravel()
    for i,(im,title) in enumerate([(reference,'RGB reference'),(aif,'AIF RGB')]):
        axes[i].imshow(cv2.cvtColor(prepare_input(im,'upright'),cv2.COLOR_BGR2RGB));axes[i].set_title(title);axes[i].axis('off')
    titles={'calibrated_focus':'Calibrated focus', 'unadjusted_metric_aif':'Unadjusted metric AIF',
            'focus_scaled_relative_aif':'Focus-scaled relative AIF','focus_neural_fusion':'Focus/neural fusion','direct_lidar':'Direct LiDAR'}
    for i,(method,values) in enumerate(maps.items(),2):
        shown=axes[i].imshow(prepare_input(values,'upright'),vmin=.15,vmax=1.,cmap='viridis',interpolation='nearest')
        axes[i].set_title(titles[method]+' (m)',fontsize=10);axes[i].axis('off')
    for ax in axes[len(maps)+2:]:ax.axis('off')
    fig.colorbar(shown,ax=list(axes),shrink=.5,label='Meters; display range 0.15–1.0, arrays unclipped')
    fig.suptitle(label);fig.savefig(output/'comparison.png',dpi=130);plt.close(fig)


def evaluate(annotations_path,root,calibration_path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    annotations=json.loads(annotations_path.read_text());model_raw=calibration_path.read_bytes();calibration=json.loads(model_raw)
    relative=DepthPredictor('depth-anything-v2-small')
    metric=DepthPredictor('depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf',expected_type='metric',cache_dir='/private/tmp/focusdepth-metric-cache')
    all_rows=[];all_common_rows=[];scene_results=[]
    for scene in annotations['scenes']:
        label=scene['scene'];out=root/label;meta=json.loads((out/'preprocessing.json').read_text());folder=Path(scene['capture_directory']);manifest=json.loads((folder/'manifest.json').read_text())
        if meta['registration_failed']:raise ValueError('Cannot evaluate failed registration')
        if signature(manifest)!=calibration['signature']:raise ValueError('Calibration camera mismatch')
        if meta['calibration_model_sha256']!=hashlib.sha256(model_raw).hexdigest():raise ValueError('Calibration changed since preprocessing')
        if meta['source_manifest_sha256']!=hashlib.sha256((folder/'manifest.json').read_bytes()).hexdigest():raise ValueError('Source capture changed')
        reference=cv2.imread(str(out/'reference_rgb.png'));aif=cv2.imread(str(out/'all_in_focus.png'))
        regions=transform_regions(scene['regions'],meta['native_size'],meta['resized_size'],meta['crop'])
        optical=np.load(out/'optical_meters.npy');conf=np.load(out/'optical_confidence.npy')
        logging.info('%s: neural inference, upright orientation',label)
        rel,rel_info=relative.predict(prepare_input(aif,'upright'));rel=restore_prediction(rel,'upright')
        neural_metric,metric_info=metric.predict(prepare_input(aif,'upright'));neural_metric=restore_prediction(neural_metric,'upright')
        maps={'calibrated_focus':optical,'unadjusted_metric_aif':neural_metric};fit_status={}
        try:
            scaled,fit_status=scale_relative_with_focus(rel,optical,conf)
            maps['focus_scaled_relative_aif']=scaled
            maps['focus_neural_fusion']=inverse_depth_fusion(optical,scaled,conf)
        except ValueError as error:fit_status={'failed':str(error)}
        frame=next(f for f in manifest['frames'] if f['file']==meta['reference_file'])
        pair=json.loads((folder/frame['pair_metadata_file']).read_text());depth=load_depth(folder/frame['depth_file'],pair)
        if pair['orientation']!='native_sensor_unmirrored_rotation_0' or manifest['geometric_distortion_correction']:raise ValueError('Native RGB/depth mismatch')
        depth=np.where(np.isfinite(depth)&(depth>0),depth,np.nan)
        x0,y0,x1,y1=meta['crop'];resized=cv2.resize(depth,tuple(meta['resized_size']),interpolation=cv2.INTER_NEAREST)
        maps['direct_lidar']=resized[y0:y1,x0:x1]
        np.save(out/'relative_aif_raw.npy',rel)
        rows=[]
        for method,values in maps.items():
            np.save(out/(method+'_meters.npy'),values)
            for region in regions:
                row=region_statistics(values,region);row.update(scene=label,method=method);rows.append(row);all_rows.append(row)
        common=np.logical_and.reduce([np.isfinite(v)&(v>0) for v in maps.values()])
        common_rows=[]
        for method,values in maps.items():
            for region in regions:
                row=region_statistics(np.where(common,values,np.nan),region)
                row.update(scene=label,method=method);common_rows.append(row);all_common_rows.append(row)
        result=dict(common_support_region_results=common_rows,scene=label,regions=regions,preprocessing=meta,model_runs={'relative':rel_info,'metric':metric_info},
                    focus_assisted_fit=fit_status,region_results=rows,
                    output_coverage={k:float((np.isfinite(v)&(v>0)).mean()) for k,v in maps.items()},
                    scope='Scene distances evaluate only; LiDAR not used in focus or neural-assisted fitting/fusion.')
        (out/'scene_metrics.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');scene_results.append(result)
        render_comparison(reference,aif,maps,label,out)
        fig,axes=plt.subplots(1,2,figsize=(12,5));axes[0].imshow(cv2.cvtColor(reference,cv2.COLOR_BGR2RGB));axes[1].imshow(maps['direct_lidar'],vmin=.15,vmax=1.,cmap='viridis',interpolation='nearest')
        for number,region in enumerate(regions,1):
            a,b,c,d=region['bbox']
            for ax in axes:ax.add_patch(Rectangle((a,b),c-a,d-b,fill=False,edgecolor='red'));ax.text(a,b,str(number),color='white',bbox=dict(facecolor='black',alpha=.7))
        axes[0].set_title('Registered RGB evaluation patches');axes[1].set_title('Corresponding native LiDAR, nearest display');fig.suptitle(label);fig.tight_layout();fig.savefig(out/'regions.png',dpi=130);plt.close(fig)
        logging.info('%s: done, fit %s',label,fit_status)
    summaries=[]
    supported=sum(r['within_measured_calibration_span'] for s in annotations['scenes'] for r in s['regions'])
    for method in sorted({r['method'] for r in all_rows}):
        rows=[r for r in all_rows if r['method']==method and r['score_eligible']]
        common_rows=[r for r in all_common_rows if r['method']==method and r['score_eligible']]
        summaries.append(dict(method=method,n_common_support_regions=len(common_rows),
                              common_support_subset_mae_cm=float(np.mean([abs(r['signed_residual_cm']) for r in common_rows])) if common_rows else None,
                              n_scored_regions=len(rows),n_supported_regions=supported,
                              all_supported_region_mae_cm=float(np.mean([abs(r['signed_residual_cm']) for r in rows])) if len(rows)==supported else None,
                              covered_subset_mae_cm=float(np.mean([abs(r['signed_residual_cm']) for r in rows])) if rows else None))
    result=dict(scope='Exploratory multi-object metric-map comparison; independently measured scene uncertainty unspecified.',
                annotations=annotations,calibration_model_sha256=hashlib.sha256(model_raw).hexdigest(),
                minimum_region_coverage=.5,summary=summaries,scenes=scene_results)
    (root/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(summaries,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--annotations',type=Path,default=Path('data/annotations/multiscene_20261115.json'))
    parser.add_argument('--output',type=Path,default=Path('results/multiscene_metric'))
    parser.add_argument('--calibration',type=Path,default=Path('results/focus_calibration/model.json'))
    args=parser.parse_args();logging.basicConfig(level=logging.INFO);evaluate(args.annotations,args.output,args.calibration)
