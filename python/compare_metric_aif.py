"""Unadjusted metric DAV2 reference/AIF evaluation with an orientation control.

Reuses the registered, cropped RGB inputs and regions of compare_aif.py. Nothing
fits model predictions to LiDAR or the supplied tape measurements. Package ROI
medians are the evaluation units; they are not dense ground-truth depth maps.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path

import cv2
import numpy as np

from neural_depth import DepthPredictor

MODEL = 'depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf'


def prepare_input(image, orientation):
    if orientation == 'upright':
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    if orientation == 'native':
        return image
    raise ValueError('Unknown orientation')


def restore_prediction(prediction, orientation):
    if orientation == 'upright':
        return cv2.rotate(prediction, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if orientation == 'native':
        return prediction
    raise ValueError('Unknown orientation')


def region_result(prediction, region):
    x0,y0,x1,y1 = region['bbox']
    h,w = prediction.shape
    if not (0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h):
        raise ValueError('Region outside prediction')
    patch=prediction[y0:y1,x0:x1]
    valid=np.isfinite(patch)&(patch>0)
    if not valid.any():
        raise ValueError('No positive finite metric predictions in region')
    values=patch[valid]
    median=float(np.median(values))
    residual_cm=(median-region['distance_m'])*100
    uncertainty=region['approximate_uncertainty_m']
    uncertainty_cm=None if uncertainty is None else uncertainty*100
    return dict(object=region['name'],nominal_cm=region['distance_m']*100,
                predicted_cm=median*100,signed_residual_cm=residual_cm,
                absolute_residual_cm=abs(residual_cm),
                measurement_uncertainty_cm=uncertainty_cm,
                absolute_residual_lower_bound_cm=None if uncertainty_cm is None else max(0,abs(residual_cm)-uncertainty_cm),
                absolute_residual_upper_bound_cm=None if uncertainty_cm is None else abs(residual_cm)+uncertainty_cm,
                valid_fraction=float(valid.mean()),p10_cm=float(np.percentile(values,10)*100),
                p90_cm=float(np.percentile(values,90)*100))


def aggregate(rows):
    result=[]
    for orientation in ['native','upright']:
        for input_name in ['reference','aif']:
            packages=[r for r in rows if r['orientation']==orientation and r['input']==input_name and r['object']!='wall']
            residual=np.array([r['signed_residual_cm'] for r in packages])
            result.append(dict(orientation=orientation,input=input_name,n_package_regions=len(packages),
                               mae_cm=float(np.abs(residual).mean()),rmse_cm=float(np.sqrt(np.mean(residual**2))),
                               mean_signed_residual_cm=float(residual.mean()),
                               mean_absolute_relative_error=float(np.mean([r['absolute_residual_cm']/r['nominal_cm'] for r in packages])),
                               within_1cm=sum(r['absolute_residual_cm']<=1 for r in packages)))
    return result


def write_report(rows, summary, report, model_info):
    upright={item['input']:item for item in summary if item['orientation']=='upright'}
    all_farther=all(row['signed_residual_cm']>0 for row in rows if row['object']!='wall')
    sentence=('The unadjusted checkpoint overestimates every sampled package-region '
              'distance in both tested orientations.' if all_farther else
              'The unadjusted checkpoint has mixed signed package-region residuals.')
    lines=['# Unadjusted metric depth: reference versus AIF','','## Finding','',
           sentence,
           f"Upright package MAE is {upright['reference']['mae_cm']:.2f} cm for reference "
           f"and {upright['aif']['mae_cm']:.2f} cm for AIF inputs.",
           'The tables below retain both orientations and all measured regions. These',
           'results apply to this Small indoor checkpoint and these captures, not all',
           'possible models. AIF sharpness alone does not establish physical scale.',
           '', '## Experiment','','Indoor metric Depth Anything V2 Small was applied to the existing registered','reference/AIF pairs. No scale, shift, focal-length correction, LiDAR fitting,','tape-distance fitting, or fusion was applied. Metric outputs were used directly','in meters and converted to centimeters only for reporting.','','Both original sideways sensor orientation and a predefined 90-degree clockwise','upright input were tested. Upright predictions were rotated back to the shared','native crop before region sampling. This is an orientation control, not a choice','of the best orientation separately for each scene. The primary package table','below uses upright inputs; native results remain in the aggregate table and CSV.','','Checkpoint: `'+MODEL+'`.','Model revision: `'+str(model_info.get('model_revision'))+'`.','Depth type: `'+str(model_info.get('depth_estimation_type'))+'`; maximum output depth: '+str(model_info.get('model_max_depth'))+' m.','','Source: [official metric model card](https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf).','','## Package-region aggregates','','MAE/RMSE are across nine package-region medians, equally weighted by region.','They are not per-pixel errors, independently repeated trials, or held-out benchmark','results. Wall patches are excluded from these aggregates.','','| Orientation | Input | Package MAE cm | Package RMSE cm | Mean signed residual cm | Regions within nominal ±1 cm |','| --- | --- | ---: | ---: | ---: | ---: |']
    for s in summary:
        lines.append(f"| {s['orientation']} | {s['input']} | {s['mae_cm']:.2f} | {s['rmse_cm']:.2f} | {s['mean_signed_residual_cm']:+.2f} | {s['within_1cm']}/9 |")
    lines+=['','## Upright region predictions','','Positive residual means predicted farther than the nominal measured face.','','| Scene | Region | Nominal cm | Reference cm | AIF cm | Reference residual cm | AIF residual cm |','| --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for a in rows:
        if a['orientation']!='upright' or a['input']!='reference':continue
        b=next(r for r in rows if r['orientation']=='upright' and r['input']=='aif' and r['scene']==a['scene'] and r['object']==a['object'])
        lines.append(f"| {a['scene'].removeprefix('scene-')} | {a['object']} | {a['nominal_cm']:.0f} | {a['predicted_cm']:.2f} | {b['predicted_cm']:.2f} | {a['signed_residual_cm']:+.2f} | {b['signed_residual_cm']:+.2f} |")
    lines+=['','## Interpretation limits','','- Meter-valued prediction is not proof of accurate physical scale.','- Object nominal distances have approximate ±1 cm surface-location ambiguity.','  JSON/CSV retain residual bounds from this interval; these are descriptive','  bounds, not statistical confidence intervals or a complete error budget.','- Wall distance is approximately 119 cm with unspecified uncertainty.','- Measurement origin, axial versus slant distance, surface tilt and package','  curvature remain unresolved. The 85 cm rear box is a weak, partly occluded ROI.','- The broad indoor checkpoint may not match close tabletop imagery or iPhone','  intrinsics. AIF halos and image processing can also affect predictions.','- Three arrangements without independent repeats do not establish generalization.','- This metric model is separate from the relative checkpoint in the earlier AIF','  ablation. The relative-fusion path remains in lens-position units.','','## Reproduction and artifacts','','First generate the shared registered inputs with `python python/compare_aif.py`.','Then run:','','```bash','python python/compare_metric_aif.py','```','','The default model cache is `/private/tmp/focusdepth-metric-cache`; it may need','downloading again after temporary files are cleared. Model inference metadata,','shared regions/crops, orientation, and per-region spatial percentiles are saved in','`results/metric_aif/region_results.json`. Percentiles describe spatial predictions,','not uncertainty in the true measured distance. CSV and per-scene metric arrays','are saved beside it, with upright visual comparisons in each scene folder.']
    report.parent.mkdir(parents=True,exist_ok=True)
    report.write_text('\n'.join(lines)+'\n')


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=Path('results/aif_ablation'))
    parser.add_argument('--output',type=Path,default=Path('results/metric_aif'))
    parser.add_argument('--report',type=Path,default=Path('docs/metric_aif_results.md'))
    parser.add_argument('--cache-dir',default='/private/tmp/focusdepth-metric-cache')
    args=parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    predictor=DepthPredictor(MODEL,expected_type='metric',cache_dir=args.cache_dir)
    previous=json.loads((args.input/'summary.json').read_text())
    args.output.mkdir(parents=True,exist_ok=True)
    rows=[];runs=[]
    for scene in previous:
        source=args.input/scene['scene'];out=args.output/scene['scene'];out.mkdir(parents=True,exist_ok=True)
        images={name:cv2.imread(str(source/filename)) for name,filename in [('reference','reference_rgb.png'),('aif','all_in_focus.png')]}
        if any(im is None for im in images.values()):raise ValueError('Missing shared input images')
        if images['reference'].shape!=images['aif'].shape:raise ValueError('Input pair shapes differ')
        crop=scene['registration']['crop']
        if images['reference'].shape[:2]!=(crop[3]-crop[1],crop[2]-crop[0]):raise ValueError('Cached crop dimensions disagree')
        upright_maps={}
        for orientation in ['native','upright']:
            for name,image in images.items():
                logging.info('%s: metric %s %s',scene['scene'],orientation,name)
                pred,info=predictor.predict(prepare_input(image,orientation))
                pred=restore_prediction(pred,orientation)
                if pred.shape!=image.shape[:2]:raise ValueError('Prediction orientation/shape mismatch')
                np.save(out/f'{orientation}_{name}_meters.npy',pred)
                runs.append(dict(scene=scene['scene'],orientation=orientation,input=name,model=info,
                                 source_registration=scene['registration'],regions=scene['regions']))
                for region in scene['regions']:
                    row=region_result(pred,region)
                    row.update(scene=scene['scene'],orientation=orientation,input=name)
                    rows.append(row)
                if orientation=='upright':upright_maps[name]=pred
        fig,axes=plt.subplots(1,4,figsize=(12,6))
        values=np.concatenate([p.ravel() for p in upright_maps.values()]);lo,hi=np.percentile(values[np.isfinite(values)&(values>0)],[1,99])
        for ax,name in zip(axes[:2],['reference','aif']):
            ax.imshow(cv2.cvtColor(prepare_input(images[name],'upright'),cv2.COLOR_BGR2RGB));ax.set_title(name+' RGB');ax.axis('off')
        for ax,name in zip(axes[2:],['reference','aif']):
            shown=ax.imshow(prepare_input(upright_maps[name],'upright'),vmin=lo,vmax=hi,cmap='viridis');ax.set_title(name+' depth (m)');ax.axis('off')
        fig.colorbar(shown,ax=list(axes[2:]),shrink=.5,label='Predicted meters')
        fig.suptitle(scene['scene']+' — upright metric model, unadjusted')
        fig.savefig(out/'upright_comparison.png',dpi=140,bbox_inches='tight');plt.close(fig)
    summary=aggregate(rows)
    (args.output/'region_results.json').write_text(json.dumps(dict(model=MODEL,summary=summary,regions=rows,runs=runs),indent=2,allow_nan=False)+'\n')
    with (args.output/'region_results.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    write_report(rows,summary,args.report,runs[0]['model'])
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
