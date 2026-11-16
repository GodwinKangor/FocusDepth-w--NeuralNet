"""Calibration/scene estimator transfer diagnostics; preserves frozen baseline.

Compare grayscale decoding, resizing, registration, smoothing, and regional versus
per-pixel peak aggregation. Candidate calibrations fit calibration captures only.
Already inspected validation/scene data are exploratory development evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path

import cv2
import numpy as np

import confidence
import focus_depth
import registration
from evaluate_metric_scenes import region_statistics
from focus_calibration import curve_peak,depth_from_focus,fit_mapping,signature
from run_pipeline import load_manifest,load_images

STAGES=['native_roi','native_bgr_gray_roi','resized_roi','registered_roi','smoothed_roi','pixel_median']


def roi_box(roi,native_size,resized_size,crop):
    nw,nh=native_size;rw,rh=resized_size;x0,y0,x1,y1=crop
    box=[round(roi[0]*nw)*rw/nw-x0,round(roi[1]*nh)*rh/nh-y0,
         round(roi[2]*nw)*rw/nw-x0,round(roi[3]*nh)*rh/nh-y0]
    box=[int(np.ceil(box[0])),int(np.ceil(box[1])),int(np.floor(box[2])),int(np.floor(box[3]))]
    if not (0<=box[0]<box[2]<=x1-x0 and 0<=box[1]<box[3]<=y1-y0):raise ValueError('Target ROI outside reference crop')
    return box


def regional_observation(base,sharp,lens,box):
    x0,y0,x1,y1=box;curve=sharp[:,y0:y1,x0:x1].mean(axis=(1,2))
    return {**base,'sharpness':curve.tolist(),**curve_peak(lens,curve)}


def measure_capture(base,roi,output):
    folder=Path(base['folder']);output.mkdir(parents=True,exist_ok=True)
    manifest,frames=load_manifest(folder,'lens_position_actual')
    if hashlib.sha256((folder/'manifest.json').read_bytes()).hexdigest()!=base['manifest_sha256']:raise ValueError('Source manifest changed')
    if signature(manifest)!=base['signature']:raise ValueError('Camera/format changed')
    lens=np.array([f['_lens'] for f in frames],np.float32);nw,nh=manifest['image_width'],manifest['image_height']
    results={'native_roi':base};errors={}
    native_box=roi_box(roi,(nw,nh),(nw,nh),(0,0,nw,nh));x0,y0,x1,y1=native_box
    native_curve=[]
    for frame in frames:
        im=cv2.imread(str(folder/frame['file']),cv2.IMREAD_COLOR|cv2.IMREAD_IGNORE_ORIENTATION)
        if im is None or im.shape[:2]!=(nh,nw):raise ValueError('Native dimensions mismatch')
        gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY).astype(np.float32)/255
        lap=np.abs(cv2.Laplacian(gray,cv2.CV_32F,ksize=3));native_curve.append(float(lap[y0:y1,x0:x1].mean()))
    try:results['native_bgr_gray_roi']={**base,'sharpness':native_curve,**curve_peak(lens,native_curve)}
    except ValueError as error:errors['native_bgr_gray_roi']=str(error)
    images=load_images(folder,frames,1600);h,w=images[0].shape[:2];ref=len(images)//2
    gray=np.stack([registration.to_gray_float(im) for im in images]);sharp=focus_depth.sharpness_stack(gray,3,0)
    box=roi_box(roi,(nw,nh),(w,h),(0,0,w,h))
    try:results['resized_roi']=regional_observation(base,sharp,lens,box)
    except ValueError as error:errors['resized_roi']=str(error)
    del gray,sharp
    logging.info('%s: registering %d frames',folder.name,len(images))
    reg=registration.register_stack(images,ref,motion='affine',iterations=200,eps=1e-6,pyramid_levels=3)
    meta=dict(crop=list(reg.crop),failed=reg.failed,warps=[warp.tolist() for warp in reg.warps],ecc_scores=reg.ecc_scores)
    if reg.failed:
        errors['registration']=f'Failed frame links: {reg.failed}'
    else:
        box=roi_box(roi,(nw,nh),(w,h),reg.crop);x0,y0,x1,y1=box
        gray=np.stack([registration.to_gray_float(im) for im in reg.images]);sharp=focus_depth.sharpness_stack(gray,3,0)
        try:results['registered_roi']=regional_observation(base,sharp,lens,box)
        except ValueError as error:errors['registered_roi']=str(error)
        del sharp
        smooth=focus_depth.sharpness_stack(gray,3,3.)
        try:results['smoothed_roi']=regional_observation(base,smooth,lens,box)
        except ValueError as error:errors['smoothed_roi']=str(error)
        point=focus_depth.optical_depth(smooth,lens,'peak');conf=confidence.optical_confidence(smooth,'prominence',True,.05,.25)
        values=point[y0:y1,x0:x1];supported=np.isfinite(values)&(conf[y0:y1,x0:x1]>=.5)
        np.save(output/'roi_pixel_focus.npy',np.where(supported,values,np.nan))
        fraction=float(supported.mean())
        if fraction<.5:errors['pixel_median']=f'Only {fraction:.3f} target ROI pixel confidence coverage'
        else:
            peak=float(np.median(values[supported]))
            if not lens[0]<peak<lens[-1]:errors['pixel_median']='Median pixel focus at sweep endpoint'
            else:results['pixel_median']={**base,'peak_lens_position':peak,'pixel_confidence_coverage':fraction,'peak_aggregation':'Median of per-pixel focus coordinates at confidence >=0.5'}
    payload=dict(source=base['folder'],observations=results,errors=errors,registration=meta)
    (output/'descriptors.json').write_text(json.dumps(payload,indent=2)+'\n')
    return payload


def score_validation(obs,model,stage,cache):
    if stage=='pixel_median':
        pixels=np.load(cache/'roi_pixel_focus.npy');meters=depth_from_focus(pixels,model);coverage=float(np.isfinite(meters).mean())
        if coverage<.5:return dict(status='insufficient_metric_coverage',coverage=coverage,predicted_m=None)
        value=float(np.nanmedian(meters))
    else:
        value=float(depth_from_focus(np.array([obs['peak_lens_position']]),model)[0]);coverage=float(np.isfinite(value))
        if not np.isfinite(value):return dict(status='outside_calibrated_range',coverage=coverage,predicted_m=None)
    return dict(status='valid',coverage=coverage,predicted_m=value,measured_m=obs['distance_m'],signed_residual_cm=(value-obs['distance_m'])*100)


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline',type=Path,default=Path('results/focus_calibration/model.json'))
    parser.add_argument('--output',type=Path,default=Path('results/focus_preprocessing_study'))
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True);logging.basicConfig(level=logging.INFO)
    baseline_raw=args.baseline.read_bytes();baseline=json.loads(baseline_raw)
    validation_rows=json.loads(Path('results/focus_validation_20261115/inspection.json').read_text())
    bases=[('calibration',o) for o in baseline['observations']]+[('validation',r['observation']) for r in validation_rows]
    calibration_hashes={o['manifest_sha256'] for role,o in bases if role=='calibration'}
    if any(o['manifest_sha256'] in calibration_hashes for role,o in bases if role=='validation'):
        raise ValueError('Calibration/validation source overlap')
    descriptors=[]
    for purpose,base in bases:
        cache=args.output/Path(base['folder']).name
        if (cache/'descriptors.json').exists():payload=json.loads((cache/'descriptors.json').read_text())
        else:payload=measure_capture(base,baseline['roi'],cache)
        # Cached results must identify the same source observation and capture role.
        manifest=json.loads((Path(base['folder'])/'manifest.json').read_text())
        if hashlib.sha256((Path(base['folder'])/'manifest.json').read_bytes()).hexdigest()!=base['manifest_sha256']:
            raise ValueError('Source manifest changed')
        if signature(manifest)!=base['signature']:raise ValueError('Camera/format changed')
        if manifest.get('capture_purpose')!=purpose:raise ValueError('Capture role changed')
        if payload['observations']['native_roi']['manifest_sha256']!=base['manifest_sha256']:raise ValueError('Descriptor source mismatch')
        descriptors.append(dict(purpose=purpose,base=base,payload=payload,cache=cache))
    models={'native_roi':baseline};summary=[];validation=[]
    # Candidate models are fitted without validation labels or scene labels.
    for stage in STAGES:
        calibration=[d['payload']['observations'][stage] for d in descriptors if d['purpose']=='calibration' and stage in d['payload']['observations']]
        fit_error=None
        if len(calibration)!=len(baseline['observations']):fit_error='Incomplete calibration descriptor coverage; candidate not fitted'
        elif stage!='native_roi':
            try:
                models[stage]=fit_mapping(calibration,baseline['roi']);models[stage]['focus_preprocessing_stage']=stage
                (args.output/(stage+'_model.json')).write_text(json.dumps(models[stage],indent=2)+'\n')
            except ValueError as error:fit_error=str(error)
        for d in descriptors:
            if d['purpose']!='validation':continue
            obs=d['payload']['observations'].get(stage)
            for model_name,model in [('frozen_native',baseline),('matched_stage',models.get(stage))]:
                if obs is None or model is None:
                    result=dict(status='unavailable',reason=d['payload']['errors'].get(stage,fit_error),predicted_m=None)
                else:result=score_validation(obs,model,stage,d['cache'])
                validation.append(dict(stage=stage,model=model_name,capture=Path(d['base']['folder']).name,nominal_m=d['base']['distance_m'],**result))
        conditions={}
        for model_name in ['frozen_native','matched_stage']:
            rows=[r for r in validation if r['stage']==stage and r['model']==model_name];valid=[r for r in rows if r['status']=='valid']
            conditions[model_name]=dict(n_valid=len(valid),n_total=len(rows),mae_cm=float(np.mean([abs(r['signed_residual_cm']) for r in valid])) if len(valid)==len(rows) else None,
                                        covered_subset_mae_cm=float(np.mean([abs(r['signed_residual_cm']) for r in valid])) if valid else None)
        summary.append(dict(stage=stage,calibration_capture_count=len(calibration),fit_error=fit_error,validation=conditions))
        logging.info('%s: %s',stage,conditions)
    # Apply matched calibration candidates to the identical cached scene focus maps;
    # do not change focus coordinates, confidence masks, neural scaling or fusion.
    scenes=json.loads(Path('results/multiscene_metric/summary.json').read_text());scene_rows=[]
    for scene in scenes['scenes']:
        folder=Path('results/multiscene_metric')/scene['scene'];focus=np.load(folder/'focus_coordinate.npy');conf=np.load(folder/'optical_confidence.npy')
        for stage in ['native_roi','smoothed_roi','pixel_median']:
            if stage not in models:continue
            optical=depth_from_focus(focus,models[stage]);optical[conf<.5]=np.nan
            out=args.output/scene['scene'];out.mkdir(parents=True,exist_ok=True);np.save(out/(stage+'_optical_meters.npy'),optical)
            for region in scene['regions']:
                result=region_statistics(optical,region);scene_rows.append(dict(scene=scene['scene'],stage=stage,**result))
    scene_summary=[]
    for stage in ['native_roi','smoothed_roi','pixel_median']:
        rows=[r for r in scene_rows if r['stage']==stage and r['score_eligible']]
        scene_summary.append(dict(stage=stage,n_supported_regions=len(rows),expected_supported_regions=11,mae_cm=float(np.mean([abs(r['signed_residual_cm']) for r in rows])) if len(rows)==11 else None))
    output=dict(scope='Processing-transfer diagnostic; calibration fitted only on calibration captures; already inspected validation/scenes are exploratory.',
                baseline_sha256=hashlib.sha256(baseline_raw).hexdigest(),production_model_changed=False,
                summary=summary,validation=validation,scene_summary=scene_summary,scene_regions=scene_rows,
                descriptors=[dict(purpose=d['purpose'],**d['payload']) for d in descriptors])
    (args.output/'results.json').write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')
    assert args.baseline.read_bytes()==baseline_raw
    fig,axes=plt.subplots(1,2,figsize=(13,4),layout='constrained')
    names=[r['stage'] for r in summary]
    for kind in ['frozen_native','matched_stage']:
        values=[r['validation'][kind]['mae_cm'] if r['validation'][kind]['mae_cm'] is not None else np.nan for r in summary]
        axes[0].plot(names,values,'o-',label=kind)
    axes[0].set_ylabel('Initial 4-target development MAE cm');axes[0].tick_params(axis='x',rotation=35);axes[0].legend()
    axes[1].bar([r['stage'] for r in scene_summary],[r['mae_cm'] if r['mae_cm'] is not None else np.nan for r in scene_summary]);axes[1].set_ylabel('Multi-object region MAE cm; incomplete coverage omitted');axes[1].tick_params(axis='x',rotation=25)
    fig.savefig(args.output/'comparison.png',dpi=140);print(json.dumps(summary,indent=2));print(json.dumps(scene_summary,indent=2))


if __name__=='__main__':main()
