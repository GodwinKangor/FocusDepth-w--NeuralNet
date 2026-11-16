"""Device-specific focus calibration with independent validation and no extrapolation.

Capture one flat textured target at each known distance. Use `fit` only with
calibration-purpose captures, and `validate` only with new validation captures.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

CORRECTIONS = Path(__file__).resolve().parent.parent / 'data/annotations/focus_measurement_corrections.json'


def corrected_uncertainty(original, manifest_hash, corrections):
    correction=corrections.get('corrections',{}).get(manifest_hash)
    if correction is None:return original,None
    if original!=correction['original_uncertainty_m']:
        raise ValueError('Measurement correction does not match original uncertainty')
    value=correction['corrected_uncertainty_m']
    if not np.isfinite(value) or value<0:raise ValueError('Invalid uncertainty correction')
    return value,correction


def signature(manifest):
    keys=['capture_device_id','camera_unique_id','camera','image_width','image_height','geometric_distortion_correction']
    if not manifest.get('camera_unique_id') or not manifest.get('capture_device_id'):
        raise ValueError('Capture needs camera and installation IDs; use the updated app')
    return {key:manifest[key] for key in keys}


def curve_peak(lens, scores, min_prominence=.15):
    lens=np.asarray(lens,dtype=float);scores=np.asarray(scores,dtype=float)
    if lens.ndim!=1 or scores.shape!=lens.shape or len(lens)<3 or not np.isfinite(lens).all() or not np.isfinite(scores).all() or (np.diff(lens)<=0).any():
        raise ValueError('Need finite focus scores and distinct ascending actual lens positions')
    if np.any(scores<0):raise ValueError('Sharpness cannot be negative')
    k=int(scores.argmax())
    prominence=float((scores[k]-np.median(scores))/max(scores[k],1e-12))
    if k in (0,len(lens)-1):raise ValueError('Best focus is at sweep endpoint; bracket the target focus')
    if prominence<min_prominence:raise ValueError('Focus response too flat for calibration')
    # Fit against actual (possibly nonuniform) lens positions, not frame indices.
    origin=lens[k];xs=lens[k-1:k+2]-origin
    a,b,_=np.polyfit(xs,scores[k-1:k+2],2)
    if a>=0:raise ValueError('Local focus response is not concave')
    peak=origin-b/(2*a)
    if not lens[k-1]<=peak<=lens[k+1]:raise ValueError('Unstable substep focus peak')
    return dict(peak_lens_position=float(peak),peak_frame=k,prominence=prominence)


def observation(folder, purpose, roi):
    path=folder/'manifest.json';raw=path.read_bytes();manifest=json.loads(raw)
    if manifest.get('capture_purpose')!=purpose:
        raise ValueError(f'{folder}: expected {purpose} capture; never reuse calibration as validation')
    distance=manifest.get('calibration_target_distance_m')
    uncertainty=manifest.get('measurement_uncertainty_m')
    manifest_hash=hashlib.sha256(raw).hexdigest()
    uncertainty,correction=corrected_uncertainty(uncertainty,manifest_hash,
        json.loads(CORRECTIONS.read_text()) if CORRECTIONS.exists() else {})
    if distance is None or not np.isfinite(distance) or distance<=0 or uncertainty is None or not np.isfinite(uncertainty) or uncertainty<0:
        raise ValueError('Missing positive measured target distance or nonnegative uncertainty')
    if len(roi)!=4 or not (0<=roi[0]<roi[2]<=1 and 0<=roi[1]<roi[3]<=1):
        raise ValueError('ROI must be normalized x0 y0 x1 y1')
    frames=sorted(manifest['frames'],key=lambda f:f['lens_position_actual'])
    scores=[];lens=[]
    for frame in frames:
        image=cv2.imread(str(folder/frame['file']),cv2.IMREAD_GRAYSCALE|cv2.IMREAD_IGNORE_ORIENTATION)
        if image is None:raise ValueError('Missing calibration image')
        if image.shape!=(manifest['image_height'],manifest['image_width']):raise ValueError('Native image dimensions do not match manifest')
        h,w=image.shape;x0,y0,x1,y1=[round(v*s) for v,s in zip(roi,[w,h,w,h])]
        if x1-x0<8 or y1-y0<8:raise ValueError('Target ROI is too small')
        lap=np.abs(cv2.Laplacian(image.astype(np.float32)/255,cv2.CV_32F,ksize=3))
        scores.append(float(lap[y0:y1,x0:x1].mean()));lens.append(frame['lens_position_actual'])
    peak=curve_peak(lens,scores)
    return dict(folder=str(folder),manifest_sha256=manifest_hash,
                capture_identity=[manifest['camera_unique_id'],manifest['created_at']],
                signature=signature(manifest),distance_m=float(distance),uncertainty_m=float(uncertainty),
                uncertainty_original_m=manifest.get('measurement_uncertainty_m'),uncertainty_correction=correction,
                lens_positions=lens,sharpness=scores,**peak)


def fit_mapping(observations, roi):
    if not observations:raise ValueError('No calibration observations')
    expected=observations[0]['signature']
    if any(o['signature']!=expected for o in observations):raise ValueError('Calibration mixes cameras or capture formats')
    if len({tuple(o['capture_identity']) for o in observations})!=len(observations):raise ValueError('Duplicate calibration capture')
    distances=sorted({o['distance_m'] for o in observations})
    if len(distances)<4:raise ValueError('Need at least four distinct calibration distances')
    groups=[dict(distance_m=d,peak_lens_position=float(np.median([o['peak_lens_position'] for o in observations if o['distance_m']==d])),
                 repeats=sum(o['distance_m']==d for o in observations)) for d in distances]
    peaks=np.array([g['peak_lens_position'] for g in groups])
    if (np.diff(peaks)<=0).any():raise ValueError('Focus peaks are not strictly ordered with distance; improve capture/calibration rather than forcing a fit')
    return dict(schema_version=1,method='piecewise_linear_inverse_distance',signature=expected,roi=roi,
                groups=groups,observations=observations,valid_lens_range=[float(peaks.min()),float(peaks.max())],
                note='No extrapolation. Calibration residual is not independent validation accuracy.')


def depth_from_focus(values, model):
    if model['method']!='piecewise_linear_inverse_distance':raise ValueError('Unknown calibration method')
    lens=np.array([g['peak_lens_position'] for g in model['groups']])
    inverse=np.array([1/g['distance_m'] for g in model['groups']])
    values=np.asarray(values,dtype=float)
    result=np.full(values.shape,np.nan,dtype=np.float32)
    mask=np.isfinite(values)&(values>=lens[0])&(values<=lens[-1])
    result[mask]=1/np.interp(values[mask],lens,inverse)
    return result


def validate_observation(obs,model):
    if obs['signature']!=model['signature']:raise ValueError('Camera/capture format differs from calibration')
    if any(obs['manifest_sha256']==o['manifest_sha256'] or obs['capture_identity']==o['capture_identity'] for o in model['observations']):
        raise ValueError('Calibration capture reused as validation')
    value=float(depth_from_focus(np.array([obs['peak_lens_position']]),model)[0])
    if not np.isfinite(value):
        return dict(folder=obs['folder'],status='outside_calibrated_range',measured_m=obs['distance_m'],predicted_m=None)
    return dict(folder=obs['folder'],status='valid',measured_m=obs['distance_m'],predicted_m=value,
                signed_residual_cm=(value-obs['distance_m'])*100,measurement_uncertainty_m=obs['uncertainty_m'])


def diagnose_distances(model):
    """Leave out every capture at one distance; this is not independent validation."""
    results=[]
    for distance in sorted({o['distance_m'] for o in model['observations']}):
        training=[o for o in model['observations'] if o['distance_m']!=distance]
        held_out=[o for o in model['observations'] if o['distance_m']==distance]
        if len({o['distance_m'] for o in training})<4:
            results.extend(dict(folder=o['folder'],measured_m=distance,status='insufficient_remaining_distances',predicted_m=None) for o in held_out)
            continue
        partial=fit_mapping(training,model['roi'])
        for obs in held_out:
            predicted=float(depth_from_focus(np.array([obs['peak_lens_position']]),partial)[0])
            valid=np.isfinite(predicted)
            results.append(dict(folder=obs['folder'],measured_m=distance,
                                predicted_m=predicted if valid else None,
                                signed_residual_cm=(predicted-distance)*100 if valid else None,
                                status='valid' if valid else 'outside_remaining_calibrated_range',
                                training_distances_m=sorted({o['distance_m'] for o in training})))
    return dict(scope='Leave-one-distance-out calibration diagnostic; not independent validation',results=results)


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    fit=sub.add_parser('fit');fit.add_argument('captures',nargs='+',type=Path)
    fit.add_argument('--roi',nargs=4,type=float,default=[.45,.45,.55,.55]);fit.add_argument('--output',required=True,type=Path)
    val=sub.add_parser('validate');val.add_argument('model',type=Path);val.add_argument('captures',nargs='+',type=Path);val.add_argument('--output',required=True,type=Path)
    diag=sub.add_parser('diagnose');diag.add_argument('model',type=Path);diag.add_argument('--output',required=True,type=Path)
    apply=sub.add_parser('apply');apply.add_argument('model',type=Path);apply.add_argument('--manifest',required=True,type=Path)
    apply.add_argument('--focus-map',required=True,type=Path);apply.add_argument('--confidence',required=True,type=Path)
    apply.add_argument('--min-confidence',type=float,default=.5);apply.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    if args.command=='fit':
        model=fit_mapping([observation(p,'calibration',args.roi) for p in args.captures],args.roi)
        output=model
    elif args.command=='diagnose':
        output=diagnose_distances(json.loads(args.model.read_text()))
    elif args.command=='validate':
        model=json.loads(args.model.read_text())
        output=dict(results=[validate_observation(observation(p,'validation',model['roi']),model) for p in args.captures],
                    note='Independent captures only. Approximate measurement uncertainty is not a complete error budget.')
    else:
        model=json.loads(args.model.read_text());manifest=json.loads(args.manifest.read_text())
        if signature(manifest)!=model['signature']:raise ValueError('Camera/format mismatch')
        focus=np.load(args.focus_map);confidence=np.load(args.confidence)
        if focus.shape!=confidence.shape or not 0<=args.min_confidence<=1:raise ValueError('Confidence/map mismatch')
        output=depth_from_focus(focus,model);output[~np.isfinite(confidence)|(confidence<args.min_confidence)]=np.nan
        args.output.parent.mkdir(parents=True,exist_ok=True)
        with args.output.open('wb') as stream:np.save(stream,output)
        print('Saved metric optical map; valid coverage',float(np.isfinite(output).mean()));return
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')
    print('Saved',args.output)


if __name__=='__main__':main()
