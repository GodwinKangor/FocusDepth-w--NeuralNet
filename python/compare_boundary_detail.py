"""Exploratory RGB-silhouette line probes plus controlled reference/AIF neural inputs.

Image sharpness and normalized depth-edge width are not metric depth accuracy.
Annotations are approximate RGB silhouettes, not dense depth ground truth.
"""
from __future__ import annotations

import argparse
import json
import logging
import warnings
from pathlib import Path

import cv2
import numpy as np

from compare_metric_aif import prepare_input,restore_prediction
from neural_depth import DepthPredictor


def crossings(x,y,level):
    valid=np.isfinite(y[:-1])&np.isfinite(y[1:])&(y[:-1]<=level)&(y[1:]>level)
    idx=np.flatnonzero(valid)
    return [(float(x[i]+(level-y[i])*(x[i+1]-x[i])/(y[i+1]-y[i])),float((y[i+1]-y[i])/(x[i+1]-x[i]))) for i in idx]


def score_profile(offsets, profile, global_span, min_contrast=.02):
    x=np.asarray(offsets,dtype=float);y=np.asarray(profile,dtype=float)
    fg=y[(x>=-40)&(x<=-25)];bg=y[(x>=25)&(x<=40)]
    if np.isfinite(fg).mean()<.5 or np.isfinite(bg).mean()<.5:
        return dict(status='missing_anchor_support')
    left=float(np.nanmedian(fg));right=float(np.nanmedian(bg));step=right-left
    if step<=max(1e-10,min_contrast*global_span):
        return dict(status='wrong_direction_or_weak_step',step=step)
    normalized=(y-left)/step
    half=crossings(x,normalized,.5)
    if not half:return dict(status='missing_half_crossing',normalized_profile=normalized.tolist())
    # Strongest rising crossing; never choose whichever is closest to the annotation.
    midpoint=max(half,key=lambda pair:pair[1])[0]
    lower=[p[0] for p in crossings(x,normalized,.1) if p[0]<=midpoint]
    upper=[p[0] for p in crossings(x,normalized,.9) if p[0]>=midpoint]
    width=min(upper)-max(lower) if lower and upper else None
    return dict(status='valid',signed_offset_native_px=midpoint,
                absolute_offset_native_px=abs(midpoint),transition_width_native_px=width,
                n_rising_half_crossings=len(half),normalized_profile=normalized.tolist())


def far_inverse_metric(depth):
    result=np.full(depth.shape,np.nan,np.float32)
    valid=np.isfinite(depth)&(depth>0)
    np.divide(-1,depth,out=result,where=valid)
    return result


def probe_map(values,edge,meta,half_width=45,min_coverage=.5):
    offsets=np.arange(-half_width,half_width+1,dtype=np.float32)
    start=np.asarray(edge['start'],np.float32);end=np.asarray(edge['end'],np.float32)
    # Keep the normal perpendicular to the actual annotated segment.
    tangent=end-start;normal=np.array([-tangent[1],tangent[0]],np.float32);normal/=np.linalg.norm(normal)
    if np.dot(normal,edge['background_normal'])<0:normal=-normal
    centers=np.linspace(start,end,64)
    points=centers[:,None,:]+offsets[None,:,None]*normal
    nw,nh=meta['native_size'];rw,rh=meta['resized_size'];x0,y0,_,_=meta['crop']
    mx=(points[...,0]+.5)*rw/nw-.5-x0;my=(points[...,1]+.5)*rh/nh-.5-y0
    finite=np.isfinite(values)
    samples=cv2.remap(np.where(finite,values,0).astype(np.float32),mx.astype(np.float32),my.astype(np.float32),cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
    support=cv2.remap(finite.astype(np.float32),mx.astype(np.float32),my.astype(np.float32),cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
    samples[support<.999]=np.nan
    coverage=np.isfinite(samples).mean(axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning)
        profile=np.nanmedian(samples,axis=0)
    profile[coverage<min_coverage]=np.nan
    span=float(np.diff(np.percentile(values[finite],[10,90]))[0]) if finite.any() else 0
    score=score_profile(offsets,profile,span)
    score.update(offsets_native_px=offsets.tolist(),profile_values=profile.tolist(),
                 sample_coverage=float(np.isfinite(samples).mean()),column_coverage=coverage.tolist())
    return score


def json_safe(value):
    if isinstance(value,dict):return {k:json_safe(v) for k,v in value.items()}
    if isinstance(value,list):return [json_safe(v) for v in value]
    if isinstance(value,float) and not np.isfinite(value):return None
    return value


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--annotations',type=Path,default=Path('data/annotations/boundary_probes_20261115.json'))
    parser.add_argument('--input',type=Path,default=Path('results/multiscene_metric'))
    parser.add_argument('--output',type=Path,default=Path('results/boundary_detail_study'))
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True);logging.basicConfig(level=logging.INFO)
    annotations=json.loads(args.annotations.read_text());predictor=DepthPredictor('depth-anything-v2-small')
    rows=[];image_detail=[];model_runs=[]
    for scene in annotations['scenes']:
        label=scene['scene'];source=args.input/label;out=args.output/label;out.mkdir(parents=True,exist_ok=True)
        meta=json.loads((source/'preprocessing.json').read_text());metrics=json.loads((source/'scene_metrics.json').read_text())
        reference=cv2.imread(str(source/'reference_rgb.png'));aif=cv2.imread(str(source/'all_in_focus.png'))
        logging.info('%s: reference-frame neural inference',label)
        raw,info=predictor.predict(prepare_input(reference,'upright'));raw=restore_prediction(raw,'upright')
        original=metrics['model_runs']['relative']
        if info['model']!=original['model'] or info['model_revision']!=original['model_revision'] or info['processor_input_shape']!=original['processor_input_shape']:
            raise ValueError('Reference/AIF model or preprocessing differs')
        np.save(out/'reference_relative_raw.npy',raw)
        maps={'reference_neural':-raw,'aif_neural':-np.load(source/'relative_aif_raw.npy'),
              'calibrated_focus':far_inverse_metric(np.load(source/'calibrated_focus_meters.npy')),
              'aif_focus_neural_fusion':far_inverse_metric(np.load(source/'focus_neural_fusion_meters.npy')),
              'direct_lidar':far_inverse_metric(np.load(source/'direct_lidar_meters.npy'))}
        model_runs.append(dict(scene=label,reference=info,aif=original))
        fig,axes=plt.subplots(len(scene['edges']),3,figsize=(14,3.5*len(scene['edges'])),squeeze=False,layout='constrained')
        for index,edge in enumerate(scene['edges']):
            # RGB-only guide window; show the full sampled strip context.
            nw,nh=meta['native_size'];rw,rh=meta['resized_size'];ox,oy,_,_=meta['crop']
            start=(np.array(edge['start'])+.5)*[rw/nw,rh/nh]-.5-[ox,oy]
            end=(np.array(edge['end'])+.5)*[rw/nw,rh/nh]-.5-[ox,oy]
            lo=np.floor(np.minimum(start,end)-50).astype(int);hi=np.ceil(np.maximum(start,end)+50).astype(int)
            lo=np.maximum(lo,0);hi=np.minimum(hi,[reference.shape[1],reference.shape[0]])
            for col,(image,title) in enumerate([(reference,'Reference RGB'),(aif,'AIF RGB')]):
                axes[index,col].imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB));axes[index,col].plot([start[0],end[0]],[start[1],end[1]],color='cyan')
                axes[index,col].set_xlim(lo[0],hi[0]);axes[index,col].set_ylim(hi[1],lo[1]);axes[index,col].set_title(edge['name']+' — '+title);axes[index,col].axis('off')
            for name,values in maps.items():
                score=probe_map(values,edge,meta,annotations['profile_half_width_native_px'],annotations['min_along_edge_coverage'])
                rows.append(dict(scene=label,edge=edge['name'],method=name,primary_silhouette_probe=edge['primary_silhouette_probe'],annotation_uncertainty_native_px=edge['annotation_uncertainty_native_px'],**score))
                if 'normalized_profile' in score:axes[index,2].plot(score['offsets_native_px'],score['normalized_profile'],label=name)
            axes[index,2].axvline(0,color='black',linestyle='--');axes[index,2].axvspan(-edge['annotation_uncertainty_native_px'],edge['annotation_uncertainty_native_px'],color='gray',alpha=.1);axes[index,2].set_ylim(-.5,1.5)
            axes[index,2].set_title('Locally normalized inverse-depth transition');axes[index,2].set_xlabel('Native pixels toward background');axes[index,2].legend(fontsize=7)
        fig.suptitle(label+' — approximate RGB silhouettes, not dense depth truth');fig.savefig(out/'boundary_profiles.png',dpi=130);plt.close(fig)
        for region in metrics['regions']:
            x0,y0,x1,y1=region['bbox']
            def sharpness(image):
                gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY).astype(np.float32)/255
                return float(np.abs(cv2.Laplacian(gray,cv2.CV_32F,ksize=3))[y0:y1,x0:x1].mean())
            before=sharpness(reference);after=sharpness(aif)
            image_detail.append(dict(scene=label,region=region['name'],reference_sharpness=before,aif_sharpness=after,ratio=after/max(before,1e-12),within_calibration_span=region['within_measured_calibration_span']))
    summary=[]
    for name in maps:
        subset=[r for r in rows if r['method']==name and r['primary_silhouette_probe']];valid=[r for r in subset if r['status']=='valid'];widths=[r['transition_width_native_px'] for r in valid if r['transition_width_native_px'] is not None]
        summary.append(dict(method=name,n_probes=len(subset),n_localizable=len(valid),n_with_width=len(widths),
                            median_absolute_offset_native_px=float(np.median([r['absolute_offset_native_px'] for r in valid])) if valid else None,
                            median_transition_width_native_px=float(np.median(widths)) if widths else None,
                            mean_sample_coverage=float(np.mean([r['sample_coverage'] for r in subset]))))
    paired=[]
    for scene in annotations['scenes']:
        for edge in scene['edges']:
            if not edge['primary_silhouette_probe']:continue
            pair={name:next(r for r in rows if r['scene']==scene['scene'] and r['edge']==edge['name'] and r['method']==name) for name in ['reference_neural','aif_neural']}
            if all(r['status']=='valid' for r in pair.values()):
                paired.append(dict(scene=scene['scene'],edge=edge['name'],reference_absolute_offset=pair['reference_neural']['absolute_offset_native_px'],aif_absolute_offset=pair['aif_neural']['absolute_offset_native_px'],reference_width=pair['reference_neural']['transition_width_native_px'],aif_width=pair['aif_neural']['transition_width_native_px']))
    result=dict(scope='Exploratory manually annotated RGB silhouette profiles; model/input controlled; no dense depth ground truth.',annotations=annotations,model_runs=model_runs,summary=summary,paired_neural_probes=paired,profiles=rows,image_detail=image_detail,
                limitations=['Manual visible-edge placement uncertainty approximately +/-5 native pixels.','Universal package fold is ambiguous and excluded from the primary silhouette summary; retained in profiles.',
                              'Missing depth across a silhouette can prevent a crossing; coverage is not silently filled.','All profiles use farther-increasing inverse depth, then local anchor normalization; no physical-scale accuracy inferred.','Selected probes on similar arrangements are not independent scene-level trials.'])
    (args.output/'results.json').write_text(json.dumps(json_safe(result),indent=2,allow_nan=False)+'\n');print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
