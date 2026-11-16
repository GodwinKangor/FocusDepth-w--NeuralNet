"""LiDAR-assisted metric calibration on a fixed tabletop patch, evaluated elsewhere.

This is a new sensor-assisted method. Tape distances never fit the calibration.
LiDAR used as input cannot also be called independent ground truth for this method.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

from compare_metric_aif import region_result
from depth_alignment import fit_affine
from inspect_lidar_capture import load_depth
from evaluate_lidar_regions import depth_bbox, summarize_patch

# Chosen from RGB geometry, before inspecting anchored residuals. Native RGB grid.
ANCHOR_BBOX = [1480, 450, 1770, 1150]


def anchor_samples(prediction, depth, native_size, resized_size, crop, regions, anchor_bbox):
    nw, nh = native_size
    rw, rh = resized_size
    dh, dw = depth.shape
    if nw*dh != nh*dw:
        raise ValueError('Native depth/RGB aspect ratio mismatch')
    yy,xx = np.indices(depth.shape,dtype=np.float32)
    native_x=(xx+.5)*nw/dw-.5
    native_y=(yy+.5)*nh/dh-.5
    mx=(xx+.5)*rw/dw-.5-crop[0]
    my=(yy+.5)*rh/dh-.5-crop[1]
    valid=np.isfinite(depth)&(depth>0)
    safe=np.where(valid,depth,0)
    local_range=cv2.dilate(safe,np.ones((3,3),np.uint8))-cv2.erode(safe,np.ones((3,3),np.uint8))
    valid=cv2.erode(valid.astype(np.uint8),np.ones((3,3),np.uint8)).astype(bool)&(local_range<.05)
    x0,y0,x1,y1=anchor_bbox
    valid &= (native_x>=x0)&(native_x<x1)&(native_y>=y0)&(native_y<y1)
    valid &= (mx>=0)&(mx<=prediction.shape[1]-1)&(my>=0)&(my<=prediction.shape[0]-1)
    # Exclude all evaluation regions plus a four-depth-pixel margin in native RGB.
    for region in regions:
        bx0,by0,bx1,by1=region['bbox_native']
        valid &= ~((native_x>=bx0-24)&(native_x<bx1+24)&(native_y>=by0-24)&(native_y<by1+24))
    sampled=cv2.remap(prediction,mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
    valid &= np.isfinite(sampled)&(sampled>0)
    if valid.sum()<100:
        raise ValueError('Insufficient disjoint tabletop anchors')
    return sampled[valid].astype(np.float64),depth[valid].astype(np.float64),valid


def calibrate(prediction, source, target, method):
    if not (np.isfinite(source).all() and np.isfinite(target).all() and (source>0).all() and (target>0).all()):
        raise ValueError('Anchors must be positive finite depths')
    if method=='scale':
        scale=float(np.median(target/source))
        return (prediction*scale).astype(np.float32),dict(scale=scale,representation='meters',shift=0)
    if method!='inverse_affine':raise ValueError('Unknown calibration method')
    inverse=1/source
    if np.ptp(inverse)<1e-6:raise ValueError('Insufficient anchor depth variation')
    scale,shift=fit_affine(inverse,1/target)
    if scale<=0:raise ValueError('Inverse-depth calibration would reverse depth ordering')
    with np.errstate(divide='ignore', invalid='ignore'):
        calibrated=scale/prediction+shift
    output=np.full(prediction.shape,np.nan,np.float32)
    valid=np.isfinite(prediction)&(prediction>0)&np.isfinite(calibrated)&(calibrated>0)
    output[valid]=1/calibrated[valid]
    return output,dict(scale=scale,shift=shift,representation='inverse_meters',
                       invalid_fraction=float(1-valid.mean()))


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=Path('results/metric_aif'))
    parser.add_argument('--shared',type=Path,default=Path('results/aif_ablation'))
    parser.add_argument('--measurements',type=Path,default=Path('data/annotations/lidar_capture_measurements.json'))
    parser.add_argument('--output',type=Path,default=Path('results/lidar_anchored'))
    args=parser.parse_args()
    measurements=json.loads(args.measurements.read_text())
    metric_source=json.loads((args.source/'region_results.json').read_text())
    shared=json.loads((args.shared/'summary.json').read_text())
    args.output.mkdir(parents=True,exist_ok=True)
    rows=[];fits=[]
    for scene in shared:
        label=scene['scene'];record=next(s for s in measurements['scenes'] if Path(s['archive']).stem==label)
        folder=Path(record['local_capture_directory']);manifest=json.loads((folder/'manifest.json').read_text())
        frame=next(f for f in manifest['frames'] if f['file']==scene['reference_frame'])
        meta=json.loads((folder/frame['pair_metadata_file']).read_text())
        if meta['orientation']!='native_sensor_unmirrored_rotation_0' or manifest['geometric_distortion_correction']:
            raise ValueError('Expected matching native unrectified buffers')
        if meta['depth_accuracy']!='absolute':
            raise ValueError('Expected absolute LiDAR depth anchors')
        depth=load_depth(folder/frame['depth_file'],meta)
        # Reconstruct the resized full reference dimensions from the shared config.
        scale=min(1,scene['config']['capture']['max_side']/max(manifest['image_width'],manifest['image_height']))
        full=(round(manifest['image_width']*scale),round(manifest['image_height']*scale))
        native=(manifest['image_width'],manifest['image_height']);crop=scene['registration']['crop']
        out=args.output/label;out.mkdir(parents=True,exist_ok=True)
        for region in scene['regions']:
            bbox=depth_bbox(region['bbox_native'],native,(depth.shape[1],depth.shape[0]))
            stats=summarize_patch(depth,bbox)
            residual=(stats['median_m']-region['distance_m'])*100
            rows.append(dict(scene=label,input='sensor',method='direct_lidar',object=region['name'],
                             nominal_cm=region['distance_m']*100,predicted_cm=stats['median_m']*100,
                             signed_residual_cm=residual,absolute_residual_cm=abs(residual),
                             valid_fraction=stats['valid_fraction']))
        for input_name in ['reference','aif']:
            prediction=np.load(args.source/label/f'upright_{input_name}_meters.npy')
            if prediction.shape!=(crop[3]-crop[1],crop[2]-crop[0]):raise ValueError('Prediction crop mismatch')
            source,target,mask=anchor_samples(prediction,depth,native,full,crop,scene['regions'],ANCHOR_BBOX)
            np.save(out/f'{input_name}_anchor_mask.npy',mask)
            variants={'unadjusted':prediction}
            for method in ['scale','inverse_affine']:
                anchored,params=calibrate(prediction,source,target,method)
                params.update(scene=label,input=input_name,method=method,n_anchors=len(source),
                              anchor_lidar_range_m=[float(target.min()),float(target.max())],
                              anchor_prediction_range_m=[float(source.min()),float(source.max())])
                fits.append(params);variants[method]=anchored
                np.save(out/f'{input_name}_{method}_meters.npy',anchored)
            for method,values in variants.items():
                for region in scene['regions']:
                    x0,y0,x1,y1=region['bbox'];patch=values[y0:y1,x0:x1]
                    if np.any(np.isfinite(patch)&(patch>0)):
                        result=region_result(values,region)
                    else:
                        result=dict(object=region['name'],nominal_cm=region['distance_m']*100,
                                    predicted_cm=None,signed_residual_cm=None,absolute_residual_cm=None,valid_fraction=0)
                    result.update(scene=label,input=input_name,method=method)
                    rows.append(result)
        image=cv2.cvtColor(cv2.imread(str(args.shared/label/'reference_rgb.png')),cv2.COLOR_BGR2RGB)
        fig,axes=plt.subplots(1,2,figsize=(10,4))
        axes[0].imshow(image)
        x0,y0,x1,y1=ANCHOR_BBOX
        box=[x0*scale-crop[0],y0*scale-crop[1],(x1-x0)*scale,(y1-y0)*scale]
        axes[0].add_patch(Rectangle(box[:2],box[2],box[3],fill=False,edgecolor='cyan',linewidth=2))
        for region in scene['regions']:
            a,b,c,d=region['bbox'];axes[0].add_patch(Rectangle((a,b),c-a,d-b,fill=False,edgecolor='red'))
        axes[0].set_title('Cyan: anchors; red: evaluated regions');axes[0].axis('off')
        axes[1].imshow(depth,vmin=.15,vmax=1.35,cmap='viridis')
        axes[1].contour(mask,levels=[.5],colors=['cyan']);axes[1].set_title('Native LiDAR and retained anchors');axes[1].axis('off')
        fig.suptitle(label);fig.tight_layout();fig.savefig(out/'anchor_regions.png',dpi=140);plt.close(fig)
    summary=[]
    for input_name in ['reference','aif','sensor']:
        for method in (['direct_lidar'] if input_name=='sensor' else ['unadjusted','scale','inverse_affine']):
            subset=[r for r in rows if r['input']==input_name and r['method']==method and r['object']!='wall']
            complete=[r for r in subset if r['valid_fraction']==1]
            residual=np.array([r['signed_residual_cm'] for r in complete])
            summary.append(dict(input=input_name,method=method,n_package_regions=len(subset),
                                n_fully_valid_regions=len(complete),
                                mae_cm=float(np.abs(residual).mean()) if len(complete)==len(subset) else None,
                                fully_valid_subset_mae_cm=float(np.abs(residual).mean()) if len(complete) else None))
    result=dict(scope='LiDAR-assisted per-scene calibration; tape measurements used only for provisional evaluation.',
                anchor_bbox_native=ANCHOR_BBOX,anchor_exclusion_margin_native_px=24,
                measurement_notes=measurements.get('uncertainty_description'),
                tape_condition=measurements.get('tape_condition'),
                total_uncertainty_status=measurements.get('object_distance_uncertainty_status'),
                model_source=metric_source['model'],
                model_runs=[r['model'] for r in metric_source['runs'] if r['orientation']=='upright'],
                summary=summary,fits=fits,regions=rows)
    (args.output/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    fields=list(dict.fromkeys(key for row in rows for key in row))
    with (args.output/'region_results.csv').open('w') as file:
        writer=csv.DictWriter(file,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
