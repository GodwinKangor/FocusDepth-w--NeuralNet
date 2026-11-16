"""Decompose existing optical confidence on calibration/validation target ROIs.

Exploratory diagnostics only: no threshold selection, refitting, or default changes.
Cached preprocessing warps are reused after checking manifest identity.
"""
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

import confidence
import focus_depth
import registration
from compare_focus_preprocessing import roi_box
from focus_calibration import depth_from_focus, signature
from run_pipeline import load_images, load_manifest


def summarize(sharpness, lens, box, model, nominal):
    prominence=confidence.peak_prominence(sharpness)
    texture=confidence.texture_gate(sharpness)
    x0,y0,x1,y1=box
    focus=focus_depth.optical_depth(sharpness[:,y0:y1,x0:x1],lens)
    meters=depth_from_focus(focus,model)
    p=prominence[y0:y1,x0:x1];t=texture[y0:y1,x0:x1]
    masks={'all':np.ones(p.shape,bool),'prominence_only':p>=.5,
           'texture_only':t>=.5,'combined':p*t>=.5,'rejected_combined':p*t<.5}
    groups={}
    for name,mask in masks.items():
        valid=mask&np.isfinite(meters)
        residual=np.abs(meters[valid]-nominal)*100
        groups[name]=dict(pixel_fraction=float(mask.mean()),metric_pixel_fraction=float(valid.mean()),
            median_depth_m=float(np.median(meters[valid])) if valid.any() else None,
            median_absolute_pixel_residual_cm=float(np.median(residual)) if valid.any() else None)
    return dict(groups=groups,prominence_quantiles=np.quantile(p,[.1,.5,.9]).tolist(),
                texture_quantiles=np.quantile(t,[.1,.5,.9]).tolist(),
                endpoint_fraction=float(((focus==lens[0])|(focus==lens[-1])).mean()))


def main():
    root=Path('results/focus_preprocessing_study')
    baseline_path=Path('results/focus_calibration/model.json');raw=baseline_path.read_bytes();model=json.loads(raw)
    rows=[]
    for path in sorted(root.glob('*/descriptors.json')):
        payload=json.loads(path.read_text());base=payload['observations']['native_roi'];folder=Path(base['folder'])
        if hashlib.sha256((folder/'manifest.json').read_bytes()).hexdigest()!=base['manifest_sha256']:
            raise ValueError('Manifest changed')
        manifest,frames=load_manifest(folder,'lens_position_actual')
        if signature(manifest)!=base['signature']:raise ValueError('Camera signature changed')
        if payload['registration']['failed']:raise ValueError('Registration failed')
        images=load_images(folder,frames,1600);h,w=images[0].shape[:2]
        crop=payload['registration']['crop'];cx,cy,ex,ey=crop
        warps=payload['registration']['warps']
        if len(warps)!=len(images):raise ValueError('Warp/frame count mismatch')
        gray=np.stack([registration.to_gray_float(cv2.warpAffine(im,np.array(warp,np.float32),(w,h),
            flags=cv2.INTER_LINEAR+cv2.WARP_INVERSE_MAP,borderMode=cv2.BORDER_CONSTANT)[cy:ey,cx:ex])
            for im,warp in zip(images,warps)])
        sharp=focus_depth.sharpness_stack(gray,3,3.)
        box=roi_box(model['roi'],(manifest['image_width'],manifest['image_height']),(w,h),crop)
        result=summarize(sharp,np.array([f['_lens'] for f in frames]),box,model,base['distance_m'])
        rows.append(dict(capture=folder.name,role=manifest['capture_purpose'],nominal_m=base['distance_m'],
                         regional_peak=payload['observations']['smoothed_roi']['peak_lens_position'],**result))
        print(folder.name,json.dumps(result['groups']),flush=True)
    output=Path('results/focus_confidence_diagnostic');output.mkdir(parents=True,exist_ok=True)
    (output/'results.json').write_text(json.dumps(dict(
        scope='Inspected development captures; pixel residuals use an approximate uniform target distance, not dense ground truth.',
        baseline_sha256=hashlib.sha256(raw).hexdigest(),rows=rows),indent=2,allow_nan=False)+'\n')
    assert baseline_path.read_bytes()==raw


if __name__=='__main__':main()
