"""Fixed-grid regional focus anchors; no measured scene distances or LiDAR inputs."""
import numpy as np
from depth_alignment import fit_affine
from focus_calibration import curve_peak, depth_from_focus


CONFIG=dict(patch_size=64,min_prominence=.5,min_relative_texture=.05,
            max_neural_cv=.10,min_anchors=8,min_relative_neural_span=.10)


def extract_anchors(sharpness,lens,raw_inverse,model):
    if sharpness.shape[1:]!=raw_inverse.shape:raise ValueError('Sharpness/neural grids differ')
    size=CONFIG['patch_size'];h,w=raw_inverse.shape
    reference=float(np.percentile(sharpness.max(axis=0),99))+1e-8
    anchors=[];rejections={}
    for y in range(0,h-size+1,size):
        for x in range(0,w-size+1,size):
            curve=sharpness[:,y:y+size,x:x+size].mean(axis=(1,2))
            reason=None
            if curve.max()/reference<CONFIG['min_relative_texture']:reason='weak_texture'
            try:peak=curve_peak(lens,curve,CONFIG['min_prominence'])
            except ValueError:reason=reason or 'weak_or_unbracketed_peak'
            if reason:rejections[reason]=rejections.get(reason,0)+1;continue
            meters=float(depth_from_focus(np.array([peak['peak_lens_position']]),model)[0])
            if not np.isfinite(meters):reason='outside_calibration'
            patch=raw_inverse[y:y+size,x:x+size]
            if not np.isfinite(patch).all() or patch.mean()<=0:reason=reason or 'invalid_neural'
            elif patch.std()/patch.mean()>CONFIG['max_neural_cv']:reason=reason or 'mixed_neural_patch'
            if reason:rejections[reason]=rejections.get(reason,0)+1;continue
            anchors.append(dict(bbox=[x,y,x+size,y+size],focus_m=meters,
                neural_inverse=float(np.median(patch)),prominence=peak['prominence'],
                peak_lens_position=peak['peak_lens_position']))
    return anchors,rejections


def scale_with_anchors(raw_inverse,anchors):
    if len(anchors)<CONFIG['min_anchors']:raise ValueError('Too few regional anchors')
    src=np.array([a['neural_inverse'] for a in anchors]);dst=1/np.array([a['focus_m'] for a in anchors])
    if np.ptp(src)/np.median(src)<CONFIG['min_relative_neural_span']:raise ValueError('Insufficient neural anchor span')
    scale,shift=fit_affine(src,dst,np.array([a['prominence'] for a in anchors]))
    if not np.isfinite([scale,shift]).all() or scale<=0:raise ValueError('Invalid regional scale fit')
    inverse=scale*raw_inverse+shift;out=np.full(raw_inverse.shape,np.nan,np.float32)
    good=np.isfinite(inverse)&(inverse>0);out[good]=1/inverse[good]
    return out,dict(scale=float(scale),shift=float(shift),n_anchors=len(anchors))
