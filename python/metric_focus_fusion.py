"""Optical-assisted metric scaling and fusion; no LiDAR or tape inputs."""
import numpy as np
from depth_alignment import fit_affine


def scale_relative_with_focus(raw_inverse, optical_meters, confidence, min_pixels=500, threshold=.5):
    if raw_inverse.shape!=optical_meters.shape or confidence.shape!=raw_inverse.shape:
        raise ValueError('Depth/confidence grids differ')
    valid=np.isfinite(raw_inverse)&np.isfinite(optical_meters)&(optical_meters>0)&np.isfinite(confidence)&(confidence>=threshold)
    idx=np.flatnonzero(valid)
    if len(idx)<min_pixels:raise ValueError('Too few calibrated optical anchors for neural scaling')
    total=len(idx)
    if len(idx)>200000:idx=np.random.default_rng(0).choice(idx,200000,replace=False)
    src=raw_inverse.ravel()[idx].astype(float);dst=1/optical_meters.ravel()[idx]
    if np.ptp(src)<1e-8:raise ValueError('Relative prediction lacks anchor variation')
    scale,shift=fit_affine(src,dst,confidence.ravel()[idx])
    if scale<=0:raise ValueError('Optical fit would reverse relative neural ordering')
    inverse=scale*raw_inverse+shift
    meters=np.full(inverse.shape,np.nan,np.float32)
    good=np.isfinite(inverse)&(inverse>0);meters[good]=1/inverse[good]
    residual=dst-(scale*src+shift)
    return meters,dict(scale=scale,shift=shift,fit_representation='inverse_meters',n_optical_anchors=total,
                       anchor_residual_mad_inverse_m=float(np.median(abs(residual-np.median(residual)))),
                       source='Calibrated focus only; no LiDAR or measured scene distances in fit')


def inverse_depth_fusion(optical, neural, confidence):
    if optical.shape!=neural.shape or optical.shape!=confidence.shape:raise ValueError('Fusion grids differ')
    opt=np.isfinite(optical)&(optical>0);nn=np.isfinite(neural)&(neural>0)
    w=np.where(opt,np.clip(np.nan_to_num(confidence,nan=0),0,1),0)
    w=np.where(opt&~nn,1,w)
    uopt=np.zeros(optical.shape,np.float32);unn=np.zeros(neural.shape,np.float32)
    np.divide(1,optical,out=uopt,where=opt);np.divide(1,neural,out=unn,where=nn)
    inverse=w*uopt+(1-w)*unn
    result=np.full(optical.shape,np.nan,np.float32)
    valid=(opt|nn)&np.isfinite(inverse)&(inverse>0);result[valid]=1/inverse[valid]
    return result
