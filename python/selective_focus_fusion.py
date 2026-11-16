"""Fixed development candidates for conservative optical influence.

No tape labels, LiDAR, manual boundary annotations or parameter search enter the
rules. Protecting neural structure can also preserve a neural mistake.
"""
import cv2
import numpy as np
from metric_focus_fusion import inverse_depth_fusion

PARAMETERS=dict(min_confidence=.5,max_relative_inverse_disagreement=.15,
                neural_edge_window=5,max_neural_inverse_ratio=1.10,
                support_window=7,min_local_support=.5,max_local_inverse_cv=.10)
MODES=['agreement','agreement_edge','agreement_edge_support']


def selective_fusion(optical,neural,confidence,mode):
    if mode not in MODES:raise ValueError('Unknown selective fusion mode')
    if optical.shape!=neural.shape or neural.shape!=confidence.shape or optical.ndim!=2:
        raise ValueError('Expected matching 2D maps')
    cfg=PARAMETERS
    opt=np.isfinite(optical)&(optical>0)&np.isfinite(confidence)&(confidence>=cfg['min_confidence'])
    nn=np.isfinite(neural)&(neural>0)
    uopt=np.zeros(optical.shape,np.float64);unn=np.zeros(neural.shape,np.float64)
    np.divide(1,optical,out=uopt,where=opt);np.divide(1,neural,out=unn,where=nn)
    difference=np.abs(uopt-unn)/np.maximum(unn,1e-12)
    accepted=opt&nn&(difference<=cfg['max_relative_inverse_disagreement'])
    edge=np.zeros(optical.shape,bool);support=np.ones(optical.shape,bool)
    if mode in ['agreement_edge','agreement_edge_support']:
        kernel=np.ones((cfg['neural_edge_window'],cfg['neural_edge_window']),np.uint8)
        hi=cv2.dilate(unn,kernel);lo=cv2.erode(unn,kernel)
        edge=(lo<=0)|(hi>cfg['max_neural_inverse_ratio']*lo)
        accepted &= ~edge
    if mode=='agreement_edge_support':
        size=(cfg['support_window'],cfg['support_window'])
        fraction=cv2.boxFilter(opt.astype(np.float64),-1,size,normalize=True,borderType=cv2.BORDER_CONSTANT)
        mean=cv2.boxFilter(uopt,-1,size,normalize=True,borderType=cv2.BORDER_CONSTANT)/np.maximum(fraction,1e-12)
        mean_sq=cv2.boxFilter(uopt*uopt,-1,size,normalize=True,borderType=cv2.BORDER_CONSTANT)/np.maximum(fraction,1e-12)
        cv=np.sqrt(np.maximum(mean_sq-mean*mean,0))/np.maximum(mean,1e-12)
        support=(fraction>=cfg['min_local_support'])&(cv<=cfg['max_local_inverse_cv'])
        accepted &= support
    effective_optical=np.where(accepted,optical,np.nan)
    result=inverse_depth_fusion(effective_optical,neural,confidence)
    weights=np.where(accepted,np.clip(confidence,0,1),0).astype(np.float32)
    changed=np.isfinite(result)&nn&(np.abs(result-neural)>1e-6)
    stats=dict(mode=mode,parameters=cfg,initial_optical_fraction=float(opt.mean()),
               accepted_optical_fraction=float(accepted.mean()),changed_neural_fraction=float(changed.mean()),
               neural_edge_guard_fraction=float(edge.mean()),local_support_pass_fraction=float(support.mean()),
               output_coverage=float((np.isfinite(result)&(result>0)).mean()))
    return result,weights,stats
