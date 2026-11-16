"""Prepare a registered scene and calibrated optical maps without metric extrapolation."""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

import all_in_focus
import confidence
import focus_depth
import registration
from focus_calibration import signature, depth_from_focus
from run_pipeline import load_manifest, load_images


def prepare(folder: Path, output: Path, calibration_path: Path):
    raw=calibration_path.read_bytes();model=json.loads(raw)
    manifest,frames=load_manifest(folder,'lens_position_actual')
    if manifest.get('capture_purpose')!='scene':raise ValueError('Expected Scene capture')
    if signature(manifest)!=model['signature']:raise ValueError('Scene camera/format does not match calibration')
    output.mkdir(parents=True,exist_ok=True)
    images=load_images(folder,frames,1600);reference=len(images)//2
    reg=registration.register_stack(images,reference,motion='affine',iterations=200,eps=1e-6,pyramid_levels=3)
    info=dict(scene=folder.name,reference_index=reference,reference_file=frames[reference]['file'],
              native_size=[manifest['image_width'],manifest['image_height']],resized_size=[images[reference].shape[1],images[reference].shape[0]],
              crop=list(reg.crop),registration_failed=reg.failed,ecc_scores=reg.ecc_scores,
              warps=[w.tolist() for w in reg.warps],lens_positions=[f['_lens'] for f in frames],
              source_manifest_sha256=hashlib.sha256((folder/'manifest.json').read_bytes()).hexdigest(),
              calibration_model_sha256=hashlib.sha256(raw).hexdigest(),confidence_threshold=.5)
    if reg.failed:
        (output/'preprocessing.json').write_text(json.dumps(info,indent=2)+'\n')
        raise ValueError(f'Registration failed for frames {reg.failed}; no calibrated scene output')
    gray=np.stack([registration.to_gray_float(im) for im in reg.images])
    sharp=focus_depth.sharpness_stack(gray,3,3.)
    focus=focus_depth.optical_depth(sharp,np.array(info['lens_positions']),method='peak')
    conf=confidence.optical_confidence(sharp,'prominence',True,.05,.25)
    meters=depth_from_focus(focus,model)
    optical=np.where(conf>=.5,meters,np.nan)
    aif,labels=all_in_focus.compose(reg.images,sharp,reference)
    for name,arr in [('focus_coordinate',focus),('optical_confidence',conf),('optical_meters_unmasked',meters),('optical_meters',optical),('aif_source_indices',labels)]:
        np.save(output/(name+'.npy'),arr)
    cv2.imwrite(str(output/'reference_rgb.png'),reg.images[reference]);cv2.imwrite(str(output/'all_in_focus.png'),aif)
    info.update(calibrated_range_fraction=float(np.isfinite(meters).mean()),confident_metric_coverage=float(np.isfinite(optical).mean()))
    (output/'preprocessing.json').write_text(json.dumps(info,indent=2)+'\n')
    return info


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('capture',type=Path);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--calibration',type=Path,default=Path('results/focus_calibration/model.json'))
    args=parser.parse_args();print(json.dumps(prepare(args.capture,args.output,args.calibration),indent=2))
