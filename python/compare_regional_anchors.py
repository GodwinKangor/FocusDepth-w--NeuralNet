"""Exploratory fixed regional-anchor scaling versus cached pixel-anchor scaling."""
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np
import focus_depth
import registration
from focus_calibration import signature
from run_pipeline import load_images,load_manifest
from evaluate_metric_scenes import region_statistics
from regional_focus_anchors import CONFIG,extract_anchors,scale_with_anchors


def main():
    root=Path('results/multiscene_metric');output=Path('results/regional_anchor_study');output.mkdir(parents=True,exist_ok=True)
    modelpath=Path('results/focus_calibration/model.json');raw=modelpath.read_bytes();model=json.loads(raw)
    annotations=json.loads(Path('data/annotations/multiscene_20261115.json').read_text())
    # Scene labels are used only after automatic anchor extraction and scaling.
    captures={s['scene']:Path(s['capture_directory']) for s in annotations['scenes']}
    scenes=json.loads((root/'summary.json').read_text())['scenes'];rows=[];runs=[]
    for scene in scenes:
        name=scene['scene'];folder=root/name;meta=json.loads((folder/'preprocessing.json').read_text())
        capture=captures[name];manifest,frames=load_manifest(capture,'lens_position_actual')
        if signature(manifest)!=model['signature'] or meta['registration_failed']:raise ValueError('Incompatible scene')
        if hashlib.sha256((capture/'manifest.json').read_bytes()).hexdigest()!=meta['source_manifest_sha256']:raise ValueError('Source changed')
        if hashlib.sha256(raw).hexdigest()!=meta['calibration_model_sha256']:raise ValueError('Calibration changed')
        images=load_images(capture,frames,1600);h,w=images[0].shape[:2];x,y,ex,ey=meta['crop']
        if len(images)!=len(meta['warps']):raise ValueError('Warp/frame mismatch')
        gray=np.stack([registration.to_gray_float(cv2.warpAffine(im,np.array(warp,np.float32),(w,h),
            flags=cv2.INTER_LINEAR+cv2.WARP_INVERSE_MAP,borderMode=cv2.BORDER_CONSTANT)[y:ey,x:ex])
            for im,warp in zip(images,meta['warps'])])
        sharp=focus_depth.sharpness_stack(gray,3,3.);neural=np.load(folder/'relative_aif_raw.npy')
        anchors,rejections=extract_anchors(sharp,np.array([f['_lens'] for f in frames]),neural,model)
        run=dict(scene=name,anchors=anchors,rejections=rejections)
        maps={'pixel_anchor':np.load(folder/'focus_scaled_relative_aif_meters.npy')}
        try:maps['regional_anchor'],run['fit']=scale_with_anchors(neural,anchors)
        except ValueError as error:run['failure']=str(error)
        out=output/name;out.mkdir(parents=True,exist_ok=True)
        for method,values in maps.items():
            if method=='regional_anchor':np.save(out/'regional_scaled_meters.npy',values)
            for region in scene['regions']:rows.append(dict(scene=name,method=method,**region_statistics(values,region)))
        run['coverage']={k:float(np.isfinite(v).mean()) for k,v in maps.items()}
        cv2.imwrite(str(out/'anchor_locations.png'),draw_anchors(cv2.imread(str(folder/'all_in_focus.png')),anchors))
        runs.append(run);print(name,len(anchors),run.get('fit',run.get('failure')),flush=True)
    summary=[]
    for method in ['pixel_anchor','regional_anchor']:
        selected=[r for r in rows if r['method']==method and r['score_eligible']]
        summary.append(dict(method=method,n_eligible=len(selected),expected=11,
            mae_cm=float(np.mean([abs(r['signed_residual_cm']) for r in selected])) if len(selected)==11 else None))
    result=dict(scope='Inspected development scenes; fixed settings, no threshold search; labels evaluate only.',
                config=CONFIG,baseline_sha256=hashlib.sha256(raw).hexdigest(),summary=summary,runs=runs,regions=rows)
    (output/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    assert modelpath.read_bytes()==raw
    print(json.dumps(summary,indent=2))


def draw_anchors(image,anchors):
    for anchor in anchors:
        x,y,ex,ey=anchor['bbox'];cv2.rectangle(image,(x,y),(ex,ey),(0,255,0),1)
    return image


if __name__=='__main__':main()
