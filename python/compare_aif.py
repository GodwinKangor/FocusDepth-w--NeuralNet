"""Controlled reference-vs-AIF relative-depth ablation on the measured LiDAR sweeps.

Same registration, crop, model, resolution, regions and reference LiDAR mask for
both inputs. LiDAR and tape measurements never fit the predictions or fusion.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import cv2
import numpy as np
import yaml
from scipy.stats import spearmanr

import all_in_focus
import confidence
import depth_alignment
import evaluate
import focus_depth
import fusion
import neural_depth
import registration
from run_pipeline import load_images, load_manifest


def transform_regions(regions, native_size, resized_size, crop):
    nw, nh = native_size
    rw, rh = resized_size
    x0, y0, x1, y1 = crop
    result = []
    for region in regions:
        bx0, by0, bx1, by1 = region['bbox']
        box = [int(np.ceil(bx0*rw/nw))-x0, int(np.ceil(by0*rh/nh))-y0,
               int(np.floor(bx1*rw/nw))-x0, int(np.floor(by1*rh/nh))-y0]
        if not (0 <= box[0] < box[2] <= x1-x0 and 0 <= box[1] < box[3] <= y1-y0):
            raise ValueError(f"Region {region['name']} falls outside registration crop")
        result.append({**region, 'bbox_native': region['bbox'], 'bbox': box,
                       'coordinate_space': 'registered_cropped_rgb'})
    return result


def lidar_samples(folder, frame, native_size, resized_size, crop, maps, regions):
    from inspect_lidar_capture import load_depth
    metadata = json.loads((folder/frame['pair_metadata_file']).read_text())
    if metadata['orientation'] != 'native_sensor_unmirrored_rotation_0':
        raise ValueError('Unexpected RGB/depth orientation')
    if (metadata['intrinsics_reference_width'], metadata['intrinsics_reference_height']) != native_size:
        raise ValueError('Unexpected calibration reference grid')
    depth = load_depth(folder/frame['depth_file'], metadata)
    valid = np.isfinite(depth) & (depth > 0)
    # Exclude invalid samples and steep depth boundaries, identically for methods.
    safe = np.where(valid, depth, 0)
    local_range = cv2.dilate(safe, np.ones((3,3),np.uint8)) - cv2.erode(safe, np.ones((3,3),np.uint8))
    valid = cv2.erode(valid.astype(np.uint8), np.ones((3,3),np.uint8)).astype(bool) & (local_range < .05)
    yy, xx = np.indices(depth.shape, dtype=np.float32)
    rw,rh = resized_size
    x0,y0,x1,y1 = crop
    mx = (xx+.5)*rw/depth.shape[1]-.5-x0
    my = (yy+.5)*rh/depth.shape[0]-.5-y0
    valid &= (mx >= 0) & (mx <= x1-x0-1) & (my >= 0) & (my <= y1-y0-1)
    packages = np.zeros(depth.shape, bool)
    for region in regions:
        if region['name'] == 'wall':
            continue
        bx0, by0, bx1, by1 = region['bbox']
        packages |= (mx >= bx0) & (mx < bx1) & (my >= by0) & (my < by1)
    result = {}
    for name, prediction in maps.items():
        sampled = cv2.remap(prediction, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        mask = valid & np.isfinite(sampled)
        # All maps provided here increase with distance. This is rank agreement,
        # not a metric-depth accuracy calculation.
        result[name] = dict(spearman_vs_lidar=float(spearmanr(sampled[mask],depth[mask]).statistic),
                            n_depth_samples=int(mask.sum()),
                            spearman_packages_vs_lidar=float(spearmanr(sampled[mask & packages],depth[mask & packages]).statistic),
                            n_package_depth_samples=int((mask & packages).sum()))
    return result


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--measurements',type=Path,default=Path('data/annotations/lidar_capture_measurements.json'))
    parser.add_argument('--config',type=Path,default=Path('configs/default.yaml'))
    parser.add_argument('--output',type=Path,default=Path('results/aif_ablation'))
    args=parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    cfg=yaml.safe_load(args.config.read_text())
    record=json.loads(args.measurements.read_text())
    predictor=neural_depth.DepthPredictor(cfg['neural']['model'])
    summaries=[]
    for scene in record['scenes']:
        label=Path(scene['archive']).stem
        folder=Path(scene['local_capture_directory']);out=args.output/label
        out.mkdir(parents=True,exist_ok=True)
        logging.info('%s: registering stack',label)
        manifest,frames=load_manifest(folder,cfg['capture']['lens_position_field'])
        images=load_images(folder,frames,cfg['capture']['max_side'])
        ref=len(images)//2
        rcfg=cfg['registration']
        reg=registration.register_stack(images,ref,rcfg['motion'],rcfg['iterations'],float(rcfg['eps']),pyramid_levels=rcfg['pyramid_levels'])
        # Do not silently evaluate a composite with known failed registration links.
        if reg.failed:
            raise RuntimeError(f'{label}: registration failed for frames {reg.failed}')
        gray=np.stack([registration.to_gray_float(im) for im in reg.images])
        ocfg=cfg['optical']
        sharp=focus_depth.sharpness_stack(gray,ocfg['laplacian_ksize'],ocfg['smooth_sigma'])
        aif,labels=all_in_focus.compose(reg.images,sharp,ref)
        reference=reg.images[ref]
        cv2.imwrite(str(out/'reference_rgb.png'),reference)
        cv2.imwrite(str(out/'all_in_focus.png'),aif)
        np.save(out/'aif_source_indices.npy',labels)
        lens=np.array([f['_lens'] for f in frames],np.float32)
        optical=focus_depth.optical_depth(sharp,lens,ocfg['method'])
        ccfg=cfg['confidence']
        conf=confidence.optical_confidence(sharp,ccfg['measure'],ccfg['texture_gate'],ccfg['gate_low'],ccfg['gate_high'])
        np.save(out/'optical_depth.npy',optical);np.save(out/'optical_confidence.npy',conf)
        regions=transform_regions(scene['regions'],(manifest['image_width'],manifest['image_height']),
                                  (images[ref].shape[1],images[ref].shape[0]),reg.crop)
        maps={'optical':optical};run_info={};fits={}
        for name,image in [('reference',reference),('aif',aif)]:
            logging.info('%s: neural inference on %s',label,name)
            raw,info=predictor.predict(image)
            np.save(out/f'{name}_neural_raw.npy',raw)
            # Direct sign reversal preserves neural ordering without optical fit.
            maps[f'{name}_neural']=-raw
            acfg=cfg['alignment']
            aligned,fit=depth_alignment.align_neural_to_optical(raw,optical,conf,acfg['conf_threshold'],acfg['min_pixels'])
            fits[name]=fit.to_dict()
            if fit.scale < 0:
                maps[f'{name}_fixed_fusion']=fusion.fixed_fusion(optical,aligned,cfg['fusion']['fixed_weight'])
                maps[f'{name}_confidence_fusion']=fusion.confidence_fusion(optical,aligned,conf)
            else:
                logging.warning('Omitting %s fusion because optical alignment reverses neural ordering',name)
            run_info[name]=info
        for name,d in maps.items():np.save(out/f'{name}.npy',d)
        ordering={name:evaluate.pairwise_ordering(d,regions) for name,d in maps.items()}
        rank=lidar_samples(folder,frames[ref],(manifest['image_width'],manifest['image_height']),
                           (images[ref].shape[1],images[ref].shape[0]),reg.crop,maps,regions)
        detail=[]
        aif_gray=registration.to_gray_float(aif)
        reference_gray=registration.to_gray_float(reference)
        for region in regions:
            x0,y0,x1,y1=region['bbox']
            def sharpness(image):
                return float(np.abs(cv2.Laplacian(image,cv2.CV_32F,ksize=3))[y0:y1,x0:x1].mean())
            baseline=sharpness(reference_gray);composite=sharpness(aif_gray)
            detail.append(dict(name=region['name'],bbox=region['bbox'],reference_sharpness=baseline,
                               aif_sharpness=composite,aif_to_reference_sharpness=composite/max(baseline,1e-12)))
        result=dict(scene=label,reference_frame=frames[ref]['file'],regions=regions,
                    registration=dict(crop=list(reg.crop),failed=reg.failed,ecc_scores=reg.ecc_scores,warps=[w.tolist() for w in reg.warps]),
                    aif_settings=dict(label_window=5,feather_sigma=1.0),
                    model_runs=run_info,optical_alignment=fits,ordering=ordering,lidar_rank=rank,
                    region_sharpness=detail,config=cfg,
                    units='Neural raw: relative inverse depth; optical/fusion: lens-position units. No metric accuracy claim.',
                    scope='Three exploratory scenes; settings not tuned on these results. LiDAR excluded from inference and fusion.')
        (out/'metrics.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
        fig,axes=plt.subplots(1,5,figsize=(20,4))
        for ax,image,title in zip(axes[:2],[reference,aif],['Reference RGB','All-in-focus RGB']):
            ax.imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB));ax.set_title(title);ax.axis('off')
        # Common display limits across the two raw neural predictions.
        nn=np.concatenate([maps['reference_neural'].ravel(),maps['aif_neural'].ravel()]);lo,hi=np.percentile(nn,[1,99])
        for ax,name in zip(axes[2:4],['reference_neural','aif_neural']):
            ax.imshow(maps[name],vmin=lo,vmax=hi,cmap='viridis');ax.set_title(name+' (relative)');ax.axis('off')
        axes[4].imshow(labels,vmin=0,vmax=len(images)-1,cmap='turbo');axes[4].set_title('AIF source frame');axes[4].axis('off')
        fig.suptitle(label);fig.tight_layout();fig.savefig(out/'comparison.png',dpi=140);plt.close(fig)
        summaries.append(result)
        logging.info('%s: done; ordering %s',label,{k:v['accuracy'] for k,v in ordering.items()})
    (args.output/'summary.json').write_text(json.dumps(summaries,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
