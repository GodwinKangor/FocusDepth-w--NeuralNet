"""Qualitative boundary close-ups; image edges are guides, not depth ground truth."""
import json
from pathlib import Path

import cv2
import numpy as np


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    windows={'scene-20-40-65':[950,600,1500,1220],
             'scene-40-60-85':[900,0,1500,650],
             'scene-40-60-100':[1000,250,1500,700]}
    scenes=json.loads(Path('results/aif_ablation/summary.json').read_text())
    measurements=json.loads(Path('data/annotations/lidar_capture_measurements.json').read_text())
    output=Path('results/boundary_review');output.mkdir(parents=True,exist_ok=True)
    for scene in scenes:
        label=scene['scene'];source=Path('results/aif_ablation')/label
        record=next(r for r in measurements['scenes'] if Path(r['archive']).stem==label)
        raw=Path(record['local_capture_directory'])
        manifest=json.loads((raw/'manifest.json').read_text())
        frame=next(f for f in manifest['frames'] if f['file']==scene['reference_frame'])
        meta=json.loads((raw/frame['pair_metadata_file']).read_text())
        depth=np.fromfile(raw/frame['depth_file'],dtype='<f4').reshape(meta['depth_height'],meta['depth_width'])
        depth=np.where(np.isfinite(depth)&(depth>0),depth,np.nan)
        x0,y0,x1,y1=scene['registration']['crop']
        scale=min(1,1600/max(manifest['image_width'],manifest['image_height']))
        full=(round(manifest['image_width']*scale),round(manifest['image_height']*scale))
        native_window=windows[label]
        a,b,c,d=[round(v*scale)-o for v,o in zip(native_window,[x0,y0,x0,y0])]
        a=max(0,a);b=max(0,b);c=min(x1-x0,c);d=min(y1-y0,d)
        depth=cv2.resize(depth,full,interpolation=cv2.INTER_NEAREST)[y0:y1,x0:x1]
        reference=cv2.cvtColor(cv2.imread(str(source/'reference_rgb.png')),cv2.COLOR_BGR2RGB)
        aif=cv2.cvtColor(cv2.imread(str(source/'all_in_focus.png')),cv2.COLOR_BGR2RGB)
        optical=np.load(source/'optical_depth.npy');conf=np.load(source/'optical_confidence.npy')
        neural=np.load(source/'aif_neural.npy')
        panels=[(reference,None,None,None,'Reference RGB'),(aif,None,None,None,'AIF RGB'),
                (depth,'viridis',.15,1.35,'LiDAR meters: nearest display only'),
                (optical,'viridis',0,1,'Optical focus coordinate, not meters'),
                (conf,'gray',0,1,'Optical confidence'),
                (neural,'viridis',*np.percentile(neural,[1,99]),'AIF neural relative depth')]
        fig,axes=plt.subplots(2,3,figsize=(12,9))
        for ax,(image,cmap,lo,hi,title) in zip(axes.ravel(),panels):
            patch=image[b:d,a:c]
            patch=cv2.rotate(patch,cv2.ROTATE_90_CLOCKWISE)
            ax.imshow(patch,cmap=cmap,vmin=lo,vmax=hi,interpolation='nearest');ax.set_title(title);ax.axis('off')
        fig.suptitle(label+' — qualitative boundary review; no boundary accuracy claim')
        fig.tight_layout();fig.savefig(output/(label+'.png'),dpi=140);plt.close(fig)
    (output/'windows.json').write_text(json.dumps(dict(native_rgb_windows=windows,
        scope='Qualitative only. No depth boundary ground truth or metric calibration is inferred.'),indent=2)+'\n')


if __name__=='__main__':main()
