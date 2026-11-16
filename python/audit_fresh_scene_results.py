"""Audit all predefined fresh regions without inference or annotation changes."""
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from compare_aif import transform_regions
from evaluate_metric_scenes import region_statistics

ROOT=Path('results/fresh_20261116')
ANNOTATIONS=Path('data/annotations/fresh_20261116.json')
METHODS=['calibrated_focus','focus_scaled_relative_aif','focus_neural_fusion','direct_lidar','unadjusted_metric_aif']


def summarize(rows,expected):
    eligible=[r for r in rows if r['score_eligible']]
    return dict(expected_regions=expected,eligible_regions=len(eligible),
        complete_set_mae_cm=float(np.mean([abs(r['signed_residual_cm']) for r in eligible])) if len(eligible)==expected and expected else None,
        eligible_subset_mae_cm=float(np.mean([abs(r['signed_residual_cm']) for r in eligible])) if eligible else None)


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    annotation_raw=ANNOTATIONS.read_bytes();annotations=json.loads(annotation_raw)
    summary_raw=(ROOT/'summary.json').read_bytes();saved=json.loads(summary_raw)
    output=Path('results/fresh_20261116_audit');output.mkdir(parents=True,exist_ok=True)
    all_rows=[];pairs=[];rgb=[]
    for scene in annotations['scenes']:
        name=scene['scene'];folder=ROOT/name;meta=json.loads((folder/'preprocessing.json').read_text())
        capture=Path(scene['capture_directory'])
        if hashlib.sha256((capture/'manifest.json').read_bytes()).hexdigest()!=meta['source_manifest_sha256']:raise ValueError('Source changed')
        if meta['registration_failed']:raise ValueError('Failed registration')
        regions=transform_regions(scene['regions'],meta['native_size'],meta['resized_size'],meta['crop'])
        stored=next(s for s in saved['scenes'] if s['scene']==name)
        maps={method:np.load(folder/(method+'_meters.npy')) for method in METHODS}
        for method,values in maps.items():
            for region in regions:
                row=dict(scene=name,method=method,**region_statistics(values,region))
                original=next(r for r in stored['region_results'] if r['method']==method and r['name']==region['name'])
                for key in ['coverage','median_cm','signed_residual_cm','score_eligible']:
                    if row[key] is None:assert original[key] is None
                    else:assert np.isclose(row[key],original[key]),(name,method,key)
                all_rows.append(row)
        for other in ['focus_neural_fusion','direct_lidar']:
            first=maps['focus_scaled_relative_aif'];second=maps[other]
            common=np.isfinite(first)&(first>0)&np.isfinite(second)&(second>0)
            for method,values in [('focus_scaled_relative_aif',first),(other,second)]:
                for region in regions:pairs.append(dict(scene=name,pair='focus_scaled_relative_aif vs '+other,method=method,
                    **region_statistics(np.where(common,values,np.nan),region)))
        if 'multi_smooth' in name:
            image=cv2.imread(str(capture/meta['reference_file']),cv2.IMREAD_COLOR|cv2.IMREAD_IGNORE_ORIENTATION)
            if image is None:raise ValueError('Missing native reference')
            rgb.append((scene,image))
    expected=sum(len(s['regions']) for s in annotations['scenes'])
    full={m:summarize([r for r in all_rows if r['method']==m],expected) for m in METHODS}
    # Retain the old exclusion only as a labeled sensitivity analysis.
    restricted=[r for r in all_rows if not ('multi_smooth' in r['scene'] and r['name']=='smooth_box_70')]
    sensitivity={m:summarize([r for r in restricted if r['method']==m],expected-3) for m in METHODS}
    output_data=dict(scope='Full predefined nominal-label audit; uncertain labels retained. No tuning, relabeling or inference.',
        annotations_sha256=hashlib.sha256(annotation_raw).hexdigest(),source_summary_sha256=hashlib.sha256(summary_raw).hexdigest(),
        full_set=full,restricted_sensitivity_excluding_three_unresolved_far_patches=sensitivity,
        original_rows_reproduced=True,regions=all_rows,pairwise_common_support_regions=pairs)
    (output/'audit.json').write_text(json.dumps(output_data,indent=2,allow_nan=False)+'\n')
    # RGB-only evidence: no depth values are used to draw or rename boxes.
    fig,axes=plt.subplots(len(rgb),2,figsize=(15,14),layout='constrained')
    for index,(scene,image) in enumerate(rgb):
        for ax in axes[index]:ax.imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
        for number,region in enumerate(scene['regions'],1):
            x,y,ex,ey=region['bbox']
            for ax in axes[index]:
                ax.add_patch(Rectangle((x,y),ex-x,ey-y,fill=False,edgecolor='red',linewidth=2))
                ax.text(x,y-8,f'{number}: nominal {region["distance_m"]*100:.0f} cm',color='white',fontsize=10,
                        bbox=dict(facecolor='black',alpha=.7,pad=2),clip_on=True)
        axes[index,0].set_title(scene['scene']+' · native reference RGB');axes[index,0].axis('off')
        far=next(r for r in scene['regions'] if r['name']=='smooth_box_70');x,y,ex,ey=far['bbox']
        axes[index,1].set_xlim(x-130,ex+130);axes[index,1].set_ylim(ey+130,y-130)
        axes[index,1].set_title('Far-patch context · nominal labels remain provisional')
    fig.savefig(output/'rgb_patch_audit.png',dpi=140);plt.close(fig)
    assert ANNOTATIONS.read_bytes()==annotation_raw and (ROOT/'summary.json').read_bytes()==summary_raw
    print(json.dumps(dict(full_set=full,restricted_sensitivity=sensitivity),indent=2))


if __name__=='__main__':main()
