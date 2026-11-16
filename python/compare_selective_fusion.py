"""Development-only selective fusion ablation, retaining all frozen baseline maps."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from compare_boundary_detail import probe_map,far_inverse_metric,json_safe
from compare_metric_aif import prepare_input
from evaluate_metric_scenes import region_statistics
from metric_focus_fusion import inverse_depth_fusion
from selective_focus_fusion import MODES,PARAMETERS,selective_fusion


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=Path('results/multiscene_metric'))
    parser.add_argument('--output',type=Path,default=Path('results/selective_fusion'))
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    source=json.loads((args.input/'summary.json').read_text());guides=json.loads(Path('data/annotations/boundary_probes_20261115.json').read_text())
    all_rows=[];all_common=[];all_profiles=[];all_stats=[]
    for scene in source['scenes']:
        label=scene['scene'];folder=args.input/label;out=args.output/label;out.mkdir(parents=True,exist_ok=True)
        optical=np.load(folder/'calibrated_focus_meters.npy');neural=np.load(folder/'focus_scaled_relative_aif_meters.npy');conf=np.load(folder/'optical_confidence.npy')
        baseline=np.load(folder/'focus_neural_fusion_meters.npy');lidar=np.load(folder/'direct_lidar_meters.npy')
        np.testing.assert_allclose(inverse_depth_fusion(optical,neural,conf),baseline,atol=0,rtol=0,equal_nan=True)
        maps={'focus_scaled_neural':neural,'baseline_fusion':baseline};weights={};stats=[]
        for name,values,active in [('focus_scaled_neural',neural,np.zeros(neural.shape,bool)),
                                   ('baseline_fusion',baseline,np.isfinite(optical)&(optical>0))]:
            changed=np.isfinite(values)&np.isfinite(neural)&(np.abs(values-neural)>1e-6)
            stat=dict(scene=label,mode=name,initial_optical_fraction=float((np.isfinite(optical)&(optical>0)).mean()),
                      accepted_optical_fraction=float(active.mean()),changed_neural_fraction=float(changed.mean()),
                      output_coverage=float((np.isfinite(values)&(values>0)).mean()))
            stats.append(stat);all_stats.append(stat)
        for mode in MODES:
            fused,weight,stat=selective_fusion(optical,neural,conf,mode);maps[mode]=fused;weights[mode]=weight
            stat['scene']=label;stats.append(stat);all_stats.append(stat)
            np.save(out/(mode+'_meters.npy'),fused);np.save(out/(mode+'_optical_weights.npy'),weight)
        # Preserve the previously used common support, independent of candidate outputs.
        common=np.logical_and.reduce([np.isfinite(v)&(v>0) for v in [optical,neural,baseline,lidar,np.load(folder/'unadjusted_metric_aif_meters.npy')]])
        rows=[];common_rows=[];profiles=[]
        edge_annotations=next(s for s in guides['scenes'] if s['scene']==label)
        for name,values in maps.items():
            for region in scene['regions']:
                row=region_statistics(values,region);row.update(scene=label,method=name);rows.append(row);all_rows.append(row)
                row=region_statistics(np.where(common,values,np.nan),region);row.update(scene=label,method=name);common_rows.append(row);all_common.append(row)
            for edge in edge_annotations['edges']:
                if not edge['primary_silhouette_probe']:continue
                result=probe_map(far_inverse_metric(values),edge,scene['preprocessing'],guides['profile_half_width_native_px'],guides['min_along_edge_coverage'])
                row=dict(scene=label,method=name,edge=edge['name'],**result);profiles.append(row);all_profiles.append(row)
        (out/'metrics.json').write_text(json.dumps(json_safe(dict(regions=rows,common_support_regions=common_rows,profiles=profiles,influence=stats)),indent=2,allow_nan=False)+'\n')
        fig,axes=plt.subplots(2,3,figsize=(13,11),layout='constrained');axes=axes.ravel()
        for ax,(name,values) in zip(axes,maps.items()):
            image=ax.imshow(prepare_input(values,'upright'),vmin=.15,vmax=1,cmap='viridis',interpolation='nearest');ax.set_title(name);ax.axis('off')
        axes[-1].imshow(prepare_input(weights['agreement_edge_support'],'upright'),vmin=0,vmax=1,cmap='gray');axes[-1].set_title('Strict candidate optical weights');axes[-1].axis('off')
        fig.colorbar(image,ax=list(axes),shrink=.5,label='Meters; display 0.15–1.0, arrays unclipped');fig.suptitle(label+' — development candidates');fig.savefig(out/'comparison.png',dpi=130);plt.close(fig)
    summary=[]
    for method in ['focus_scaled_neural','baseline_fusion']+MODES:
        regions=[r for r in all_rows if r['method']==method and r['score_eligible']];shared=[r for r in all_common if r['method']==method and r['score_eligible']];probes=[p for p in all_profiles if p['method']==method];valid=[p for p in probes if p['status']=='valid'];widths=[p['transition_width_native_px'] for p in valid if p['transition_width_native_px'] is not None]
        influence=[s for s in all_stats if s['mode']==method]
        row=dict(method=method,n_regions=len(regions),region_mae_cm=float(np.mean([abs(r['signed_residual_cm']) for r in regions])) if len(regions)==11 else None,
                 n_common_regions=len(shared),common_support_subset_mae_cm=float(np.mean([abs(r['signed_residual_cm']) for r in shared])) if shared else None,
                 n_localizable=len(valid),n_primary_probes=len(probes),median_absolute_guide_offset_px=float(np.median([p['absolute_offset_native_px'] for p in valid])) if valid else None,
                 median_transition_width_px=float(np.median(widths)) if widths else None,
                 mean_accepted_optical_fraction=float(np.mean([s['accepted_optical_fraction'] for s in influence])) if influence else None,
                 mean_changed_neural_fraction=float(np.mean([s['changed_neural_fraction'] for s in influence])) if influence else None)
        summary.append(row)
    result=dict(scope='Development ablation designed after inspecting these scenes. No final held-out improvement claim.',
                constants=PARAMETERS,calibration_model_sha256=source['calibration_model_sha256'],baseline_reproduced_exactly=True,
                neural_scale_fit_reused=True,lidar_or_tape_used_for_gating=False,summary=summary,influence=all_stats,
                region_results=all_rows,common_support_region_results=all_common,profiles=all_profiles)
    (args.output/'results.json').write_text(json.dumps(json_safe(result),indent=2,allow_nan=False)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
