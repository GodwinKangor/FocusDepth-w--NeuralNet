"""Offline focus-estimator sensitivity study; never overwrite production calibration.

Candidate selection uses only leave-one-distance-out calibration errors. Previously
inspected validation captures are exploratory diagnostics, not fresh final tests.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from focus_calibration import curve_peak, diagnose_distances, fit_mapping, validate_observation

METHODS=['parabola3','sampled_max','log_parabola3','parabola5','log_parabola5']


def estimate_peak(lens, scores, method):
    if method not in METHODS:raise ValueError('Unknown estimator')
    baseline=curve_peak(lens,scores)
    if method=='parabola3':return baseline
    x=np.asarray(lens,dtype=float);y=np.asarray(scores,dtype=float);k=baseline['peak_frame']
    if method=='sampled_max':return {**baseline,'peak_lens_position':float(x[k])}
    radius=2 if method.endswith('5') else 1
    if k<radius or k+radius>=len(x):raise ValueError('Insufficient points around focus peak')
    xx=x[k-radius:k+radius+1]-x[k];yy=y[k-radius:k+radius+1]
    if method.startswith('log_'):
        if np.any(yy<=0):raise ValueError('Log fit requires positive sharpness')
        yy=np.log(yy)
    a,b,_=np.polyfit(xx,yy,2)
    if a>=0:raise ValueError('Candidate fit is not concave')
    peak=x[k]-b/(2*a)
    if not x[k-1]<=peak<=x[k+1]:raise ValueError('Candidate peak leaves neighboring bracket')
    return {**baseline,'peak_lens_position':float(peak)}


def reestimate(obs,method):
    return {**obs,**estimate_peak(obs['lens_positions'],obs['sharpness'],method),'peak_estimator':method}


def calibration_score(model):
    diagnostics=diagnose_distances(model)
    distances=sorted({o['distance_m'] for o in model['observations']})
    interior=distances[1:-1]
    relevant=[r for r in diagnostics['results'] if r['measured_m'] in interior]
    # Reject incomplete candidates rather than favoring estimates that omit hard cases.
    if not relevant or any(r['status']!='valid' for r in relevant):
        return None,diagnostics
    macro=float(np.mean([np.mean([abs(r['signed_residual_cm']) for r in relevant if r['measured_m']==d]) for d in interior]))
    return macro,diagnostics


def load_development_captures():
    groups=[('initial_targets',Path('results/focus_validation_20261115/inspection.json')),
            ('rice_broad24',Path('results/focus_validation_repeats_20261115/inspection.json')),
            ('narrow_sampling',Path('results/focus_dense_validation_20261115/inspection.json'))]
    result=[];seen=set()
    for label,path in groups:
        value=json.loads(path.read_text());rows=value if isinstance(value,list) else value['captures']
        for row in rows:
            obs=row['observation'];manifest_path=Path(obs['folder'])/'manifest.json';raw=manifest_path.read_bytes()
            if hashlib.sha256(raw).hexdigest()!=obs['manifest_sha256']:raise ValueError('Source manifest changed')
            manifest=json.loads(raw)
            if manifest.get('capture_purpose')!='validation':raise ValueError('Development result must originate from validation capture')
            if obs['manifest_sha256'] in seen:raise ValueError('Duplicate development capture')
            seen.add(obs['manifest_sha256'])
            condition=label
            if label=='narrow_sampling':condition='narrow'+str(len(manifest['frames']))
            result.append(dict(condition=condition,observation=obs))
    return result


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',type=Path,default=Path('results/focus_calibration/model.json'))
    parser.add_argument('--output',type=Path,default=Path('results/focus_estimator_study'))
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    original=args.model.read_bytes();baseline=json.loads(original);candidates={};summary=[]
    # Lock selection before loading any validation/development captures.
    for method in METHODS:
        try:
            observations=[reestimate(o,method) for o in baseline['observations']]
            candidate=fit_mapping(observations,baseline['roi']);candidate['peak_estimator']=method
            score,diagnostics=calibration_score(candidate)
            if method=='parabola3':
                np.testing.assert_allclose([g['peak_lens_position'] for g in candidate['groups']],
                                           [g['peak_lens_position'] for g in baseline['groups']],atol=1e-12,rtol=0)
            candidates[method]=candidate
            (args.output/(method+'_model.json')).write_text(json.dumps(candidate,indent=2)+'\n')
            summary.append(dict(method=method,calibration_distance_macro_mae_cm=score,
                                calibration_diagnostics=diagnostics,calibration_status='valid'))
        except ValueError as error:
            summary.append(dict(method=method,calibration_distance_macro_mae_cm=None,calibration_status=str(error)))
    eligible=[r for r in summary if r['calibration_distance_macro_mae_cm'] is not None]
    selected=min(eligible,key=lambda r:r['calibration_distance_macro_mae_cm'])['method'] if eligible else None
    development=load_development_captures();results=[]
    for method,candidate in candidates.items():
        for item in development:
            try:
                obs=reestimate(item['observation'],method);result=validate_observation(obs,candidate)
                result.update(peak_lens_position=obs['peak_lens_position'],measured_uncertainty_m=obs['uncertainty_m'])
            except ValueError as error:
                result=dict(folder=item['observation']['folder'],status='rejected',reason=str(error))
            results.append(dict(method=method,condition=item['condition'],**result))
    for row in summary:
        row['development']={}
        for condition in sorted({d['condition'] for d in development}):
            subset=[r for r in results if r['method']==row['method'] and r['condition']==condition]
            valid=[r for r in subset if r['status']=='valid']
            row['development'][condition]=dict(n_total=len([d for d in development if d['condition']==condition]),n_valid=len(valid),
                mae_cm=float(np.mean([abs(r['signed_residual_cm']) for r in valid])) if len(valid)==len([d for d in development if d['condition']==condition]) else None,
                predictions_cm=[r['predicted_m']*100 for r in valid])
        print(row['method'],'calibration macro MAE',row['calibration_distance_macro_mae_cm'],
              'development', {k:v['mae_cm'] for k,v in row['development'].items()})
    output=dict(scope='Candidate selection on calibration only; previously inspected validation is exploratory development evidence.',
                baseline_model_sha256=hashlib.sha256(original).hexdigest(),selection_metric='Equal weight per interior omitted calibration distance; all repeats omitted together; incomplete candidates ineligible',
                selected_on_calibration_only=selected,production_model_changed=False,summary=summary,development_results=results)
    (args.output/'results.json').write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')
    print('Selected on calibration only:',selected)
    assert args.model.read_bytes()==original
    fig,axes=plt.subplots(1,2,figsize=(14,5))
    names=[r['method'] for r in summary];scores=[r['calibration_distance_macro_mae_cm'] if r['calibration_distance_macro_mae_cm'] is not None else np.nan for r in summary]
    axes[0].bar(names,scores);axes[0].set_ylabel('Calibration distance-macro MAE cm');axes[0].tick_params(axis='x',rotation=30);axes[0].set_title('Selection uses calibration only')
    for condition in ['initial_targets','rice_broad24','narrow40']:
        axes[1].plot(names,[r['development'][condition]['mae_cm'] if r['development'][condition]['mae_cm'] is not None else np.nan for r in summary],'o-',label=condition)
    axes[1].set_ylabel('Exploratory development MAE cm');axes[1].tick_params(axis='x',rotation=30);axes[1].set_title('Previously inspected captures: not final holdout');axes[1].legend();fig.tight_layout();fig.savefig(args.output/'comparison.png',dpi=140)


if __name__=='__main__':main()
