import numpy as np
import pytest
import compare_focus_estimators as study


def test_log_peak_recovers_gaussian_on_nonuniform_actual_lens_positions():
    lens=np.array([0,.12,.3,.41,.6,.85,1.])
    scores=np.exp(-.5*((lens-.43)/.15)**2)
    for method in ['log_parabola3','log_parabola5']:
        assert study.estimate_peak(lens,scores,method)['peak_lens_position']==pytest.approx(.43)
    assert study.estimate_peak(lens,scores,'sampled_max')['peak_lens_position']==.41
    with pytest.raises(ValueError,match='Unknown'):study.estimate_peak(lens,scores,'unknown')


def test_reestimate_preserves_source_observation():
    lens=np.linspace(0,1,9);scores=np.exp(-.5*((lens-.43)/.15)**2)
    obs=dict(lens_positions=lens.tolist(),sharpness=scores.tolist(),peak_lens_position=-1)
    revised=study.reestimate(obs,'log_parabola3')
    assert obs['peak_lens_position']==-1
    assert revised['peak_lens_position']==pytest.approx(.43)


def test_selection_weights_distances_not_repeat_count_and_rejects_missing_cases(monkeypatch):
    distances=[.2,.4,.4,.4,.6,.8,1.]
    rows=[dict(measured_m=d,status='valid',signed_residual_cm=10 if d==.4 else 0) for d in distances]
    model=dict(observations=[dict(distance_m=d) for d in distances])
    monkeypatch.setattr(study,'diagnose_distances',lambda model:dict(results=rows))
    score,_=study.calibration_score(model)
    assert score==pytest.approx(10/3)
    rows[4]['status']='outside_remaining_calibrated_range'
    assert study.calibration_score(model)[0] is None
