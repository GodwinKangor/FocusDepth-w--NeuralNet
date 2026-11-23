import numpy as np
import pytest
from outdoor_agreement import agreement


def test_agreement_keeps_reference_gate_and_missing_prediction_coverage_visible():
    lidar=np.array([.1,.3,.5,.9]);prediction=np.array([5.,.4,np.nan,1.])
    mask=(lidar>=.2)&(lidar<=1.)
    result=agreement(prediction,lidar,mask)
    assert result['n_evaluation_pixels']==3
    assert result['n_valid_pixels']==2
    assert result['in_span_coverage']==pytest.approx(2/3)
    assert result['median_abs_diff_cm']==pytest.approx(10)


def test_empty_support_is_missing_score_and_grid_mismatch_is_rejected():
    result=agreement(np.array([np.nan]),np.array([.3]),np.array([True]))
    assert result['median_abs_diff_cm'] is None
    assert result['in_span_coverage']==0
    assert agreement(np.array([.3]),np.array([.3]),np.array([False]))['in_span_coverage'] is None
    with pytest.raises(ValueError,match='grids differ'):
        agreement(np.ones((2,2)),np.ones(2),np.ones(2,bool))


@pytest.fixture
def outdoor_scene(tmp_path, monkeypatch):
    import hashlib
    import json
    import outdoor_agreement as evaluation

    monkeypatch.chdir(tmp_path)
    scene = 'fixture_scene'
    folder = tmp_path / 'results/outdoor_dusk' / scene
    folder.mkdir(parents=True)
    capture = tmp_path / 'data/raw' / scene
    capture.mkdir(parents=True)
    manifest = capture / 'manifest.json'
    manifest.write_text('{"capture":"fixture"}')
    calibration = tmp_path / 'results/focus_calibration/model.json'
    calibration.parent.mkdir(parents=True)
    calibration.write_text('{"frozen":true}')
    metadata = dict(registration_failed=False,
                    source_manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
                    calibration_model_sha256=hashlib.sha256(calibration.read_bytes()).hexdigest())
    (folder / 'preprocessing.json').write_text(json.dumps(metadata))
    np.save(folder / 'direct_lidar_meters.npy', np.array([.3, .5, .7, 1.2]))
    for method in evaluation.METHODS:
        values = np.array([.4, .6, .8, 1.3])
        if method == 'calibrated_focus':
            values[1] = np.nan
        if method == 'focus_scaled_relative_aif':
            values[2] = np.nan
        np.save(folder / (method + '_meters.npy'), values)
    monkeypatch.setattr(evaluation, 'ROOT', folder.parent)
    monkeypatch.setattr(evaluation, 'SCENES', [(scene, 'Fixture scene')])
    return evaluation, folder, manifest, calibration


def test_summary_compares_identical_pixels_and_retains_method_coverage(outdoor_scene):
    evaluation, _, _, _ = outdoor_scene
    row, = evaluation.summarize()
    assert row['in_span_fraction'] == pytest.approx(3 / 4)
    assert row['common_support_image_fraction'] == pytest.approx(1 / 4)
    assert row['common_support_in_span_fraction'] == pytest.approx(1 / 3)
    assert row['methods']['calibrated_focus']['n_valid_pixels'] == 2
    assert row['methods']['focus_neural_fusion']['n_valid_pixels'] == 3
    assert {v['n_valid_pixels'] for v in row['common_support_methods'].values()} == {1}
    assert {v['n_evaluation_pixels'] for v in row['common_support_methods'].values()} == {1}
    assert row['common_support_methods']['calibrated_focus']['median_abs_diff_cm'] == pytest.approx(10)


@pytest.mark.parametrize('changed_file,error', [('manifest', 'Source manifest changed'),
                                              ('calibration', 'Calibration changed'),
                                              ('registration', 'Registration failed')])
def test_summary_rejects_changed_sources_or_failed_registration(outdoor_scene, changed_file, error):
    import json
    evaluation, folder, manifest, calibration = outdoor_scene
    if changed_file == 'manifest':
        manifest.write_text('{}')
    elif changed_file == 'calibration':
        calibration.write_text('{}')
    else:
        path = folder / 'preprocessing.json'
        metadata = json.loads(path.read_text())
        metadata['registration_failed'] = True
        path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match=error):
        evaluation.summarize()
