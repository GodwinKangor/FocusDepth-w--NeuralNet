import numpy as np
import pytest
from focus_calibration import curve_peak, fit_mapping, depth_from_focus, validate_observation, observation


def test_uncertainty_correction_is_bound_to_original_manifest():
    from focus_calibration import corrected_uncertainty
    record={'corrections':{'hash':{'original_uncertainty_m':.13,'corrected_uncertainty_m':.02}}}
    assert corrected_uncertainty(.13,'hash',record)[0]==.02
    assert corrected_uncertainty(.13,'different',record)==(.13,None)
    with pytest.raises(ValueError,match='original'):
        corrected_uncertainty(.03,'hash',record)


def observations():
    return [dict(signature={'camera':'same'},distance_m=d,peak_lens_position=p,
                 manifest_sha256=str(i),capture_identity=['camera',str(i)])
            for i,(d,p) in enumerate(zip([.2,.4,.6,1.],[.2,.4,.6,.8]))]


def test_nonuniform_actual_positions_and_ambiguous_peaks():
    lens=np.array([0,.12,.3,.41,.6,.85,1.])
    peak=curve_peak(lens,1-3*(lens-.43)**2)
    assert peak['peak_lens_position']==pytest.approx(.43)
    with pytest.raises(ValueError,match='endpoint'):curve_peak(lens,np.arange(7))
    with pytest.raises(ValueError,match='flat'):curve_peak(lens,[1,1,1,1.001,1,1,1])


def test_mapping_interpolates_inverse_depth_and_rejects_extrapolation():
    model=fit_mapping(observations(),[.4,.4,.6,.6])
    depths=depth_from_focus(np.array([.1,.2,.3,.4,.8,.9]),model)
    assert np.isnan(depths[[0,5]]).all()
    np.testing.assert_allclose(depths[1:5],[.2,1/3.75,.4,1.])
    new=dict(signature={'camera':'same'},manifest_sha256='new',capture_identity=['camera','new'],
             peak_lens_position=.3,distance_m=1/3.75,uncertainty_m=.01,folder='validation')
    assert abs(validate_observation(new,model)['signed_residual_cm'])<1e-5
    new['capture_identity']=model['observations'][0]['capture_identity']
    with pytest.raises(ValueError,match='reused'):validate_observation(new,model)


def test_bad_order_and_mixed_camera_rejected():
    data=observations();data[2]['peak_lens_position']=.3
    with pytest.raises(ValueError,match='ordered'):fit_mapping(data,[.4,.4,.6,.6])
    data=observations();data[2]['signature']={'camera':'different'}
    with pytest.raises(ValueError,match='mixes'):fit_mapping(data,[.4,.4,.6,.6])


def test_capture_role_cannot_be_relabelled_by_cli(tmp_path):
    (tmp_path/'manifest.json').write_text('{"capture_purpose":"calibration"}')
    with pytest.raises(ValueError,match='expected validation'):observation(tmp_path,'validation',[.4,.4,.6,.6])


def test_observation_extracts_native_image_focus_curve(tmp_path):
    import json
    import cv2
    texture=np.random.default_rng(8).integers(0,256,(96,128),dtype=np.uint8)
    frames=[]
    for i in range(9):
        image=cv2.GaussianBlur(texture,(0,0),.3+abs(i-4)*.7)
        name=f'frame_{i:03d}.jpg';cv2.imwrite(str(tmp_path/name),image)
        frames.append(dict(file=name,lens_position_actual=i/8))
    manifest=dict(capture_purpose='calibration',capture_device_id='installation',camera_unique_id='camera',camera='lidar_rgb',
                  image_width=128,image_height=96,geometric_distortion_correction=False,
                  calibration_target_distance_m=.4,measurement_uncertainty_m=.01,
                  created_at='test',frames=frames)
    (tmp_path/'manifest.json').write_text(json.dumps(manifest))
    obs=observation(tmp_path,'calibration',[.25,.25,.75,.75])
    assert obs['peak_frame']==4
    assert obs['peak_lens_position']==pytest.approx(.5,abs=.02)
    assert obs['distance_m']==.4


def test_leave_one_distance_out_removes_all_repeats_and_preserves_range():
    from focus_calibration import diagnose_distances
    data=observations()
    for o in data:o['folder']=o['manifest_sha256']
    data.append(dict(signature={'camera':'same'},distance_m=1.2,peak_lens_position=.9,
                     manifest_sha256='extra',capture_identity=['camera','extra'],folder='extra'))
    repeated=dict(data[1]);repeated.update(manifest_sha256='repeat',capture_identity=['camera','repeat'],folder='repeat')
    data.append(repeated)
    model=fit_mapping(data,[.45,.45,.55,.55])
    report=diagnose_distances(model)
    assert len(report['results'])==6
    for row in report['results']:
        assert row['measured_m'] not in row['training_distances_m']
        if row['measured_m'] in [.2,1.2]:assert row['predicted_m'] is None
    four=fit_mapping(data[:4],[.45,.45,.55,.55])
    assert all(r['status']=='insufficient_remaining_distances' for r in diagnose_distances(four)['results'])
