import importlib.util
from pathlib import Path

import pytest

spec=importlib.util.spec_from_file_location('presentation',Path(__file__).resolve().parents[1]/'scripts/build_presentation_video.py')
presentation=importlib.util.module_from_spec(spec)
spec.loader.exec_module(presentation)


def test_long_voice_take_is_rejected_before_any_audio_is_cut(tmp_path,monkeypatch):
    clips=tmp_path/'clips';clips.mkdir()
    (clips/'01.m4a').touch()
    monkeypatch.setattr(presentation,'audio_duration',lambda ffmpeg,path:23.)
    def forbidden(*args,**kwargs):
        raise AssertionError('Encoding must not start on an overlong voice take')
    monkeypatch.setattr(presentation.subprocess,'run',forbidden)
    with pytest.raises(ValueError,match='no speech was truncated'):
        presentation.assemble_voice_clips('ffmpeg',clips,tmp_path)


def test_duplicate_takes_are_not_chosen_arbitrarily(tmp_path):
    (tmp_path/'01.m4a').touch();(tmp_path/'01.wav').touch()
    with pytest.raises(ValueError,match='exactly one voice clip'):
        presentation.assemble_voice_clips('ffmpeg',tmp_path,tmp_path)
