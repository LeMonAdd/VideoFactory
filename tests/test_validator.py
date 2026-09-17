from pathlib import Path

import pytest

from videofactory.validator import validate_output


def test_validator_accepts_expected_probe_result(tmp_path, monkeypatch):
    output = tmp_path / "draft.mp4"
    output.write_bytes(b"media")
    monkeypatch.setattr("videofactory.validator.probe", lambda path: {
        "format": {"duration": "20"},
        "streams": [
            {"codec_type": "video", "codec_name": "h264",
             "width": 1920, "height": 1080, "avg_frame_rate": "30/1"},
            {"codec_type": "audio", "codec_name": "aac"},
        ],
    })
    result = validate_output(output, {"width": 1920, "height": 1080, "fps": 30})
    assert result["video_codec"] == "h264" and result["audio_codec"] == "aac"


def test_validator_rejects_missing_audio(tmp_path, monkeypatch):
    output = tmp_path / "draft.mp4"
    output.write_bytes(b"media")
    monkeypatch.setattr("videofactory.validator.probe", lambda path: {
        "format": {"duration": "20"},
        "streams": [{"codec_type": "video", "codec_name": "h264"}],
    })
    with pytest.raises(ValueError, match="video and audio"):
        validate_output(output, {"width": 1920, "height": 1080, "fps": 30})
