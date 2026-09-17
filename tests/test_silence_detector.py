"""FFmpeg silence analysis is fully mocked in these tests."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from videofactory import silence_detector


def test_parser_pairs_one_and_multiple_intervals():
    log = """[silencedetect @ x] silence_start: 12.635333
[silencedetect @ x] silence_end: 13.887937 | silence_duration: 1.252604
[silencedetect @ x] silence_start: 29.914417
[silencedetect @ x] silence_end: 32.553458 | silence_duration: 2.639042"""
    assert silence_detector.parse_silencedetect(log, 64.03) == [
        (12.635333, 13.887937), (29.914417, 32.553458)]


@pytest.mark.parametrize("log", [
    "silence_start: 1",
    "silence_end: 2",
    "silence_start: 1\nsilence_start: 2\nsilence_end: 3",
    "silence_start: NaN\nsilence_end: 2",
    "silence_start: 1\nsilence_end: inf",
    "silence_start: -1\nsilence_end: 2",
    "silence_start: 2\nsilence_end: 1",
    "silence_start: 1\nsilence_end: 3\nsilence_start: 2\nsilence_end: 4",
    "silence_start: 19\nsilence_end: 21",
    "silence_start: bad\nsilence_end: 2",
])
def test_parser_rejects_malformed_nonfinite_overlap_and_out_of_bounds(log):
    with pytest.raises(ValueError):
        silence_detector.parse_silencedetect(log, 20)


def test_detector_uses_first_real_audio_stream_and_argument_array(monkeypatch):
    monkeypatch.setattr(silence_detector, "probe", lambda path: {"streams": [
        {"index": 0, "codec_type": "video"}, {"index": 1, "codec_type": "data"},
        {"index": 2, "codec_type": "audio"}, {"index": 3, "codec_type": "audio"}]})
    captured = []
    def fake_run(command, **kwargs):
        captured.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, "", "silence_start: 1\nsilence_end: 2")
    monkeypatch.setattr(silence_detector.subprocess, "run", fake_run)
    assert silence_detector.LocalAudioSilenceDetector().detect(Path("primary.mov"), 20, 1.0) == [(1, 2)]
    command, kwargs = captured[0]
    assert isinstance(command, list) and all(isinstance(arg, str) for arg in command)
    assert command[:5] == [str(silence_detector.FFMPEG), "-hide_banner", "-loglevel", "info", "-nostdin"]
    assert command[command.index("-map") + 1] == "0:2"
    assert command[command.index("-af") + 1] == "silencedetect=noise=-35dB:d=1"
    assert command[-3:] == ["-f", "null", "-"]
    assert kwargs == {"capture_output": True, "text": True, "check": False}
    silence_detector.LocalAudioSilenceDetector().detect(Path("primary.mov"), 20, 0.5, -42)
    assert captured[1][0][captured[1][0].index("-af") + 1] == "silencedetect=noise=-42dB:d=0.5"


def test_ffmpeg_failure_and_missing_audio_rejected(monkeypatch):
    monkeypatch.setattr(silence_detector, "probe", lambda path: {"streams": [{"index": 0, "codec_type": "data"}]})
    with pytest.raises(ValueError, match="audio stream"):
        silence_detector.LocalAudioSilenceDetector().detect(Path("primary.mov"), 20, 1)
    monkeypatch.setattr(silence_detector, "probe", lambda path: {"streams": [{"index": 1, "codec_type": "audio"}]})
    monkeypatch.setattr(silence_detector.subprocess, "run", lambda command, **kwargs:
                        subprocess.CompletedProcess(command, 1, "", "X" * 1000))
    with pytest.raises(RuntimeError, match="FFmpeg silence analysis failed") as exc:
        silence_detector.LocalAudioSilenceDetector().detect(Path("primary.mov"), 20, 1)
    assert len(str(exc.value)) < 600


@pytest.mark.parametrize("noise", [-100, 0, -101, 1, float("nan"), float("inf")])
def test_invalid_noise_threshold_rejected(noise):
    with pytest.raises(ValueError, match="noise threshold"):
        silence_detector.validate_noise_db(noise)
