"""V5A caption tests use synthetic documents and never process media."""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import pytest

from videofactory import cli
from videofactory.caption_export import (CaptionExporter, render_srt, render_vtt,
                                         validate_sidecar, wrap_caption_text)
from videofactory.caption_timeline import build_caption_timeline, map_caption_time
from videofactory.models import read_json, write_json


def fixture(tmp_path: Path):
    project = tmp_path / "projects" / "demo"
    output = tmp_path / "output" / "demo"
    project.mkdir(parents=True)
    output.mkdir(parents=True)
    keeps = [
        {"source_start": 0, "source_end": 2, "source_duration": 2,
         "edited_start": 0, "edited_end": 2},
        {"source_start": 3, "source_end": 5, "source_duration": 2,
         "edited_start": 2, "edited_end": 4},
        {"source_start": 6, "source_end": 10, "source_duration": 4,
         "edited_start": 4, "edited_end": 8},
    ]
    mapping = {"schema_version": 1, "project_name": "demo", "source_duration": 10,
               "edited_duration": 8, "segments": keeps}
    primary = {"role": "PRIMARY", "source_path": "source.mov", "keep_segments": keeps}
    retimed = {"schema_version": 1, "project_name": "demo", "mode": "TALKING_HEAD",
               "source_duration": 10, "duration": 8, "retime_source": "retime_map.json",
               "base_video": primary, "base_audio": primary, "visual_overlays": []}
    words = [(0.5, 1.0, "Привет"), (1.5, 3.5, "мир."), (4.0, 4.5, "Это"),
             (5.5, 6.5, "тест!"), (7.0, 7.4, "Конец.")]
    transcript = {"schema_version": 1, "duration": 10, "language": "ru", "segments": [
        {"start": 0.5, "end": 7.4, "text": "Привет мир. Это тест! Конец.", "words": [
            {"start": a, "end": b, "text": text} for a, b, text in words]}]}
    for name, data in (("transcript.json", transcript), ("retime_map.json", mapping),
                       ("retimed_edit_timeline.json", retimed)):
        write_json(project / name, data)
    return project, output, transcript, mapping, retimed


def test_collapse_mapping_and_strict_boundaries(tmp_path):
    _, _, transcript, mapping, retimed = fixture(tmp_path)
    assert [map_caption_time(value, mapping) for value in (0, 1, 2, 2.5, 3, 4, 5.5, 6, 7, 10)] == [
        0, 1, 2, 2, 2, 3, 4, 4, 5, 8]
    for value in (-0.1, float("nan"), float("inf"), 11):
        with pytest.raises(ValueError):
            map_caption_time(value, mapping)
    timeline = build_caption_timeline("demo", transcript, mapping, retimed)
    assert [cue["text"] for cue in timeline["cues"]] == ["Привет мир.", "Это тест!", "Конец."]
    assert [(cue["start"], cue["end"]) for cue in timeline["cues"]] == [
        (0.5, 2.5), (3, 4.5), (5, 5.4)]
    assert [cue["index"] for cue in timeline["cues"]] == [1, 2, 3]
    assert timeline == build_caption_timeline("demo", transcript, mapping, retimed)


@pytest.mark.parametrize("bad", [(-1, 1), (float("nan"), 1), (1, float("inf")), (8, 11)])
def test_bad_word_timing_rejected(tmp_path, bad):
    _, _, transcript, mapping, retimed = fixture(tmp_path)
    transcript["segments"][0]["words"][0]["start"] = bad[0]
    transcript["segments"][0]["words"][0]["end"] = bad[1]
    with pytest.raises(ValueError):
        build_caption_timeline("demo", transcript, mapping, retimed)


def test_duration_and_project_mismatch_rejected(tmp_path):
    _, _, transcript, mapping, retimed = fixture(tmp_path)
    bad = copy.deepcopy(retimed)
    bad["duration"] = 7
    with pytest.raises(ValueError, match="disagree"):
        build_caption_timeline("demo", transcript, mapping, bad)
    bad = copy.deepcopy(mapping)
    bad["project_name"] = "other"
    with pytest.raises(ValueError, match="disagree"):
        build_caption_timeline("demo", transcript, bad, retimed)


def test_language_fallback_and_collapsed_word_repair(tmp_path):
    _, _, transcript, mapping, retimed = fixture(tmp_path)
    transcript.pop("language")
    assert build_caption_timeline("demo", transcript, mapping, retimed,
                                  project_language="en")["language"] == "en"
    assert build_caption_timeline("demo", transcript, mapping, retimed,
                                  project_language="auto")["language"] == "und"
    transcript["segments"][0]["words"][0] = {"start": 2.2, "end": 2.3, "text": "Привет"}
    transcript["segments"][0]["words"][1] = {"start": 2.4, "end": 3.5, "text": "мир."}
    timeline = build_caption_timeline("demo", transcript, mapping, retimed)
    assert timeline["cues"][0]["text"] == "Привет мир."
    assert timeline["cues"][0]["end"] > timeline["cues"][0]["start"]


def test_grouping_duration_wrap_and_unicode(tmp_path):
    _, _, transcript, mapping, retimed = fixture(tmp_path)
    transcript["segments"][0]["words"] = [
        {"start": 0.1 + n * 0.5, "end": 0.5 + n * 0.5, "text": "слово"}
        for n in range(16)]
    transcript["segments"][0]["text"] = " ".join("слово" for _ in range(16))
    timeline = build_caption_timeline("demo", transcript, mapping, retimed)
    assert len(timeline["cues"]) > 1
    assert all(cue["end"] - cue["start"] <= 6 for cue in timeline["cues"])
    assert sum(cue["text"].count("слово") for cue in timeline["cues"]) == 16
    assert wrap_caption_text("a " * 30).count("\n") == 1
    very_long = "д" * 90
    assert wrap_caption_text(very_long) == very_long


def test_sidecars_and_manifest_are_deterministic(tmp_path):
    project, output, transcript, mapping, retimed = fixture(tmp_path)
    exporter = CaptionExporter()
    manifest = exporter.export(project, output, "demo", transcript, mapping, retimed)
    timeline = read_json(project / "caption_timeline.json")
    srt = (output / "captions.srt").read_bytes()
    vtt = (output / "captions.vtt").read_bytes()
    assert srt == render_srt(timeline).encode("utf-8")
    assert vtt == render_vtt(timeline).encode("utf-8")
    assert b"00:00:00,500 --> 00:00:02,500" in srt
    assert b"00:00:00.500 --> 00:00:02.500" in vtt
    assert srt.endswith(b"\n") and vtt.startswith(b"WEBVTT\n\n") and vtt.endswith(b"\n")
    validate_sidecar(srt.decode(), timeline, "srt")
    validate_sidecar(vtt.decode(), timeline, "vtt")
    assert manifest["srt_sha256"] == hashlib.sha256(srt).hexdigest()
    assert manifest["vtt_sha256"] == hashlib.sha256(vtt).hexdigest()
    for name in ("transcript.json", "retime_map.json", "retimed_edit_timeline.json", "caption_timeline.json"):
        assert manifest[name.replace(".json", "_sha256")] == hashlib.sha256((project / name).read_bytes()).hexdigest()
    assert read_json(project / "caption_export_manifest.json") == manifest
    with pytest.raises(FileExistsError, match="overwrite-captions"):
        exporter.export(project, output, "demo", transcript, mapping, retimed)
    mp4s = [output / name for name in ("edited_draft.mp4", "speech_edited_draft.mp4", "punch_in_draft.mp4")]
    for path in mp4s:
        path.write_bytes(b"baseline")
    assert exporter.export(project, output, "demo", transcript, mapping, retimed, overwrite=True) == manifest
    assert (output / "captions.srt").read_bytes() == srt
    assert (output / "captions.vtt").read_bytes() == vtt
    assert all(path.read_bytes() == b"baseline" for path in mp4s)
    assert not list(output.glob("*.part")) and not list(output.glob("*.backup"))


def test_failure_cleans_staging_and_preserves_sidecars(tmp_path, monkeypatch):
    project, output, transcript, mapping, retimed = fixture(tmp_path)
    exporter = CaptionExporter()
    exporter.export(project, output, "demo", transcript, mapping, retimed)
    previous = [(output / name).read_bytes() for name in ("captions.srt", "captions.vtt")]
    import videofactory.caption_export as module
    original = module.validate_sidecar
    def fail_on_vtt(text, timeline, kind):
        if kind == "vtt":
            raise ValueError("injected validation failure")
        return original(text, timeline, kind)
    monkeypatch.setattr(module, "validate_sidecar", fail_on_vtt)
    with pytest.raises(ValueError, match="injected"):
        exporter.export(project, output, "demo", transcript, mapping, retimed, overwrite=True)
    assert [(output / name).read_bytes() for name in ("captions.srt", "captions.vtt")] == previous
    assert not list(output.glob("*.part")) and not list(output.glob("*.backup"))


def test_atomic_overwrite_rolls_back_if_second_publication_fails(tmp_path, monkeypatch):
    project, output, transcript, mapping, retimed = fixture(tmp_path)
    exporter = CaptionExporter()
    exporter.export(project, output, "demo", transcript, mapping, retimed)
    previous = [(output / name).read_bytes() for name in ("captions.srt", "captions.vtt")]
    import videofactory.caption_export as module
    original = module.os.replace
    def fail_second(source, target):
        if str(source).endswith("captions.vtt.part"):
            raise OSError("injected publication failure")
        return original(source, target)
    monkeypatch.setattr(module.os, "replace", fail_second)
    with pytest.raises(OSError, match="injected"):
        exporter.export(project, output, "demo", transcript, mapping, retimed, overwrite=True)
    assert [(output / name).read_bytes() for name in ("captions.srt", "captions.vtt")] == previous
    assert not list(output.glob("*.part")) and not list(output.glob("*.backup"))


def test_cli_export_routes_without_media_processing(tmp_path, monkeypatch, capsys):
    project, output, _, _, _ = fixture(tmp_path)
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    monkeypatch.setattr(cli, "default_whisper_model", lambda: "unused-local-model")
    assert cli.main(["--project", "demo", "--export-captions"]) == 0
    assert (output / "captions.srt").exists()
    assert "Caption cues: 3" in capsys.readouterr().out
    assert cli.main(["--project", "demo", "--export-captions", "--overwrite-captions"]) == 0
    with pytest.raises(SystemExit):
        cli.main(["--project", "demo", "--overwrite-captions"])
    with pytest.raises(SystemExit):
        cli.main(["--project", "demo", "--export-captions", "--video", "x.mov"])
