"""V4B timeline and render tests use synthetic files and mocked media commands."""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import pytest

from test_edit_renderer import fake_probe, render_fixture
from videofactory import cli, edit_renderer, speech_renderer
from videofactory.models import read_json, write_json
from videofactory.retimed_edit import (build_retimed_edit_timeline, retime_overlay,
                                       validate_retimed_edit_timeline)
from videofactory.speech_edit import (SpeechEditPlanner, build_retime_map, map_source_time,
                                      validate_retime_map, validate_saved_speech_edit_plan)


def speech_fixture(tmp_path):
    project_dir, project, director, clips, timeline = render_fixture(tmp_path)
    transcript = {"schema_version": 1, "duration": 20, "language": "en", "segments": [
        {"start": 0.5, "end": 19.2, "text": "fixture", "words": [
            {"start": a, "end": b, "text": "word"} for a, b in
            ((0.5, 1), (1.4, 1.8), (3, 3.4), (5, 5.4), (10, 10.3), (19, 19.2))]}]}
    acoustic = [(6, 7.5), (11, 12.5)]
    speech = SpeechEditPlanner().plan(project, transcript, timeline, acoustic)
    mapping = build_retime_map(speech)
    retimed = build_retimed_edit_timeline(project, transcript, director, clips, timeline,
                                          speech, mapping, project_dir)
    for name, document in (("project.json", project), ("transcript.json", transcript),
                           ("director_plan.json", director), ("clip_plan.json", clips),
                           ("edit_timeline.json", timeline), ("speech_edit_plan.json", speech),
                           ("retime_map.json", mapping), ("retimed_edit_timeline.json", retimed)):
        write_json(project_dir / name, document)
    return project_dir, project, transcript, director, clips, timeline, speech, mapping, retimed


def test_saved_artifacts_and_canonical_timeline(tmp_path):
    values = speech_fixture(tmp_path)
    project_dir, project, transcript, director, clips, timeline, speech, mapping, retimed = values
    assert validate_saved_speech_edit_plan(speech, project, transcript, timeline) == speech
    assert validate_retime_map(mapping, speech) == mapping
    assert validate_retimed_edit_timeline(retimed, project, clips, mapping) == retimed
    assert retimed["base_video"]["keep_segments"] == retimed["base_audio"]["keep_segments"] == mapping["segments"]
    assert [piece["shot_id"] for piece in retimed["visual_overlays"]] == [2, 2, 7, 7]
    assert [piece["segment_index"] for piece in retimed["visual_overlays"]] == [1, 2, 1, 2]
    assert [piece["source_in"] for piece in retimed["visual_overlays"]][1] > retimed["visual_overlays"][0]["source_out"]
    assert retimed["duration"] == speech["edited_duration"]
    assert map_source_time(20, mapping) == retimed["duration"]
    with pytest.raises(ValueError, match="removed region"):
        map_source_time(6.5, mapping)
    assert build_retimed_edit_timeline(project, transcript, director, clips, timeline,
                                       speech, mapping, project_dir) == retimed


def test_saved_plan_rejects_corruption_without_audio_analysis(tmp_path):
    project_dir, project, transcript, director, clips, timeline, speech, mapping, _ = speech_fixture(tmp_path)
    bad = copy.deepcopy(speech)
    bad["project_name"] = "other"
    with pytest.raises(ValueError, match="project or duration"):
        validate_saved_speech_edit_plan(bad, project, transcript, timeline)
    bad = copy.deepcopy(speech)
    bad["source_duration"] = 21
    with pytest.raises(ValueError, match="project or duration"):
        validate_saved_speech_edit_plan(bad, project, transcript, timeline)
    bad = copy.deepcopy(speech)
    bad["edited_duration"] = 1
    with pytest.raises(ValueError, match="totals"):
        validate_saved_speech_edit_plan(bad, project, transcript, timeline)
    bad = copy.deepcopy(speech)
    bad["keep_segments"][1]["source_start"] = 0
    with pytest.raises(ValueError, match="Keep segments"):
        validate_saved_speech_edit_plan(bad, project, transcript, timeline)
    bad = copy.deepcopy(mapping)
    bad["segments"][1]["edited_start"] = 0
    with pytest.raises(ValueError):
        validate_retime_map(bad, speech)
    bad = copy.deepcopy(speech)
    bad["cuts"][1]["source_start"] = bad["cuts"][0]["source_start"]
    with pytest.raises(ValueError, match="ordered"):
        validate_saved_speech_edit_plan(bad, project, transcript, timeline)


def test_overlay_intersections_boundaries_and_source_advancement():
    clip = {"status": "PLANNED", "candidate_id": "pexels:video:42",
            "relative_path": "media/broll/pexels_video_42.mp4", "asset_duration": 20,
            "source_in": 3, "source_out": 8}
    overlay = {"shot_id": 7, "visual_type": "B_ROLL", "timeline_start": 10,
               "timeline_end": 15, "source_path": clip["relative_path"], "source_in": 3,
               "source_out": 8}
    keeps = [{"source_start": 0, "source_end": 12, "edited_start": 0, "edited_end": 12},
             {"source_start": 13, "source_end": 20, "edited_start": 12, "edited_end": 19}]
    pieces = retime_overlay(overlay, clip, keeps)
    assert [(p["timeline_start"], p["timeline_end"], p["source_in"], p["source_out"])
            for p in pieces] == [(10, 12, 3, 5), (12, 14, 6, 8)]
    assert retime_overlay(overlay | {"timeline_start": 0, "timeline_end": 2,
                                     "source_in": 3, "source_out": 5}, clip, keeps)[0]["timeline_start"] == 0
    assert retime_overlay(overlay | {"timeline_start": 13, "timeline_end": 15,
                                     "source_in": 3, "source_out": 5}, clip, keeps)[0]["timeline_start"] == 12
    assert retime_overlay(overlay | {"timeline_start": 12, "timeline_end": 13,
                                     "source_in": 3, "source_out": 4}, clip, keeps) == []
    assert len(retime_overlay(overlay | {"timeline_start": 10, "timeline_end": 12,
                                         "source_in": 3, "source_out": 5}, clip, keeps)) == 1
    assert len(retime_overlay(overlay | {"timeline_start": 13, "timeline_end": 15,
                                         "source_in": 3, "source_out": 5}, clip, keeps)) == 1
    more_keeps = [{"source_start": 0, "source_end": 11, "edited_start": 0, "edited_end": 11},
                  {"source_start": 12, "source_end": 13, "edited_start": 11, "edited_end": 12},
                  {"source_start": 14, "source_end": 20, "edited_start": 12, "edited_end": 18}]
    assert [(p["source_in"], p["source_out"]) for p in retime_overlay(overlay, clip, more_keeps)] == [
        (3, 4), (5, 6), (7, 8)]
    with pytest.raises(ValueError, match="asset duration"):
        retime_overlay(overlay, clip | {"asset_duration": 7}, keeps)


def mock_media(monkeypatch, expected_duration):
    def probe(path):
        info = fake_probe(path)
        if path.name.endswith(".part.mp4"):
            info["format"]["duration"] = str(expected_duration)
            info["streams"][1]["duration"] = str(expected_duration)
            info["streams"][0]["pix_fmt"] = "yuv420p"
        return info
    monkeypatch.setattr(edit_renderer, "probe", probe)
    commands = []
    def run(command):
        commands.append(command)
        Path(command[-1]).write_bytes(b"mock speech edit")
    monkeypatch.setattr(speech_renderer, "run_command", run)
    return commands


def render(values, output, overwrite=False):
    project_dir, project, transcript, director, clips, timeline, speech, mapping, retimed = values
    return speech_renderer.SpeechEditRenderer().render(
        project_dir, output, project, transcript, director, clips, timeline,
        read_json(project_dir / "download_manifest.json"), speech, mapping, retimed,
        overwrite=overwrite)


def test_filter_graph_and_atomic_publication(tmp_path, monkeypatch):
    values = speech_fixture(tmp_path)
    speech, retimed = values[6], values[8]
    output = tmp_path / "output" / "demo"
    output.mkdir(parents=True)
    (output / "edited_draft.mp4").write_bytes(b"V3D baseline")
    commands = mock_media(monkeypatch, speech["edited_duration"])
    manifest = render(values, output)
    assert len(commands) == 1
    command = commands[0]
    assert command[-1].endswith("speech_edited_draft.mp4.part.mp4")
    assert command.count("-i") == 2  # primary and one physical repeated B-roll asset
    graph = command[command.index("-filter_complex") + 1]
    assert "[0:0]split=3[vsrc0][vsrc1][vsrc2]" in graph
    assert "[0:1]asplit=3[asrc0][asrc1][asrc2]" in graph
    assert graph.count("trim=start=") == 10  # 3 video + 3 audio + 4 B-roll pieces
    assert graph.count("asetpts=PTS-STARTPTS") == 3
    assert "concat=n=3:v=1:a=0[base]" in graph
    assert "concat=n=3:v=0:a=1[aout]" in graph
    assert "[1:v:0]split=4" in graph
    assert "-map" in command and command[command.index("-map") + 1] == "[vout]"
    assert command[command.index("-map", command.index("-map") + 1) + 1] == "[aout]"
    assert "-map_metadata" in command and "-map_chapters" in command and "-dn" in command and "-sn" in command
    assert command[command.index("-pix_fmt") + 1] == "yuv420p"
    assert command[command.index("-r") + 1] == "30"
    assert command[command.index("-c:v") + 1] == "libx264"
    assert command[command.index("-c:a") + 1] == "aac"
    assert command[command.index("-ar") + 1] == "48000"
    assert "+faststart" in command
    assert (output / "edited_draft.mp4").read_bytes() == b"V3D baseline"
    assert (output / "speech_edited_draft.mp4").read_bytes() == b"mock speech edit"
    assert not (output / "speech_edited_draft.mp4.part.mp4").exists()
    assert manifest["keep_segment_count"] == 3
    assert manifest["retimed_broll_overlay_segment_count"] == 4
    assert manifest["sha256"] == hashlib.sha256(b"mock speech edit").hexdigest()
    assert manifest["speech_edit_plan_sha256"] == hashlib.sha256(
        (values[0] / "speech_edit_plan.json").read_bytes()).hexdigest()
    assert read_json(values[0] / "speech_render_manifest.json") == manifest


def test_primary_stream_indexes_and_broll_integrity_before_ffmpeg(tmp_path, monkeypatch):
    values = speech_fixture(tmp_path)
    project_dir, retimed = values[0], values[8]
    info = {"streams": [{"index": 2, "codec_type": "video"},
                        {"index": 3, "codec_type": "audio"}]}
    source = project_dir / "media" / "broll" / "pexels_video_42.mp4"
    command = speech_renderer.build_speech_render_command(
        retimed, project_dir, tmp_path / "part.mp4", info, {source.resolve(): fake_probe(source)})
    graph = command[command.index("-filter_complex") + 1]
    assert "[0:2]split=3" in graph and "[0:3]asplit=3" in graph
    commands = mock_media(monkeypatch, values[6]["edited_duration"])
    source.write_bytes(b"corrupted B-roll")
    with pytest.raises(ValueError, match="hash differs"):
        render(values, tmp_path / "output")
    assert commands == []


def test_fully_cut_broll_creates_no_unused_split_branch(tmp_path, monkeypatch):
    project_dir, project, transcript, director, clips, timeline, _, _, _ = speech_fixture(tmp_path)
    assert timeline["visual_overlays"]  # the original edit contains valid B-roll
    speech = SpeechEditPlanner().plan(project, transcript, timeline, [(3.5, 14.5)])
    mapping = build_retime_map(speech)
    retimed = build_retimed_edit_timeline(project, transcript, director, clips, timeline,
                                          speech, mapping, project_dir)
    assert retimed["visual_overlays"] == []
    monkeypatch.setattr(edit_renderer, "probe", fake_probe)
    primary_info, visual_sources = edit_renderer.inspect_render_inputs(
        project, timeline, clips, director, read_json(project_dir / "download_manifest.json"),
        project_dir)
    assert len(visual_sources) == 1  # integrity-checked input has no surviving uses
    command = speech_renderer.build_speech_render_command(
        retimed, project_dir, tmp_path / "speech.part.mp4", primary_info, visual_sources)
    graph = command[command.index("-filter_complex") + 1]
    assert "split=0" not in graph
    assert "[1:v:0]split=" not in graph
    assert "[base]null[vout]" in graph
    assert command[command.index("-map") + 1] == "[vout]"
    assert command[command.index("-map", command.index("-map") + 1) + 1] == "[aout]"
    assert "[1:a" not in graph and "-map 1:" not in " ".join(command)


def test_changed_saved_plan_rejected_before_ffmpeg(tmp_path, monkeypatch):
    values = speech_fixture(tmp_path)
    commands = mock_media(monkeypatch, values[6]["edited_duration"])
    changed = read_json(values[0] / "retime_map.json")
    changed["edited_duration"] = 1
    write_json(values[0] / "retime_map.json", changed)
    with pytest.raises(ValueError, match="changed before speech rendering"):
        render(values, tmp_path / "output")
    assert commands == []


def test_render_failure_validation_and_overwrite(tmp_path, monkeypatch):
    values = speech_fixture(tmp_path)
    output = tmp_path / "output" / "demo"
    output.mkdir(parents=True)
    (output / "speech_edited_draft.mp4").write_bytes(b"previous")
    (output / "edited_draft.mp4").write_bytes(b"V3D")
    commands = mock_media(monkeypatch, values[6]["edited_duration"])
    with pytest.raises(FileExistsError, match="--overwrite-render"):
        render(values, output)
    assert commands == []
    monkeypatch.setattr(speech_renderer, "run_command", lambda command: (_ for _ in ()).throw(RuntimeError("FFmpeg failed")))
    with pytest.raises(RuntimeError, match="FFmpeg failed"):
        render(values, output, overwrite=True)
    assert (output / "speech_edited_draft.mp4").read_bytes() == b"previous"
    assert not (output / "speech_edited_draft.mp4.part.mp4").exists()
    commands = mock_media(monkeypatch, values[6]["edited_duration"])
    def bad_probe(path):
        info = fake_probe(path)
        if path.name.endswith(".part.mp4"):
            info["format"]["duration"] = "1"
        return info
    monkeypatch.setattr(edit_renderer, "probe", bad_probe)
    with pytest.raises(ValueError, match="duration differs"):
        render(values, output, overwrite=True)
    assert (output / "speech_edited_draft.mp4").read_bytes() == b"previous"
    assert not (output / "speech_edited_draft.mp4.part.mp4").exists()
    mock_media(monkeypatch, values[6]["edited_duration"])
    render(values, output, overwrite=True)
    assert (output / "speech_edited_draft.mp4").read_bytes() == b"mock speech edit"
    assert (output / "edited_draft.mp4").read_bytes() == b"V3D"


def test_cli_route_uses_frozen_artifacts_without_detector(tmp_path, monkeypatch):
    values = speech_fixture(tmp_path)
    config = tmp_path / "config" / "transcription.json"
    write_json(config, {"schema_version": 1, "default_whisper_model": "local/model"})
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    class Forbidden:
        def __init__(self, *_args, **_kwargs):
            raise AssertionError("Detector, Whisper, Codex, or network path invoked")
    for name in ("LocalAudioSilenceDetector", "RealMLXTranscriber", "CodexDirector", "PexelsSourceProvider"):
        monkeypatch.setattr(cli, name, Forbidden)
    calls = []
    class FakeRenderer:
        def render(self, *args, **kwargs):
            calls.append((args, kwargs))
            kwargs["before_render"]()
            kwargs["before_validate"]()
    monkeypatch.setattr(cli, "SpeechEditRenderer", FakeRenderer)
    assert cli.main(["--project", "demo", "--render-speech-edits", "--encoder", "libx264"]) == 0
    assert len(calls) == 1
    assert read_json(values[0] / "retimed_edit_timeline.json") == values[8]
    with pytest.raises(SystemExit):
        cli.main(["--project", "demo", "--render-speech-edits", "--video", "input.mov"])
