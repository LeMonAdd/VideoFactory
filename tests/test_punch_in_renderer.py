"""V4C tests use synthetic project files and mocked render/probe calls only."""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import pytest

from test_edit_renderer import fake_probe
from test_speech_renderer import mock_media, speech_fixture
from videofactory import cli, edit_renderer, punch_in_renderer
from videofactory.jump_cut_style import (build_jump_cut_style_plan, build_styled_edit_timeline,
                                         validate_jump_cut_style_plan,
                                         validate_punch_in_scale,
                                         validate_styled_edit_timeline)
from videofactory.models import read_json, write_json


def styled_fixture(tmp_path, scale=1.08):
    values = speech_fixture(tmp_path)
    project_dir, retimed = values[0], values[8]
    style = build_jump_cut_style_plan(retimed, scale)
    styled = build_styled_edit_timeline(retimed, style)
    write_json(project_dir / "jump_cut_style_plan.json", style)
    write_json(project_dir / "styled_edit_timeline.json", styled)
    return values, style, styled


def render(values, style, styled, output, overwrite=False):
    project_dir, project, transcript, director, clips, timeline, speech, mapping, retimed = values
    return punch_in_renderer.PunchInRenderer().render(
        project_dir, output, project, transcript, director, clips, timeline,
        read_json(project_dir / "download_manifest.json"), speech, mapping, retimed,
        style, styled, overwrite=overwrite)


def mock_punch_media(monkeypatch, expected_duration):
    mock_media(monkeypatch, expected_duration)  # shared ffprobe behavior
    commands = []
    def fake_run(command):
        commands.append(command)
        Path(command[-1]).write_bytes(b"mock punch-in video")
    monkeypatch.setattr(punch_in_renderer, "run_command", fake_run)
    return commands


@pytest.mark.parametrize("count,expected", [
    (1, [1.0]), (2, [1.0, 1.08]), (3, [1.0, 1.08, 1.0]),
    (4, [1.0, 1.08, 1.0, 1.08]),
])
def test_style_alternates_exact_canonical_segments(tmp_path, count, expected):
    values = speech_fixture(tmp_path)
    retimed = copy.deepcopy(values[8])
    keeps = retimed["base_video"]["keep_segments"]
    while len(keeps) < count:
        keeps.append(copy.deepcopy(keeps[-1]))
    retimed["base_video"]["keep_segments"] = keeps[:count]
    plan = build_jump_cut_style_plan(retimed)
    assert [item["scale"] for item in plan["segments"]] == expected
    assert [item["edited_start"] for item in plan["segments"]] == [
        item["edited_start"] for item in keeps[:count]]
    assert validate_jump_cut_style_plan(plan, retimed) == plan


@pytest.mark.parametrize("scale", [1, 0.99, 1.201, float("nan"), float("inf")])
def test_invalid_scale_rejected(scale):
    with pytest.raises(ValueError, match="Punch-in scale"):
        validate_punch_in_scale(scale)


def test_styled_timeline_preserves_audio_broll_and_timing(tmp_path):
    values, style, styled = styled_fixture(tmp_path, 1.12)
    retimed = values[8]
    assert [item["scale"] for item in style["segments"]] == [1, 1.12, 1]
    assert styled["duration"] == retimed["duration"]
    assert styled["base_audio"] == retimed["base_audio"]
    assert styled["visual_overlays"] == retimed["visual_overlays"]
    assert [dict(item, scale=None, anchor=None) for item in retimed["base_video"]["keep_segments"]] == [
        dict(item, scale=None, anchor=None) for item in styled["base_video"]["segments"]]
    assert validate_styled_edit_timeline(styled, retimed, style) == styled
    assert build_jump_cut_style_plan(retimed, 1.12) == style
    assert build_styled_edit_timeline(retimed, style) == styled
    bad = copy.deepcopy(styled)
    bad["visual_overlays"][0]["timeline_start"] += 0.1
    with pytest.raises(ValueError, match="changed canonical"):
        validate_styled_edit_timeline(bad, retimed, style)
    bad = copy.deepcopy(style)
    bad["segments"][1]["scale"] = 1.1
    with pytest.raises(ValueError, match="differ"):
        validate_jump_cut_style_plan(bad, retimed)


def test_filter_graph_static_primary_punch_and_separate_publication(tmp_path, monkeypatch):
    values, style, styled = styled_fixture(tmp_path)
    output = tmp_path / "output" / "demo"
    output.mkdir(parents=True)
    (output / "edited_draft.mp4").write_bytes(b"V3D")
    (output / "speech_edited_draft.mp4").write_bytes(b"V4B")
    commands = mock_punch_media(monkeypatch, values[6]["edited_duration"])
    manifest = render(values, style, styled, output)
    assert len(commands) == 1
    command = commands[0]
    graph = command[command.index("-filter_complex") + 1]
    assert command[-1].endswith("punch_in_draft.mp4.part.mp4")
    assert command.count("-i") == 2  # primary plus one repeated B-roll asset
    assert graph.count("scale=2074:-2") == 1
    assert "scale=2074:-2,crop=1920:1080:(iw-1920)/2:(ih-1080)/2" in graph
    assert graph.count("scale=1920:1080:force_original_aspect_ratio=increase") == 7
    assert "[0:0]split=3" in graph and "[0:1]asplit=3" in graph
    assert "concat=n=3:v=1:a=0[base]" in graph
    assert "concat=n=3:v=0:a=1[aout]" in graph
    assert "[1:v:0]split=4" in graph
    assert command[command.index("-map") + 1] == "[vout]"
    assert command[command.index("-map", command.index("-map") + 1) + 1] == "[aout]"
    assert "[1:a" not in graph
    assert command[command.index("-pix_fmt") + 1] == "yuv420p"
    assert command[command.index("-r") + 1] == "30"
    assert command[command.index("-c:v") + 1] == "libx264"
    assert command[command.index("-c:a") + 1] == "aac"
    assert command[command.index("-ar") + 1] == "48000"
    assert "-map_metadata" in command and "-map_chapters" in command and "-dn" in command and "-sn" in command
    assert "+faststart" in command
    assert (output / "edited_draft.mp4").read_bytes() == b"V3D"
    assert (output / "speech_edited_draft.mp4").read_bytes() == b"V4B"
    assert (output / "punch_in_draft.mp4").read_bytes() == b"mock punch-in video"
    assert not (output / "punch_in_draft.mp4.part.mp4").exists()
    assert manifest["punch_in_segment_count"] == 1
    assert manifest["styled_segment_count"] == 3
    assert manifest["styled_broll_overlay_count"] == 4
    assert manifest["sha256"] == hashlib.sha256(b"mock punch-in video").hexdigest()
    assert manifest["jump_cut_style_plan_sha256"] == hashlib.sha256(
        (values[0] / "jump_cut_style_plan.json").read_bytes()).hexdigest()
    assert read_json(values[0] / "punch_in_render_manifest.json") == manifest


def test_punch_output_protection_and_failure_cleanup(tmp_path, monkeypatch):
    values, style, styled = styled_fixture(tmp_path)
    output = tmp_path / "output" / "demo"
    output.mkdir(parents=True)
    (output / "punch_in_draft.mp4").write_bytes(b"earlier")
    (output / "edited_draft.mp4").write_bytes(b"V3D")
    (output / "speech_edited_draft.mp4").write_bytes(b"V4B")
    commands = mock_punch_media(monkeypatch, values[6]["edited_duration"])
    with pytest.raises(FileExistsError, match="--overwrite-render"):
        render(values, style, styled, output)
    assert commands == []
    def fail(command):
        Path(command[-1]).write_bytes(b"partial")
        raise RuntimeError("mock FFmpeg failure")
    monkeypatch.setattr(punch_in_renderer, "run_command", fail)
    with pytest.raises(RuntimeError, match="mock FFmpeg failure"):
        render(values, style, styled, output, overwrite=True)
    assert not (output / "punch_in_draft.mp4.part.mp4").exists()
    assert (output / "punch_in_draft.mp4").read_bytes() == b"earlier"
    mock_punch_media(monkeypatch, values[6]["edited_duration"])
    def bad_probe(path):
        info = fake_probe(path)
        if path.name.endswith(".part.mp4"):
            info["format"]["duration"] = "1"
        return info
    monkeypatch.setattr(edit_renderer, "probe", bad_probe)
    with pytest.raises(ValueError, match="duration differs"):
        render(values, style, styled, output, overwrite=True)
    assert not (output / "punch_in_draft.mp4.part.mp4").exists()
    assert (output / "punch_in_draft.mp4").read_bytes() == b"earlier"
    mock_punch_media(monkeypatch, values[6]["edited_duration"])
    render(values, style, styled, output, overwrite=True)
    assert (output / "punch_in_draft.mp4").read_bytes() == b"mock punch-in video"
    assert (output / "edited_draft.mp4").read_bytes() == b"V3D"
    assert (output / "speech_edited_draft.mp4").read_bytes() == b"V4B"


def test_cli_routes_without_detection_or_network(tmp_path, monkeypatch):
    values, style, styled = styled_fixture(tmp_path)
    config = tmp_path / "config" / "transcription.json"
    write_json(config, {"schema_version": 1, "default_whisper_model": "local/model"})
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    class Forbidden:
        def __init__(self, *_args, **_kwargs):
            raise AssertionError("Forbidden detection, transcription, Codex, or network path")
    for name in ("LocalAudioSilenceDetector", "RealMLXTranscriber", "CodexDirector", "PexelsSourceProvider"):
        monkeypatch.setattr(cli, name, Forbidden)
    calls = []
    class FakeRenderer:
        def render(self, *args, **kwargs):
            calls.append((args, kwargs))
            kwargs["before_render"]()
            kwargs["before_validate"]()
    monkeypatch.setattr(cli, "PunchInRenderer", FakeRenderer)
    before = (values[0] / "retimed_edit_timeline.json").read_bytes()
    assert cli.main(["--project", "demo", "--render-punch-ins", "--punch-in-scale", "1.12"]) == 0
    assert len(calls) == 1
    assert read_json(values[0] / "jump_cut_style_plan.json")["punch_in_scale"] == 1.12
    assert (values[0] / "retimed_edit_timeline.json").read_bytes() == before
    first = ((values[0] / "jump_cut_style_plan.json").read_bytes(),
             (values[0] / "styled_edit_timeline.json").read_bytes())
    assert cli.main(["--project", "demo", "--render-punch-ins", "--punch-in-scale", "1.12"]) == 0
    assert first == ((values[0] / "jump_cut_style_plan.json").read_bytes(),
                     (values[0] / "styled_edit_timeline.json").read_bytes())
    with pytest.raises(SystemExit):
        cli.main(["--project", "demo", "--punch-in-scale", "1.1", "--plan-speech-edits"])
    with pytest.raises(SystemExit):
        cli.main(["--project", "demo", "--punch-in-scale", "1.08", "--plan-speech-edits"])
    with pytest.raises(SystemExit):
        cli.main(["--project", "demo", "--render-punch-ins", "--video", "input.mov"])
