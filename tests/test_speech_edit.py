"""V4A planning tests use saved JSON shapes and never inspect or render media."""

from __future__ import annotations

import copy
import pytest

from test_edit_timeline import fixture as planning_fixture
from videofactory import cli
from videofactory.clip_planner import build_clip_plan
from videofactory.edit_timeline import build_edit_timeline
from videofactory.models import read_json, write_json
from videofactory.speech_edit import (SpeechEditPlanner, build_retime_map, map_source_time,
                                      validate_retime_map, validate_speech_edit_plan)


def inputs(tmp_path):
    project_dir, project, director, sources, selection, manifest = planning_fixture(tmp_path)
    clips = build_clip_plan(project_dir, "TALKING_HEAD", 20, director, sources, selection, manifest)
    timeline = build_edit_timeline(project, 20, director, clips, project_dir)
    transcript = {"schema_version": 1, "language": "en", "duration": 20,
                  "segments": [{"start": 0.5, "end": 19.2, "text": "fixture",
                                "words": [{"start": start, "end": end, "text": "word"} for start, end in
                                          ((0.5, 1), (1.4, 1.8), (3, 3.4), (5, 5.4),
                                           (10, 10.3), (19, 19.2))]}]}
    return project_dir, project, transcript, timeline


def silences():
    return [(0.0, 0.4), (1.8, 3.0), (3.4, 5.0), (5.4, 10.0),
            (10.3, 19.0), (19.3, 20.0)]


def test_cuts_keeps_durations_and_overlay_diagnostics(tmp_path):
    _, project, transcript, timeline = inputs(tmp_path)
    original_timeline = copy.deepcopy(timeline)
    plan = SpeechEditPlanner().plan(project, transcript, timeline, silences())
    assert [(cut["source_start"], cut["source_end"]) for cut in plan["cuts"]] == [
        (1.925, 2.875), (3.525, 4.875), (5.525, 9.875), (10.425, 18.875)]
    assert [cut["affected_overlay_shot_ids"] for cut in plan["cuts"]] == [[], [2], [2], [7]]
    assert plan["total_removed_duration"] == 15.1
    assert plan["edited_duration"] == 4.9
    assert plan["keep_segments"][0] == {"source_start": 0, "source_end": 1.925,
                                         "source_duration": 1.925, "edited_start": 0,
                                         "edited_end": 1.925}
    assert plan["keep_segments"][-1]["source_end"] == 20
    assert plan["keep_segments"][-1]["edited_end"] == 4.9
    assert timeline == original_timeline
    assert validate_speech_edit_plan(plan, project, transcript, timeline, silences()) == plan
    assert plan["detection"] == {"method": "ffmpeg_silencedetect", "noise_db": -35.0}
    assert plan["cuts"][0]["detected_silence_start"] == 1.8
    assert validate_retime_map(build_retime_map(plan), plan)


def test_threshold_retention_padding_and_words(tmp_path):
    _, project, transcript, timeline = inputs(tmp_path)
    plan = SpeechEditPlanner().plan(project, transcript, timeline, silences(), threshold=1.2, keep=0.4)
    assert len(plan["cuts"]) == 4  # 1.2-second gap meets the threshold
    first = plan["cuts"][0]
    assert first["source_start"] == 2.0 and first["source_end"] == 2.8
    assert first["removed_duration"] == 0.8
    assert first["source_start"] - 1.8 == pytest.approx(0.2)
    assert 3.0 - first["source_end"] == pytest.approx(0.2)
    assert all(cut["detected_silence_start"] < cut["source_start"] < cut["source_end"] < cut["detected_silence_end"]
               for cut in plan["cuts"])
    zero_keep = SpeechEditPlanner().plan(project, transcript, timeline, silences(), keep=0)
    assert zero_keep["cuts"][0]["source_start"] == 1.84
    assert zero_keep["cuts"][0]["source_end"] == 2.96


def test_gap_below_threshold_and_no_cuts(tmp_path):
    _, project, transcript, timeline = inputs(tmp_path)
    plan = SpeechEditPlanner().plan(project, transcript, timeline, silences(), threshold=9)
    assert plan["cuts"] == []
    assert plan["keep_segments"] == [{"source_start": 0, "source_end": 20,
                                      "source_duration": 20, "edited_start": 0, "edited_end": 20}]


def test_multiple_opening_regions_tail_and_stretched_word(tmp_path):
    _, project, transcript, timeline = inputs(tmp_path)
    transcript["segments"][0].update(start=3.5, end=18.5, words=[
        {"start": a, "end": b, "text": "word"} for a, b in
        ((3.5, 4), (5, 5.5), (10, 10.5), (18, 18.5))])
    acoustic = [(0, 1.2), (1.21, 3.3), (6, 7.3), (18.6, 20)]
    plan = SpeechEditPlanner().plan(project, transcript, timeline, acoustic)
    assert len(plan["cuts"]) == 1
    assert plan["cuts"][0]["source_start"] == 6.125
    assert plan["cuts"][0]["source_end"] == 7.175
    assert plan["keep_segments"][0]["source_start"] == 0
    assert plan["keep_segments"][-1]["source_end"] == 20

    transcript["segments"][0].update(start=0.5, end=19.2, words=[
        {"start": a, "end": b, "text": "word"} for a, b in
        ((0.5, 1), (1.4, 1.8), (3, 3.4), (5, 10), (10, 10.3), (19, 19.2))])
    plan = SpeechEditPlanner().plan(project, transcript, timeline, silences())
    assert any(cut["detected_silence_start"] == 5.4 for cut in plan["cuts"])
    assert validate_speech_edit_plan(plan, project, transcript, timeline, silences())


def test_detected_silence_below_threshold_not_cut(tmp_path):
    _, project, transcript, timeline = inputs(tmp_path)
    plan = SpeechEditPlanner().plan(project, transcript, timeline, [(2, 2.999)])
    assert plan["cuts"] == []
    rounded_up = SpeechEditPlanner().plan(project, transcript, timeline, [(2, 2.9996)])
    assert rounded_up["cuts"] == []


def test_mapping_boundaries_and_cut_interior(tmp_path):
    _, project, transcript, timeline = inputs(tmp_path)
    mapping = build_retime_map(SpeechEditPlanner().plan(project, transcript, timeline, silences()))
    assert map_source_time(0, mapping) == 0
    assert map_source_time(1, mapping) == 1
    assert map_source_time(1.925, mapping) == 1.925
    assert map_source_time(2.875, mapping) == 1.925
    assert map_source_time(3, mapping) == 2.05
    assert map_source_time(5, mapping) == 2.7
    assert map_source_time(20, mapping) == 4.9
    with pytest.raises(ValueError, match="removed region"):
        map_source_time(2, mapping)
    with pytest.raises(ValueError, match="removed region"):
        map_source_time(1.9251, mapping)
    assert map_source_time(2, mapping, bias="left") == 1.925
    assert map_source_time(2, mapping, bias="right") == 1.925
    with pytest.raises(ValueError, match="outside"):
        map_source_time(-0.1, mapping)
    with pytest.raises(ValueError, match="Bias"):
        map_source_time(2, mapping, bias="middle")


@pytest.mark.parametrize("mutation", [
    lambda t: t["segments"][0]["words"][1].update(start=0.9),
    lambda t: t["segments"][0]["words"][1].update(end=0.9),
    lambda t: t["segments"][0]["words"][1].update(start=-0.1),
    lambda t: t["segments"][0]["words"][1].update(start=float("nan")),
    lambda t: t["segments"][0]["words"][1].update(end=float("inf")),
    lambda t: t["segments"][0].pop("words"),
    lambda t: t["segments"][0]["words"][1].pop("start"),
])
def test_invalid_word_timing_rejected(tmp_path, mutation):
    _, project, transcript, timeline = inputs(tmp_path)
    mutation(transcript)
    with pytest.raises(ValueError):
        SpeechEditPlanner().plan(project, transcript, timeline, silences())


@pytest.mark.parametrize("threshold,keep", [(0, 0), (-1, 0), (1, 1), (1, -0.1),
                                            (float("nan"), 0), (1, float("inf"))])
def test_invalid_parameters_rejected(tmp_path, threshold, keep):
    _, project, transcript, timeline = inputs(tmp_path)
    with pytest.raises(ValueError):
        SpeechEditPlanner().plan(project, transcript, timeline, silences(), threshold, keep)


def test_duration_name_and_plan_corruption_rejected(tmp_path):
    _, project, transcript, timeline = inputs(tmp_path)
    bad_timeline = copy.deepcopy(timeline)
    bad_timeline["duration"] = 21
    with pytest.raises(ValueError, match="duration mismatch"):
        SpeechEditPlanner().plan(project, transcript, bad_timeline, silences())
    bad_timeline = copy.deepcopy(timeline)
    bad_timeline["project_name"] = "other"
    with pytest.raises(ValueError, match="project_name"):
        SpeechEditPlanner().plan(project, transcript, bad_timeline, silences())
    plan = SpeechEditPlanner().plan(project, transcript, timeline, silences())
    corrupt = copy.deepcopy(plan)
    corrupt["cuts"][1]["source_start"] = 1.5
    with pytest.raises(ValueError, match="ordered"):
        validate_speech_edit_plan(corrupt, project, transcript, timeline, silences())
    corrupt = copy.deepcopy(plan)
    corrupt["keep_segments"][0]["edited_end"] = 2
    with pytest.raises(ValueError, match="Keep segments"):
        validate_speech_edit_plan(corrupt, project, transcript, timeline, silences())
    corrupt = copy.deepcopy(plan)
    corrupt["total_removed_duration"] = 1
    with pytest.raises(ValueError, match="totals"):
        validate_speech_edit_plan(corrupt, project, transcript, timeline, silences())
    corrupt = copy.deepcopy(plan)
    corrupt["cuts"][0].update(source_start=0, source_end=0.95, removed_duration=0.95)
    with pytest.raises(ValueError, match="internal detected silence"):
        validate_speech_edit_plan(corrupt, project, transcript, timeline, silences())
    mapping = build_retime_map(plan)
    corrupt_map = copy.deepcopy(mapping)
    corrupt_map["segments"][1]["edited_start"] = 0
    with pytest.raises(ValueError):
        validate_retime_map(corrupt_map, plan)


def test_cli_is_offline_deterministic_and_does_not_modify_timeline(tmp_path, monkeypatch, capsys):
    project_dir, project, transcript, timeline = inputs(tmp_path)
    for name, document in (("project.json", project), ("transcript.json", transcript),
                           ("edit_timeline.json", timeline)):
        write_json(project_dir / name, document)
    config = tmp_path / "config" / "transcription.json"
    write_json(config, {"schema_version": 1, "default_whisper_model": "local/model"})
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    def forbidden(*_args, **_kwargs):
        raise AssertionError("External media, Codex, network, or transcription path invoked")
    monkeypatch.setattr(cli, "render", forbidden)
    monkeypatch.setattr(cli, "EditRenderer", forbidden)
    monkeypatch.setattr(cli, "CodexDirector", forbidden)
    monkeypatch.setattr(cli, "PexelsSourceProvider", forbidden)
    monkeypatch.setattr(cli, "RealMLXTranscriber", forbidden)
    detected = []
    class FakeDetector:
        def detect(self, primary, duration, minimum, noise):
            detected.append((primary, duration, minimum, noise))
            return silences()
    monkeypatch.setattr(cli, "LocalAudioSilenceDetector", FakeDetector)
    before_timeline = (project_dir / "edit_timeline.json").read_bytes()
    command = ["--project", "demo", "--plan-speech-edits", "--pause-threshold-seconds", "1.2",
               "--pause-keep-seconds", "0.4", "--silence-noise-db", "-42"]
    assert cli.main(command) == 0
    plan_path = project_dir / "speech_edit_plan.json"
    map_path = project_dir / "retime_map.json"
    first = (plan_path.read_bytes(), map_path.read_bytes())
    assert cli.main(command) == 0
    assert first == (plan_path.read_bytes(), map_path.read_bytes())
    assert (project_dir / "edit_timeline.json").read_bytes() == before_timeline
    assert read_json(plan_path)["pause_threshold_seconds"] == 1.2
    assert read_json(plan_path)["detection"]["noise_db"] == -42.0
    assert len(detected) == 2 and detected[0][2:] == (1.2, -42.0)
    assert "pause 1:" in capsys.readouterr().out


def test_cli_rejects_invalid_options_and_missing_input(tmp_path, monkeypatch):
    config = tmp_path / "config" / "transcription.json"
    write_json(config, {"schema_version": 1, "default_whisper_model": "local/model"})
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    with pytest.raises(SystemExit):
        cli.main(["--project", "demo", "--plan-speech-edits", "--pause-keep-seconds", "1"])
    with pytest.raises(FileNotFoundError, match="project.json"):
        cli.run_plan_speech_edits(cli.parser().parse_args(["--project", "demo", "--plan-speech-edits"]))
