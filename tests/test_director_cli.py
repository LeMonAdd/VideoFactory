from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from videofactory import cli
from videofactory.asset_manager import discover_assets
from videofactory.models import read_json, write_json
from videofactory.progress import ProgressReporter


def prepare_project(root: Path, include_transcript: bool = True) -> Path:
    config = root / "config" / "transcription.json"
    config.parent.mkdir(parents=True)
    config.write_text('{"schema_version":1,"default_whisper_model":"local/test-model"}')
    project = root / "projects" / "demo"
    project.mkdir(parents=True)
    write_json(project / "project.json", {
        "schema_version": 1, "project_name": "demo", "mode": "TALKING_HEAD",
    })
    if include_transcript:
        write_json(project / "transcript.json", {
            "schema_version": 1, "language": "en", "duration": 6.0,
            "segments": [{"start": 2.0, "end": 5.0, "text": "I introduce the topic"}],
        })
    return project


def test_cli_director_defaults_to_offline_rule_and_parses_codex_options():
    normal = cli.parser().parse_args(["--video", "inbox/head.mov", "--project", "demo"])
    assert normal.director == "rule"
    codex = cli.parser().parse_args([
        "--project", "demo", "--director", "codex", "--plan-only",
        "--director-model", "chosen-model", "--director-timeout", "45",
    ])
    assert codex.plan_only and codex.director == "codex"
    assert codex.director_model == "chosen-model" and codex.director_timeout == 45


def test_plan_only_updates_only_director_plan(tmp_path, monkeypatch, capsys):
    project = prepare_project(tmp_path)
    original = {path.name: path.read_bytes() for path in project.iterdir()}
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    assert cli.main(["--project", "demo", "--plan-only"]) == 0
    assert {path.name for path in project.iterdir()} == set(original) | {"director_plan.json"}
    assert all((project / name).read_bytes() == data for name, data in original.items())
    plan = read_json(project / "director_plan.json")
    assert plan["metadata"]["provider"] == "rule"
    assert plan["metadata"]["invocation_count"] == 0
    output = capsys.readouterr().out
    assert "[1/3] Loading transcript..." in output
    assert "[2/3] Creating director plan..." in output
    assert "[3/3] Validating and saving director_plan.json..." in output


def test_plan_only_missing_transcript_reports_error_without_writes(tmp_path, monkeypatch, capsys):
    project = prepare_project(tmp_path, include_transcript=False)
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    assert cli.main(["--project", "demo", "--plan-only"]) == 1
    assert "Existing project and transcript are required" in capsys.readouterr().err
    assert {path.name for path in project.iterdir()} == {"project.json"}


def test_normal_cli_still_requires_source(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["--project", "demo"])
    assert exc.value.code == 2
    assert "--video or --voice is required" in capsys.readouterr().err


def test_progress_reporter_flushes_immediately():
    class Stream(io.StringIO):
        flushes = 0

        def flush(self):
            self.flushes += 1
            super().flush()

    stream = Stream()
    progress = ProgressReporter(8, stream)
    progress.start(3, "Transcribing")
    progress.complete(3, "Transcription")
    assert stream.flushes == 2
    assert "[3/8] Transcribing..." in stream.getvalue()
    assert "[3/8] Transcription complete (" in stream.getvalue()


def test_asset_sidecar_adds_only_explicit_visual_queries(tmp_path, monkeypatch):
    broll = tmp_path / "assets" / "broll"
    broll.mkdir(parents=True)
    media = broll / "tokyo.mp4"
    media.touch()
    (broll / "tokyo.mp4.json").write_text(json.dumps({
        "schema_version": 1,
        "visual_queries": ["modern Tokyo skyline and busy city streets"],
    }))
    monkeypatch.setattr(
        "videofactory.asset_manager.probe",
        lambda path: {"streams": [{"codec_type": "video", "width": 320,
                                    "height": 180, "codec_name": "h264"}],
                      "format": {"duration": "5"}},
    )
    sources = discover_assets(tmp_path, "VOICEOVER")
    assert sources["sources"][0]["visual_queries"] == [
        "modern Tokyo skyline and busy city streets"
    ]
