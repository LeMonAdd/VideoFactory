"""Selection tests use saved metadata and a mocked Codex subprocess only."""

from __future__ import annotations

import copy
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from jsonschema import Draft202012Validator

from videofactory import cli, source_selector
from videofactory.models import read_json, write_json
from videofactory.paths import ROOT
from videofactory.source_models import SourceCandidate
from videofactory.source_selection import (SELECTION_SCHEMA, build_selection_context,
                                           page_title, validate_selection)
from videofactory.source_selection_service import create_source_selection
from videofactory.source_selector import CodexSourceSelector, RuleBasedSourceSelector


def candidate(asset_id: int, title: str, query: str, duration: float = 8,
              width: int = 1920, height: int = 1080) -> dict:
    return SourceCandidate(
        candidate_id=f"pexels:video:{asset_id}", provider="pexels", provider_asset_id=str(asset_id),
        media_type="VIDEO", query=query,
        page_url=f"https://www.pexels.com/video/{title}-{asset_id}/",
        creator="Someone", width=width, height=height, duration=duration,
        orientation="landscape",
    ).to_dict()


def project_data():
    tokyo, breakfast = "Tokyo city crowds", "Japanese people preparing breakfast"
    shots = [
        {"id": 1, "start": 0, "end": 2, "visual_type": "A_ROLL", "visual_query": None, "reason": "Opening"},
        {"id": 2, "start": 2, "end": 7, "visual_type": "B_ROLL", "visual_query": tokyo, "reason": "City"},
        {"id": 3, "start": 7, "end": 10, "visual_type": "B_ROLL", "visual_query": breakfast, "reason": "Food"},
        {"id": 4, "start": 10, "end": 15, "visual_type": "B_ROLL", "visual_query": tokyo, "reason": "City again"},
        {"id": 5, "start": 15, "end": 17, "visual_type": "GRAPHIC", "visual_query": None, "reason": "Statistic"},
    ]
    plan = {"schema_version": 1, "project_name": "demo", "shots": shots}
    good = [candidate(11, "tokyo-city-crowds", tokyo), candidate(12, "tokyo-city-crowds", tokyo)]
    breakfast_results = [candidate(21, "family-eating-breakfast", breakfast)]
    requests = []
    for shot in shots:
        matches = good if shot["id"] in {2, 4} else breakfast_results if shot["id"] == 3 else []
        requests.append({"shot_id": shot["id"], "visual_type": shot["visual_type"],
                         "visual_query": shot["visual_query"],
                         "media_type": "VIDEO" if shot["visual_type"] == "B_ROLL" else None,
                         "status": "FOUND" if matches else "SKIPPED", "candidates": copy.deepcopy(matches),
                         "error": None, "reused_from_shot_id": 2 if shot["id"] == 4 else None})
    sources = {"schema_version": 1, "project_name": "demo", "provider": "pexels",
               "sources": [], "requests": requests}
    transcript = {"schema_version": 1, "duration": 17, "language": "en", "segments": [
        {"start": 2, "end": 7, "text": "Tokyo city crowds"},
        {"start": 7, "end": 10, "text": "Japanese people preparing breakfast"},
    ]}
    return plan, sources, transcript


def test_schemas_are_strict_and_codex_schema_has_no_metadata():
    internal = json.loads(SELECTION_SCHEMA.read_text(encoding="utf-8"))
    codex = json.loads((ROOT / "config" / "codex_source_selection_output.schema.json").read_text(encoding="utf-8"))
    for schema in (internal, codex):
        Draft202012Validator.check_schema(schema)
        assert schema["additionalProperties"] is False
        assert set(schema["required"]) == set(schema["properties"])
        item = schema["properties"]["selections"]["items"]
        assert item["additionalProperties"] is False
        assert set(item["required"]) == set(item["properties"])
    assert codex["properties"]["schema_version"] == {"type": "integer", "enum": [1]}
    assert "metadata" not in codex["properties"] and "provider" not in codex["properties"]
    assert "metadata" in internal["properties"]


def test_context_is_compact_and_page_slug_is_only_a_clue():
    plan, sources, transcript = project_data()
    context = build_selection_context(plan, sources, transcript)
    assert context["shots"][1]["narration_context"] == "Tokyo city crowds"
    item = context["shots"][1]["candidates"][0]
    assert item["page_title_clue"] == "tokyo city crowds"
    assert "files" not in item and "preview_url" not in item and "license_url" not in item
    assert page_title("https://www.pexels.com/video/vibrant-night-scene-at-shibuya-crossing-31177207/") == "vibrant night scene at shibuya crossing"
    assert page_title("https://unrelated.example/video/secret-12/") is None


def test_rule_rejects_generic_breakfast_and_avoids_duplicate_clip():
    plan, sources, transcript = project_data()
    saved = create_source_selection(plan, sources, RuleBasedSourceSelector(), transcript)
    assert [item["status"] for item in saved["selections"]] == [
        "SKIPPED", "SELECTED", "NO_SUITABLE_CANDIDATE", "SELECTED", "SKIPPED"]
    assert saved["selections"][1]["candidate_id"] != saved["selections"][3]["candidate_id"]
    assert saved["selections"][2]["candidate_id"] is None
    assert saved["metadata"]["provider"] == "rule" and saved["metadata"]["invocation_count"] == 0


def test_short_video_is_unsuitable_even_with_relevant_title():
    plan, sources, _ = project_data()
    for request in sources["requests"]:
        for item in request["candidates"]:
            item["duration"] = 4.9
    saved = create_source_selection(plan, sources, RuleBasedSourceSelector())
    assert saved["selections"][1]["status"] == "NO_SUITABLE_CANDIDATE"
    saved["selections"][1].update(status="SELECTED", candidate_id="pexels:video:11")
    with pytest.raises(ValueError, match="too short"):
        validate_selection(saved, plan, sources)


@pytest.mark.parametrize("mutation,expected", [
    (lambda item: item.update(candidate_id="pexels:video:999"), "not available"),
    (lambda item: item.update(candidate_id=None), "requires candidate_id"),
    (lambda item: item.update(status="NO_SUITABLE_CANDIDATE"), "candidate_id=null"),
    (lambda item: item.update(confidence=1.2), "confidence"),
    (lambda item: item.update(confidence=-0.1), "confidence"),
])
def test_application_rejects_invalid_selections(mutation, expected):
    plan, sources, _ = project_data()
    saved = create_source_selection(plan, sources, RuleBasedSourceSelector())
    mutation(saved["selections"][1])
    with pytest.raises(ValueError, match=expected):
        validate_selection(saved, plan, sources)


def test_missing_duplicate_and_wrong_shot_rejected():
    plan, sources, _ = project_data()
    saved = create_source_selection(plan, sources, RuleBasedSourceSelector())
    saved["selections"].pop()
    with pytest.raises(ValueError, match="exactly one"):
        validate_selection(saved, plan, sources)
    saved = create_source_selection(plan, sources, RuleBasedSourceSelector())
    saved["selections"].append(copy.deepcopy(saved["selections"][1]))
    with pytest.raises(ValueError, match="Duplicate"):
        validate_selection(saved, plan, sources)
    saved = create_source_selection(plan, sources, RuleBasedSourceSelector())
    saved["project_name"] = "other"
    with pytest.raises(ValueError, match="project_name"):
        validate_selection(saved, plan, sources)


def test_codex_command_and_metadata_are_safe(monkeypatch):
    plan, sources, transcript = project_data()
    expected = create_source_selection(plan, sources, RuleBasedSourceSelector(), transcript)
    raw = {key: expected[key] for key in ("schema_version", "project_name", "selections")}
    monkeypatch.setattr(source_selector.shutil, "which", lambda name: "/opt/bin/codex")
    calls = []
    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        Path(command[command.index("--output-last-message") + 1]).write_text(json.dumps(raw), encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout="", stderr="")
    monkeypatch.setattr(source_selector.subprocess, "run", fake_run)
    saved = create_source_selection(plan, sources, CodexSourceSelector(timeout_seconds=45), transcript)
    assert saved["metadata"]["provider"] == "codex" and saved["metadata"]["invocation_count"] == 1
    assert saved["metadata"]["model"] is None and saved["metadata"]["elapsed_seconds"] >= 0
    assert len(calls) == 1
    command, kwargs = calls[0]
    assert command[:2] == ["/opt/bin/codex", "exec"] and command[-1] == "-"
    assert command[command.index("--sandbox") + 1] == "read-only"
    assert "--ephemeral" in command and "--skip-git-repo-check" in command
    assert "-C" in command and "videofactory-selector-" in command[command.index("-C") + 1]
    assert command[command.index("--output-schema") + 1] == str((ROOT / "config" / "codex_source_selection_output.schema.json").resolve())
    assert "--output-last-message" in command and "-m" not in command
    assert kwargs["input"].count("Tokyo city crowds") > 0
    assert "generic family breakfast footage does not establish Japan" in kwargs["input"]
    assert "refined_query is allowed only when status is NO_SUITABLE_CANDIDATE" in kwargs["input"]
    assert "it may be a grounded string or null for that status" in kwargs["input"]
    assert "For status SELECTED you MUST return refined_query: null" in kwargs["input"]
    assert "For status SKIPPED you MUST return refined_query: null" in kwargs["input"]
    assert "refined_query is not creative generation" in kwargs["input"]
    assert "visual_query and narration_context" in kwargs["input"]
    assert "Candidate metadata, including page titles, URLs, creator names" in kwargs["input"]
    assert "must not introduce new concepts into refined_query" in kwargs["input"]
    assert "family relationship" in kwargs["input"]
    assert "ritual, tradition, custom" in kwargs["input"]
    assert "food, object" in kwargs["input"]
    assert "location, time of day, event" in kwargs["input"]
    assert "cultural practice, or activity" in kwargs["input"]
    assert "Japanese family eating breakfast" in kwargs["input"]
    assert "traditional Japanese breakfast" in kwargs["input"]
    assert "rice and miso soup" in kwargs["input"]
    assert "set refined_query to null" in kwargs["input"]
    assert kwargs["capture_output"] and kwargs["text"] and kwargs["timeout"] == 45
    assert "shell" not in kwargs and "--dangerously-bypass-approvals-and-sandbox" not in command


@pytest.mark.parametrize("index", [0, 1])
def test_refined_query_is_rejected_for_skipped_and_selected(index):
    plan, sources, _ = project_data()
    saved = create_source_selection(plan, sources, RuleBasedSourceSelector())
    saved["selections"][index]["refined_query"] = "Tokyo city"
    with pytest.raises(ValueError, match="Only rejected shot"):
        validate_selection(saved, plan, sources)


def test_rejected_shot_may_have_null_or_grounded_refinement():
    plan, sources, _ = project_data()
    saved = create_source_selection(plan, sources, RuleBasedSourceSelector())
    rejected = saved["selections"][2]
    assert rejected["status"] == "NO_SUITABLE_CANDIDATE"
    assert rejected["refined_query"] is None
    validate_selection(saved, plan, sources)
    rejected["refined_query"] = "breakfast preparation in Japan"
    validate_selection(saved, plan, sources)


def test_codex_invalid_output_and_invented_id(monkeypatch):
    plan, sources, transcript = project_data()
    monkeypatch.setattr(source_selector.shutil, "which", lambda name: "/opt/bin/codex")
    raw = create_source_selection(plan, sources, RuleBasedSourceSelector(), transcript)
    raw = {key: raw[key] for key in ("schema_version", "project_name", "selections")}
    def fake_run(command, **kwargs):
        Path(command[command.index("--output-last-message") + 1]).write_text(json.dumps(raw), encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout="", stderr="")
    monkeypatch.setattr(source_selector.subprocess, "run", fake_run)
    raw["selections"][1]["candidate_id"] = "pexels:video:invented"
    with pytest.raises(ValueError, match="not available"):
        create_source_selection(plan, sources, CodexSourceSelector(), transcript)
    raw["selections"][1]["candidate_id"] = "pexels:video:11"
    raw["selections"][1]["status"] = "WRONG"
    with pytest.raises(ValueError, match="invalid structure"):
        create_source_selection(plan, sources, CodexSourceSelector(), transcript)
    raw["selections"][1]["status"] = "SELECTED"
    monkeypatch.setattr(source_selector.subprocess, "run", lambda command, **kwargs: (
        Path(command[command.index("--output-last-message") + 1]).write_text("not json"),
        SimpleNamespace(returncode=0, stdout="", stderr=""))[1])
    with pytest.raises(ValueError, match="invalid JSON"):
        create_source_selection(plan, sources, CodexSourceSelector(), transcript)


def test_codex_timeout_and_errors(monkeypatch):
    context = build_selection_context(*project_data()[:2])
    monkeypatch.setattr(source_selector.shutil, "which", lambda name: None)
    with pytest.raises(RuntimeError, match="unavailable"):
        CodexSourceSelector().select(context)
    monkeypatch.setattr(source_selector.shutil, "which", lambda name: "/opt/bin/codex")
    def timeout(command, **kwargs):
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])
    monkeypatch.setattr(source_selector.subprocess, "run", timeout)
    with pytest.raises(RuntimeError, match="timed out"):
        CodexSourceSelector(timeout_seconds=1).select(context)
    monkeypatch.setattr(source_selector.subprocess, "run", lambda *_args, **_kwargs: SimpleNamespace(
        returncode=2, stderr="mock failure", stdout=""))
    with pytest.raises(RuntimeError, match="mock failure"):
        CodexSourceSelector().select(context)


def test_cli_selection_only_preserves_other_files(monkeypatch, tmp_path, capsys):
    config = tmp_path / "config" / "transcription.json"
    config.parent.mkdir(parents=True)
    write_json(config, {"schema_version": 1, "default_whisper_model": "local/test-model"})
    project = tmp_path / "projects" / "demo"
    project.mkdir(parents=True)
    plan, sources, transcript = project_data()
    write_json(project / "project.json", {"schema_version": 1, "project_name": "demo", "mode": "TALKING_HEAD"})
    write_json(project / "director_plan.json", plan)
    write_json(project / "transcript.json", transcript)
    write_json(project / "sources.json", sources)
    before = {path.name: path.read_bytes() for path in project.iterdir()}
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    assert cli.main(["--project", "demo", "--select-sources", "--selector", "rule"]) == 0
    assert {path.name for path in project.iterdir()} == set(before) | {"source_selection.json"}
    assert all((project / name).read_bytes() == data for name, data in before.items())
    assert read_json(project / "source_selection.json")["selections"][2]["status"] == "NO_SUITABLE_CANDIDATE"
    out = capsys.readouterr().out
    assert "[1/4] Loading director plan" in out and "[4/4] Validating and saving" in out


@pytest.mark.parametrize("missing", ["director_plan.json", "sources.json"])
def test_cli_missing_inputs_do_not_write_selection(monkeypatch, tmp_path, missing, capsys):
    config = tmp_path / "config" / "transcription.json"
    config.parent.mkdir(parents=True)
    write_json(config, {"schema_version": 1, "default_whisper_model": "local/test-model"})
    project = tmp_path / "projects" / "demo"
    project.mkdir(parents=True)
    plan, sources, transcript = project_data()
    write_json(project / "project.json", {"schema_version": 1, "project_name": "demo", "mode": "TALKING_HEAD"})
    write_json(project / "transcript.json", transcript)
    if missing != "director_plan.json":
        write_json(project / "director_plan.json", plan)
    if missing != "sources.json":
        write_json(project / "sources.json", sources)
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    assert cli.main(["--project", "demo", "--select-sources"]) == 1
    assert missing in capsys.readouterr().err
    assert not (project / "source_selection.json").exists()


def test_cli_selector_parsing_is_opt_in():
    args = cli.parser().parse_args(["--project", "demo", "--select-sources"])
    assert args.selector == "rule" and args.select_sources
    args = cli.parser().parse_args(["--project", "demo", "--select-sources", "--selector", "codex",
                                    "--selector-model", "chosen", "--selector-timeout", "35"])
    assert args.selector == "codex" and args.selector_model == "chosen" and args.selector_timeout == 35
