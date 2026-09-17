from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from jsonschema import Draft202012Validator

from videofactory import director_provider
from videofactory.director_plan import SCHEMA_PATH, validate_plan
from videofactory.director_provider import CodexDirector
from videofactory.director_service import create_director_plan
from videofactory.paths import ROOT


def context():
    return {
        "project_name": "demo", "mode": "TALKING_HEAD", "language": "ru",
        "duration": 5.0,
        "transcript": [{"start": 0, "end": 5, "text": "Токио"}],
        "editorial_rules": ["Keep presenter visible"],
        "response_shape": {"schema_version": 1, "project_name": "demo", "shots": []},
    }


def result_plan():
    return {"schema_version": 1, "project_name": "demo", "shots": [
        {"id": 1, "start": 0, "end": 5, "visual_type": "A_ROLL",
         "visual_query": None, "reason": "Opening"},
    ]}


def test_codex_schema_is_structure_only_and_strict():
    schema_path = ROOT / "config" / "codex_director_output.schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    assert schema["properties"]["schema_version"] == {"type": "integer", "enum": [1]}
    assert set(schema["required"]) == set(schema["properties"])
    shot_schema = schema["properties"]["shots"]["items"]
    assert set(shot_schema["required"]) == set(shot_schema["properties"])
    assert shot_schema["properties"]["visual_query"] == {"type": ["string", "null"]}
    assert "oneOf" not in shot_schema["properties"]["visual_query"]
    assert "metadata" not in schema["properties"]
    assert schema["additionalProperties"] is False
    assert shot_schema["additionalProperties"] is False
    assert list(Draft202012Validator(schema).iter_errors(result_plan())) == []
    with_metadata = result_plan() | {"metadata": {"provider": "codex"}}
    assert list(Draft202012Validator(schema).iter_errors(with_metadata))
    internal = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    assert internal["properties"]["schema_version"] == {"type": "integer", "const": 1}


def test_codex_command_is_one_read_only_ephemeral_invocation(monkeypatch):
    monkeypatch.setattr(director_provider.shutil, "which", lambda name: "/opt/bin/codex")
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        output = Path(command[command.index("--output-last-message") + 1])
        output.write_text(json.dumps(result_plan()), encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(director_provider.subprocess, "run", fake_run)
    result = CodexDirector(timeout_seconds=42).create(context())
    assert len(calls) == 1
    command, kwargs = calls[0]
    assert command[:2] == ["/opt/bin/codex", "exec"]
    assert command[-1] == "-"
    assert "--ephemeral" in command
    assert command[command.index("--sandbox") + 1] == "read-only"
    assert "--output-schema" in command and "--output-last-message" in command
    assert command[command.index("--output-schema") + 1] == str(
        (ROOT / "config" / "codex_director_output.schema.json").resolve()
    )
    assert "--skip-git-repo-check" in command
    assert "-C" in command and "videofactory-director-" in command[command.index("-C") + 1]
    assert "--dangerously-bypass-approvals-and-sandbox" not in command
    assert "--approve-for-me" not in command
    assert "-m" not in command
    assert kwargs["timeout"] == 42
    assert kwargs["input"].count("Токио") == 1
    assert "Ground every visual_query in the actual narration" in kwargs["input"]
    assert "never invent a specific event, ceremony, person, object, location, food" in kwargs["input"]
    assert "Do not narrow a generic concept into a culturally plausible specific example" in kwargs["input"]
    assert "without changing the narration's semantic scope" in kwargs["input"]
    assert "only entities and actions explicitly mentioned or strongly and directly implied" in kwargs["input"]
    assert "'traditional Japanese cultural customs'" in kwargs["input"]
    assert "'Japanese tea ceremony'" in kwargs["input"]
    assert "'Shinto shrine ritual'" in kwargs["input"]
    assert "prefer A_ROLL in TALKING_HEAD" in kwargs["input"]
    assert kwargs["capture_output"] is True and kwargs["text"] is True
    assert result.plan == result_plan()
    assert result.invocation_count == 1 and result.model is None


def test_videofactory_adds_metadata_after_codex_output(monkeypatch):
    monkeypatch.setattr(director_provider.shutil, "which", lambda name: "/opt/bin/codex")

    def fake_run(command, **kwargs):
        Path(command[command.index("--output-last-message") + 1]).write_text(
            json.dumps(result_plan()), encoding="utf-8"
        )
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(director_provider.subprocess, "run", fake_run)
    transcript = {
        "schema_version": 1, "language": "ru", "duration": 5.0,
        "segments": [{"start": 0.0, "end": 5.0, "text": "Токио"}],
    }
    saved = create_director_plan("demo", "TALKING_HEAD", transcript, CodexDirector())
    assert saved["metadata"]["provider"] == "codex"
    assert saved["metadata"]["model"] is None
    assert saved["metadata"]["invocation_count"] == 1
    assert saved["metadata"]["elapsed_seconds"] >= 0
    assert "metadata" not in result_plan()
    validate_plan(saved, "demo", 5.0, "TALKING_HEAD")


@pytest.mark.parametrize(
    ("shots", "message"),
    [
        ([{"id": 1, "start": 0, "end": 6, "visual_type": "A_ROLL",
           "visual_query": None, "reason": "Too long"}], "exceeds"),
        ([{"id": 1, "start": 0, "end": 3, "visual_type": "A_ROLL",
           "visual_query": None, "reason": "First"},
          {"id": 2, "start": 2, "end": 5, "visual_type": "A_ROLL",
           "visual_query": None, "reason": "Overlap"}], "overlaps"),
        ([{"id": 1, "start": 0, "end": 5, "visual_type": "B_ROLL",
           "visual_query": None, "reason": "Missing query"}], "requires a visual_query"),
    ],
)
def test_application_still_rejects_semantically_invalid_codex_output(monkeypatch, shots, message):
    monkeypatch.setattr(director_provider.shutil, "which", lambda name: "/opt/bin/codex")

    def fake_run(command, **kwargs):
        Path(command[command.index("--output-last-message") + 1]).write_text(
            json.dumps({"schema_version": 1, "project_name": "demo", "shots": shots}),
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(director_provider.subprocess, "run", fake_run)
    transcript = {
        "schema_version": 1, "language": "ru", "duration": 5.0,
        "segments": [{"start": 0.0, "end": 5.0, "text": "Токио"}],
    }
    with pytest.raises(ValueError, match=message):
        create_director_plan("demo", "TALKING_HEAD", transcript, CodexDirector())


def test_codex_explicit_model_and_invalid_json(monkeypatch):
    monkeypatch.setattr(director_provider.shutil, "which", lambda name: "/opt/bin/codex")

    def fake_run(command, **kwargs):
        assert command[command.index("-m") + 1] == "configured-model"
        Path(command[command.index("--output-last-message") + 1]).write_text("{not json")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(director_provider.subprocess, "run", fake_run)
    with pytest.raises(ValueError, match="invalid JSON"):
        CodexDirector(model="configured-model").create(context())


def test_codex_timeout_and_nonzero_exit_are_useful(monkeypatch):
    monkeypatch.setattr(director_provider.shutil, "which", lambda name: "/opt/bin/codex")

    def timed_out(command, **kwargs):
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    monkeypatch.setattr(director_provider.subprocess, "run", timed_out)
    with pytest.raises(RuntimeError, match="timed out"):
        CodexDirector(timeout_seconds=1).create(context())

    monkeypatch.setattr(
        director_provider.subprocess, "run",
        lambda command, **kwargs: SimpleNamespace(returncode=2, stdout="", stderr="auth unavailable"),
    )
    with pytest.raises(RuntimeError, match="auth unavailable"):
        CodexDirector().create(context())


def test_codex_missing_executable_or_result_fails(monkeypatch):
    monkeypatch.setattr(director_provider.shutil, "which", lambda name: None)
    with pytest.raises(RuntimeError, match="unavailable"):
        CodexDirector().create(context())
    monkeypatch.setattr(director_provider.shutil, "which", lambda name: "/opt/bin/codex")
    monkeypatch.setattr(
        director_provider.subprocess, "run",
        lambda command, **kwargs: SimpleNamespace(returncode=0, stdout="", stderr=""),
    )
    with pytest.raises(RuntimeError, match="no final JSON"):
        CodexDirector().create(context())
