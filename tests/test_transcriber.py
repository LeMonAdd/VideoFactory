"""MLX command tests use a fake process and never import MLX or access Metal."""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from videofactory import transcriber
from videofactory import cli


@pytest.mark.parametrize(
    ("offline_value", "language"),
    [(None, None), ("0", "ru"), ("1", None)],
)
def test_real_mlx_inherits_hugging_face_environment_without_forcing_offline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    offline_value: str | None, language: str | None,
) -> None:
    executable = tmp_path / ".venv" / "bin" / "mlx_whisper"
    executable.parent.mkdir(parents=True)
    executable.touch()
    monkeypatch.setattr(transcriber, "ROOT", tmp_path)
    if offline_value is None:
        monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    else:
        monkeypatch.setenv("HF_HUB_OFFLINE", offline_value)
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "user-setting")
    calls: list[tuple[list[str], dict]] = []

    def fake_run(command: list[str], **kwargs: object) -> SimpleNamespace:
        calls.append((command, kwargs))
        output_dir = Path(command[command.index("--output-dir") + 1])
        output_name = command[command.index("--output-name") + 1]
        (output_dir / f"{output_name}.json").write_text(
            '{"language":"ru","segments":[{"start":0,"end":2,"text":"Привет",'
            '"words":[{"start":0.1,"end":0.8,"word":"Привет"}]}]}',
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(transcriber.subprocess, "run", fake_run)
    result = transcriber.RealMLXTranscriber(
        tmp_path / "work", model="mlx-community/whisper-large-v3-turbo",
        language=language,
    ).transcribe(tmp_path / "audio.wav", 2.0)

    command, kwargs = calls[0]
    assert command[0] == str(executable)
    assert command[1] == str(tmp_path / "audio.wav")
    assert command[command.index("--model") + 1] == "mlx-community/whisper-large-v3-turbo"
    assert command[command.index("--output-format") + 1] == "json"
    assert command[command.index("--word-timestamps") + 1] == "True"
    if language:
        assert command[command.index("--language") + 1] == language
    else:
        assert "--language" not in command
    assert "env" not in kwargs  # subprocess inherits the user's environment as-is.
    assert os.environ.get("HF_HUB_OFFLINE") == offline_value
    assert os.environ["TRANSFORMERS_OFFLINE"] == "user-setting"
    assert result["language"] == "ru"
    assert result["segments"][0]["start"] == 0
    assert result["segments"][0]["end"] == 2
    assert result["segments"][0]["words"][0] == {
        "start": 0.1, "end": 0.8, "text": "Привет"
    }


def test_cli_uses_configured_model_and_auto_language(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = tmp_path / "config" / "transcription.json"
    config.parent.mkdir()
    config.write_text('{"schema_version":1,"default_whisper_model":"local/custom-model"}')
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    args = cli.parser().parse_args(["--video", "inbox/test.mov", "--project", "demo"])
    assert args.whisper_model == "local/custom-model"
    assert args.language is None


def test_cli_model_override_and_russian_language() -> None:
    args = cli.parser().parse_args([
        "--video", "inbox/test.mov", "--project", "demo",
        "--whisper-model", "mlx-community/whisper-large-v3-turbo", "--language", "ru",
    ])
    assert args.whisper_model == "mlx-community/whisper-large-v3-turbo"
    assert args.language == "ru"


def test_legacy_mlx_model_alias_remains_available() -> None:
    args = cli.parser().parse_args([
        "--voice", "inbox/voice.wav", "--project", "demo",
        "--mlx-model", "local/other-model",
    ])
    assert args.whisper_model == "local/other-model"
