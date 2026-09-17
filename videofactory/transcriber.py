"""Interchangeable real and fixture transcription backends."""

from __future__ import annotations

import json
import os
import subprocess
import uuid
from pathlib import Path
from typing import Protocol

from .models import normalize_transcript
from .paths import ROOT


class Transcriber(Protocol):
    def transcribe(self, audio: Path, media_duration: float) -> dict: ...


class FixtureTranscriber:
    def __init__(self, fixture: Path) -> None:
        self.fixture = fixture

    def transcribe(self, audio: Path, media_duration: float) -> dict:
        try:
            raw = json.loads(self.fixture.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Cannot read fixture transcript {self.fixture}: {exc}") from exc
        return normalize_transcript(raw, media_duration)


class RealMLXTranscriber:
    def __init__(self, work_dir: Path, model: str | None = None) -> None:
        self.work_dir = work_dir
        self.model = model

    def transcribe(self, audio: Path, media_duration: float) -> dict:
        executable = ROOT / ".venv" / "bin" / "mlx_whisper"
        if not executable.is_file():
            raise RuntimeError(f"Local MLX Whisper executable is missing: {executable}")
        self.work_dir.mkdir(parents=True, exist_ok=True)
        name = f"mlx_{uuid.uuid4().hex}"
        command = [str(executable), str(audio), "--output-dir", str(self.work_dir),
                   "--output-name", name, "--output-format", "json",
                   "--word-timestamps", "True", "--verbose", "False"]
        if self.model:
            command.extend(["--model", self.model])
        environment = os.environ.copy()
        environment["HF_HUB_OFFLINE"] = "1"
        environment["TRANSFORMERS_OFFLINE"] = "1"
        try:
            result = subprocess.run(command, capture_output=True, text=True, env=environment, check=False)
        except OSError as exc:
            raise RuntimeError(f"Could not start local MLX Whisper: {exc}") from exc
        output = self.work_dir / f"{name}.json"
        # MLX Whisper's CLI may print a traceback and return zero, so require its JSON output.
        if result.returncode != 0 or not output.is_file():
            detail = (result.stderr or result.stdout or "No JSON output was written").strip()[-2000:]
            raise RuntimeError(f"Local MLX Whisper transcription failed: {detail}")
        try:
            raw = json.loads(output.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"MLX Whisper wrote invalid JSON: {output}") from exc
        return normalize_transcript(raw, media_duration)
