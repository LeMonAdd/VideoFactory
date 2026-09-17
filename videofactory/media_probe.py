"""ffprobe-backed media inspection."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .ffmpeg_utils import FFPROBE, run_command


def probe(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Media file not found: {path}")
    result = run_command([FFPROBE, "-v", "error", "-show_format", "-show_streams", "-of", "json", path])
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise ValueError(f"ffprobe returned invalid JSON for {path}") from exc


def stream(data: dict[str, Any], kind: str) -> dict[str, Any] | None:
    return next((item for item in data.get("streams", []) if item.get("codec_type") == kind), None)


def duration(data: dict[str, Any]) -> float:
    value = data.get("format", {}).get("duration")
    if value is None:
        for item in data.get("streams", []):
            if item.get("duration") is not None:
                value = item["duration"]
                break
    if value is None or float(value) <= 0:
        raise ValueError("Media has no positive duration")
    return float(value)
