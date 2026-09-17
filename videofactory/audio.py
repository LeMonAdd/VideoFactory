"""Narration extraction for transcription only."""

from __future__ import annotations

from pathlib import Path

from .ffmpeg_utils import FFMPEG, run_command


def extract_narration(source: Path, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    run_command([FFMPEG, "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
                 "-i", source, "-map", "0:a:0", "-vn", "-ac", "1", "-ar", "16000",
                 "-c:a", "pcm_s16le", destination])
    if not destination.is_file() or destination.stat().st_size == 0:
        raise RuntimeError(f"Audio extraction produced no data: {destination}")
    return destination
