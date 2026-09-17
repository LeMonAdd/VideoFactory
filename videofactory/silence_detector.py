"""Analyze primary audio with FFmpeg silencedetect; never write media."""

from __future__ import annotations

import math
import re
import subprocess
from pathlib import Path

from .clip_planner import TIMING_TOLERANCE
from .ffmpeg_utils import FFMPEG
from .media_probe import probe, stream

_START = re.compile(r"\bsilence_start:\s*(\S+)")
_END = re.compile(r"\bsilence_end:\s*(\S+)")


def validate_noise_db(noise_db: float) -> float:
    if (isinstance(noise_db, bool) or not isinstance(noise_db, (int, float)) or
            not math.isfinite(noise_db) or not -100 < noise_db < 0):
        raise ValueError("Silence noise threshold must be finite and between -100 and 0 dB")
    return float(noise_db)


def validate_silences(intervals: list[tuple[float, float]],
                      source_duration: float) -> list[tuple[float, float]]:
    if not math.isfinite(source_duration) or source_duration <= 0:
        raise ValueError("Source duration must be finite and positive")
    previous_end = 0.0
    for start, end in intervals:
        if (not math.isfinite(start) or not math.isfinite(end) or start < 0 or
                end <= start or end > source_duration + TIMING_TOLERANCE or
                start < previous_end):
            raise ValueError("Invalid, overlapping, or out-of-order detected silence")
        previous_end = end
    return intervals


def parse_silencedetect(stderr: str, source_duration: float) -> list[tuple[float, float]]:
    """Require one start for each end; unrelated FFmpeg diagnostics are ignored."""
    intervals: list[tuple[float, float]] = []
    pending: float | None = None
    for line in stderr.splitlines():
        start_match, end_match = _START.search(line), _END.search(line)
        if start_match and end_match:
            raise ValueError("Malformed silencedetect line contains both start and end")
        if start_match:
            if pending is not None:
                raise ValueError("Unpaired silencedetect start")
            try:
                pending = float(start_match.group(1))
            except ValueError as exc:
                raise ValueError("Malformed silencedetect start") from exc
            if not math.isfinite(pending) or pending < 0:
                raise ValueError("Invalid silencedetect start")
        elif end_match:
            if pending is None:
                raise ValueError("Unpaired silencedetect end")
            try:
                end = float(end_match.group(1))
            except ValueError as exc:
                raise ValueError("Malformed silencedetect end") from exc
            intervals.append((pending, end))
            pending = None
    if pending is not None:
        raise ValueError("Unpaired silencedetect start")
    return validate_silences(intervals, source_duration)


class LocalAudioSilenceDetector:
    def detect(self, primary: Path, source_duration: float, minimum_duration: float,
               noise_db: float = -35.0) -> list[tuple[float, float]]:
        validate_noise_db(noise_db)
        if not math.isfinite(minimum_duration) or minimum_duration <= 0:
            raise ValueError("Silence detection duration must be finite and positive")
        info = probe(primary)
        audio = stream(info, "audio")
        if audio is None or not isinstance(audio.get("index"), int):
            raise ValueError("Primary source has no identified audio stream")
        command = [str(FFMPEG), "-hide_banner", "-loglevel", "info", "-nostdin",
                   "-i", str(primary), "-map", f"0:{audio['index']}",
                   "-af", f"silencedetect=noise={noise_db:g}dB:d={minimum_duration:g}",
                   "-f", "null", "-"]
        try:
            result = subprocess.run(command, capture_output=True, text=True, check=False)
        except OSError as exc:
            raise RuntimeError(f"Could not start FFmpeg silence analysis: {exc}") from exc
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg silence analysis failed (exit {result.returncode}): "
                               f"{result.stderr.strip()[-500:]}")
        return parse_silencedetect(result.stderr, source_duration)
