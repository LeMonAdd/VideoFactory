"""Input validation without modifying source media."""

from __future__ import annotations

from pathlib import Path

from .media_probe import duration, probe, stream


def inspect_input(path: Path, mode: str) -> tuple[Path, dict, float]:
    source = path.expanduser().resolve(strict=True)
    info = probe(source)
    if stream(info, "audio") is None:
        raise ValueError(f"Input has no audio stream: {source}")
    if mode == "TALKING_HEAD" and stream(info, "video") is None:
        raise ValueError(f"Talking-head input has no video stream: {source}")
    return source, info, duration(info)
