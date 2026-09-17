"""Discover and probe local visual assets only."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from . import SCHEMA_VERSION
from .media_probe import probe, stream

VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".mkv", ".webm"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def source_record(source_id: str, kind: str, path: Path, info: dict) -> dict[str, Any]:
    video = stream(info, "video")
    if video is None:
        raise ValueError(f"Visual source has no video/image stream: {path}")
    format_duration = info.get("format", {}).get("duration")
    return {
        "id": source_id, "kind": kind, "path": str(path.resolve()),
        "width": video.get("width"), "height": video.get("height"),
        "codec": video.get("codec_name"),
        "duration": float(format_duration) if format_duration is not None else None,
        "source_url": None, "license": None, "creator": None,
        "attribution": None, "download_date": None,
    }


def discover_assets(root: Path, mode: str, primary: Path | None = None,
                    primary_info: dict | None = None, asset_root: Path | None = None) -> dict[str, Any]:
    sources: list[dict[str, Any]] = []
    asset_root = asset_root or root / "assets"
    if mode == "TALKING_HEAD":
        if primary is None or primary_info is None:
            raise ValueError("Talking-head asset discovery needs the source video and probe data")
        sources.append(source_record("source_aroll", "A_ROLL", primary, primary_info))
    for directory, extensions, kind, prefix in (
        (asset_root / "broll", VIDEO_EXTENSIONS, "B_ROLL", "broll"),
        (asset_root / "images", IMAGE_EXTENSIONS, "IMAGE", "image"),
    ):
        if not directory.is_dir():
            continue
        candidates = sorted(path for path in directory.iterdir()
                            if path.is_file() and path.suffix.lower() in extensions)
        for number, path in enumerate(candidates, 1):
            sources.append(source_record(f"{prefix}_{number:03d}", kind, path, probe(path)))
    return {"schema_version": SCHEMA_VERSION, "sources": sources}
