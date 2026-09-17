"""Validate rendered media with ffprobe."""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path
from typing import Any

from .media_probe import duration, probe, stream


def validate_output(path: Path, settings: dict[str, Any]) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"Rendered output does not exist or is empty: {path}")
    info = probe(path)
    video, audio = stream(info, "video"), stream(info, "audio")
    if video is None or audio is None:
        raise ValueError("Rendered output must contain video and audio streams")
    actual_duration = duration(info)
    if actual_duration <= 0:
        raise ValueError("Rendered output has non-positive duration")
    width, height = int(video.get("width", 0)), int(video.get("height", 0))
    if (width, height) != (int(settings["width"]), int(settings["height"])):
        raise ValueError(f"Rendered resolution is {width}x{height}, expected {settings['width']}x{settings['height']}")
    frame_rate = float(Fraction(video.get("avg_frame_rate", "0/1")))
    if abs(frame_rate - float(settings["fps"])) > 0.1:
        raise ValueError(f"Rendered frame rate is {frame_rate}, expected {settings['fps']}")
    if video.get("codec_name") != "h264" or audio.get("codec_name") != "aac":
        raise ValueError(f"Rendered codecs are {video.get('codec_name')}/{audio.get('codec_name')}, expected h264/aac")
    return {
        "path": str(path.resolve()), "duration": actual_duration,
        "width": width, "height": height, "fps": frame_rate,
        "video_codec": video["codec_name"], "audio_codec": audio["codec_name"],
    }
