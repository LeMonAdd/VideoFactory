"""Render a persisted timeline with normalized FFmpeg inputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .ffmpeg_utils import FFMPEG, run_command
from .timeline import validate_timeline


def build_render_command(timeline: dict[str, Any], destination: Path,
                         encoder: str = "libx264") -> list[str]:
    validate_timeline(timeline)
    if encoder not in {"libx264", "h264_videotoolbox"}:
        raise ValueError(f"Unsupported H.264 encoder: {encoder}")
    settings = timeline["output_settings"]
    width, height, fps = int(settings["width"]), int(settings["height"]), int(settings["fps"])
    if min(width, height, fps) <= 0 or width % 2 or height % 2:
        raise ValueError("Output dimensions must be positive even numbers, and fps must be positive")
    command = [str(FFMPEG), "-hide_banner", "-loglevel", "error", "-nostdin", "-y"]
    filters: list[str] = []
    for index, event in enumerate(timeline["events"]):
        length = float(event["end"]) - float(event["start"])
        kind = event["visual_type"]
        if kind == "GRAPHIC":
            command.extend(["-f", "lavfi", "-i",
                            f"color=c={event['graphic_color']}:s={width}x{height}:r={fps}:d={length:.3f}"])
            beginning = f"[{index}:v]trim=duration={length:.3f}"
        elif kind == "IMAGE":
            command.extend(["-loop", "1", "-framerate", str(fps), "-i", event["source_path"]])
            beginning = f"[{index}:v]trim=duration={length:.3f}"
        else:
            if event["loop"]:
                command.extend(["-stream_loop", "-1"])
            command.extend(["-i", event["source_path"]])
            beginning = (f"[{index}:v]trim=start={float(event['source_in']):.3f}:"
                         f"duration={length:.3f}")
        filters.append(
            f"{beginning},setpts=PTS-STARTPTS,fps={fps},"
            f"scale={width}:{height}:force_original_aspect_ratio=increase,"
            f"crop={width}:{height},setsar=1,format=yuv420p[v{index}]"
        )
    audio_index = len(timeline["events"])
    command.extend(["-i", timeline["audio"]["source_path"]])
    duration = float(timeline["duration"])
    filters.append(
        f"[{audio_index}:a:0]atrim=duration={duration:.3f},asetpts=PTS-STARTPTS,"
        f"aresample=48000,apad=whole_dur={duration:.3f}[aout]"
    )
    labels = "".join(f"[v{i}]" for i in range(len(timeline["events"])))
    filters.append(f"{labels}concat=n={len(timeline['events'])}:v=1:a=0[vout]")
    command.extend(["-filter_complex", ";".join(filters), "-map", "[vout]", "-map", "[aout]",
                    "-c:v", encoder])
    if encoder == "libx264":
        command.extend(["-preset", "veryfast", "-crf", "22"])
    else:
        command.extend(["-b:v", "8M"])
    command.extend(["-c:a", "aac", "-b:a", "192k", "-r", str(fps),
                    "-pix_fmt", "yuv420p", "-t", f"{duration:.3f}",
                    "-movflags", "+faststart", str(destination)])
    return command


def render(timeline: dict[str, Any], destination: Path, encoder: str = "libx264",
           overwrite: bool = False) -> Path:
    if destination.exists() and not overwrite:
        raise FileExistsError(f"Rendered file already exists: {destination}; use --overwrite-render")
    destination.parent.mkdir(parents=True, exist_ok=True)
    run_command(build_render_command(timeline, destination, encoder))
    if not destination.is_file() or destination.stat().st_size == 0:
        raise RuntimeError(f"FFmpeg produced no video: {destination}")
    return destination
