"""Atomic UTF-8 SRT and WebVTT sidecar publication; no media processing."""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from . import SCHEMA_VERSION
from .caption_timeline import build_caption_timeline, validate_caption_timeline
from .models import read_json, write_json
from .paths import ROOT
from .source_downloader import sha256_file

MANIFEST_SCHEMA = ROOT / "config" / "caption_export_manifest.schema.json"
MAX_LINE_CHARACTERS = 42
_SRT_TIME = re.compile(r"^\d{2,}:\d{2}:\d{2},\d{3} --> \d{2,}:\d{2}:\d{2},\d{3}$")
_VTT_TIME = re.compile(r"^\d{2,}:\d{2}:\d{2}\.\d{3} --> \d{2,}:\d{2}:\d{2}\.\d{3}$")


def wrap_caption_text(text: str, max_width: int = MAX_LINE_CHARACTERS) -> str:
    words = text.split()
    if len(text) <= max_width or len(words) <= 1:
        return text
    candidates = [(" ".join(words[:index]), " ".join(words[index:]))
                  for index in range(1, len(words))]
    left, right = min(candidates, key=lambda pair: (
        max(len(pair[0]) - max_width, 0) + max(len(pair[1]) - max_width, 0),
        abs(len(pair[0]) - len(pair[1])), len(pair[0])))
    return f"{left}\n{right}"


def _timestamp(seconds: float, separator: str) -> str:
    milliseconds = round(seconds * 1000)
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    whole_seconds, fraction = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{whole_seconds:02d}{separator}{fraction:03d}"


def render_srt(timeline: dict[str, Any]) -> str:
    blocks = [f"{cue['index']}\n{_timestamp(cue['start'], ',')} --> {_timestamp(cue['end'], ',')}\n"
              f"{wrap_caption_text(cue['text'])}" for cue in timeline["cues"]]
    return "\n\n".join(blocks) + "\n"


def render_vtt(timeline: dict[str, Any]) -> str:
    blocks = [f"{_timestamp(cue['start'], '.')} --> {_timestamp(cue['end'], '.')}\n"
              f"{wrap_caption_text(cue['text'])}" for cue in timeline["cues"]]
    return "WEBVTT\n\n" + "\n\n".join(blocks) + "\n"


def validate_sidecar(text: str, timeline: dict[str, Any], kind: str) -> None:
    if not text.endswith("\n") or "\r" in text or text.startswith("\ufeff"):
        raise ValueError(f"Invalid {kind} line endings or encoding marker")
    if kind == "vtt":
        if not text.startswith("WEBVTT\n\n"):
            raise ValueError("WebVTT header is missing")
        body = text[len("WEBVTT\n\n"):]
        pattern = _VTT_TIME
    elif kind == "srt":
        body = text
        pattern = _SRT_TIME
    else:
        raise ValueError("Unknown caption sidecar format")
    blocks = body.rstrip("\n").split("\n\n")
    if len(blocks) != len(timeline["cues"]):
        raise ValueError(f"Invalid {kind} cue count")
    for cue, block in zip(timeline["cues"], blocks):
        lines = block.split("\n")
        if kind == "srt":
            if not lines or lines.pop(0) != str(cue["index"]):
                raise ValueError("Invalid SRT cue index")
        if not lines or not pattern.fullmatch(lines.pop(0)) or not 1 <= len(lines) <= 2:
            raise ValueError(f"Invalid {kind} cue formatting")
        if " ".join(lines) != cue["text"] or any(not line for line in lines):
            raise ValueError(f"Invalid {kind} cue text")


def validate_caption_export_manifest(document: dict[str, Any], project_name: str) -> dict[str, Any]:
    schema = json.loads(MANIFEST_SCHEMA.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    errors = list(Draft202012Validator(schema).iter_errors(document))
    if errors:
        first = errors[0]
        raise ValueError(f"Invalid caption_export_manifest.json at {'.'.join(map(str, first.path)) or 'root'}: {first.message}")
    if document["project_name"] != project_name:
        raise ValueError("Caption export manifest project_name mismatch")
    return document


class CaptionExporter:
    def export(self, project_dir: Path, output_dir: Path, project_name: str,
               transcript: dict[str, Any], retime_map: dict[str, Any],
               retimed: dict[str, Any], overwrite: bool = False,
               project_language: str | None = None) -> dict[str, Any]:
        targets = (output_dir / "captions.srt", output_dir / "captions.vtt")
        parts = (output_dir / "captions.srt.part", output_dir / "captions.vtt.part")
        backups = (output_dir / "captions.srt.backup", output_dir / "captions.vtt.backup")
        if output_dir.is_symlink() or any(path.is_symlink() for path in (*targets, *parts, *backups)):
            raise ValueError("Caption output paths must not be symlinks")
        if any(path.exists() for path in parts + backups):
            raise FileExistsError("Stale caption staging file exists")
        if not overwrite and any(path.exists() for path in targets):
            raise FileExistsError("Caption sidecar already exists; use --overwrite-captions")
        input_paths = {"transcript.json": project_dir / "transcript.json",
                       "retime_map.json": project_dir / "retime_map.json",
                       "retimed_edit_timeline.json": project_dir / "retimed_edit_timeline.json"}
        for filename, document in (("transcript.json", transcript), ("retime_map.json", retime_map),
                                   ("retimed_edit_timeline.json", retimed)):
            if read_json(input_paths[filename]) != document:
                raise ValueError(f"Saved {filename} changed before caption export")
        input_hashes = {name: sha256_file(path) for name, path in input_paths.items()}
        timeline = build_caption_timeline(project_name, transcript, retime_map, retimed,
                                          project_language=project_language)
        srt, vtt = render_srt(timeline), render_vtt(timeline)
        validate_sidecar(srt, timeline, "srt")
        validate_sidecar(vtt, timeline, "vtt")
        timeline_path = project_dir / "caption_timeline.json"
        write_json(timeline_path, timeline)
        if read_json(timeline_path) != timeline:
            raise ValueError("Caption timeline changed after save")
        timeline_hash = sha256_file(timeline_path)
        payloads = (srt.encode("utf-8"), vtt.encode("utf-8"))
        output_dir.mkdir(parents=True, exist_ok=True)
        published: list[int] = []
        backed_up: list[int] = []
        originally_present = [path.exists() for path in targets]
        completed = False
        try:
            for part, payload in zip(parts, payloads):
                part.write_bytes(payload)
            for part, kind in zip(parts, ("srt", "vtt")):
                validate_sidecar(part.read_text(encoding="utf-8"), timeline, kind)
            if any(sha256_file(path) != input_hashes[name] for name, path in input_paths.items()):
                raise ValueError("Caption input artifacts changed during export")
            if sha256_file(timeline_path) != timeline_hash:
                raise ValueError("Caption timeline changed during export")
            manifest = validate_caption_export_manifest({
                "schema_version": SCHEMA_VERSION, "project_name": project_name,
                "input_transcript": "transcript.json", "input_retime_map": "retime_map.json",
                "input_retimed_edit_timeline": "retimed_edit_timeline.json",
                "duration": timeline["duration"], "language": timeline["language"],
                "cue_count": len(timeline["cues"]),
                "srt_path": str(targets[0]), "vtt_path": str(targets[1]),
                "srt_size_bytes": len(payloads[0]), "vtt_size_bytes": len(payloads[1]),
                "srt_sha256": hashlib.sha256(payloads[0]).hexdigest(),
                "vtt_sha256": hashlib.sha256(payloads[1]).hexdigest(),
                "transcript_sha256": input_hashes["transcript.json"],
                "retime_map_sha256": input_hashes["retime_map.json"],
                "retimed_edit_timeline_sha256": input_hashes["retimed_edit_timeline.json"],
                "caption_timeline_sha256": timeline_hash,
            }, project_name)
            if overwrite:
                for index, (target, backup) in enumerate(zip(targets, backups)):
                    if originally_present[index]:
                        os.link(target, backup)
                        backed_up.append(index)
                for index, (part, target) in enumerate(zip(parts, targets)):
                    os.replace(part, target)
                    published.append(index)
            else:
                for index, (part, target) in enumerate(zip(parts, targets)):
                    os.link(part, target)
                    published.append(index)
                    part.unlink()
            if any(sha256_file(path) != input_hashes[name] for name, path in input_paths.items()):
                raise ValueError("Caption input artifacts changed during export")
            write_json(project_dir / "caption_export_manifest.json", manifest)
            completed = True
            return manifest
        finally:
            if not completed:
                for index in reversed(published):
                    if originally_present[index]:
                        os.replace(backups[index], targets[index])
                        backed_up.remove(index)
                    elif targets[index].exists():
                        targets[index].unlink()
            for path in parts:
                if path.exists():
                    path.unlink()
            for index in backed_up:
                if backups[index].exists():
                    backups[index].unlink()
