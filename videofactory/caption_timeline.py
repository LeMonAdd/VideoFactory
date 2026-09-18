"""Deterministic accessibility cues on the frozen speech-edited time axis."""

from __future__ import annotations

import json
import math
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from . import SCHEMA_VERSION
from .clip_planner import TIMING_TOLERANCE
from .paths import ROOT
from .speech_edit import validate_retime_map

SCHEMA = ROOT / "config" / "caption_timeline.schema.json"
RETIMED_SCHEMA = ROOT / "config" / "retimed_edit_timeline.schema.json"
MAX_CUE_SECONDS = 6.0
MAX_CUE_CHARACTERS = 84
MIN_REPAIR_SECONDS = 0.001
STRONG_PUNCTUATION = (".", "?", "!", "…")


@dataclass(frozen=True)
class CaptionToken:
    start: float
    end: float
    text: str


def _finite_time(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{label} must be finite numeric timing")
    return float(value)


def _schema_validate(document: dict[str, Any], path: Path) -> None:
    schema = json.loads(path.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    errors = list(Draft202012Validator(schema).iter_errors(document))
    if errors:
        first = errors[0]
        raise ValueError(f"Invalid {path.name} at {'.'.join(map(str, first.path)) or 'root'}: {first.message}")


def validate_caption_inputs(project_name: str, transcript: dict[str, Any],
                            retime_map: dict[str, Any], retimed: dict[str, Any]) -> None:
    _schema_validate(retimed, RETIMED_SCHEMA)
    pseudo_plan = {"project_name": retime_map.get("project_name"),
                   "source_duration": retime_map.get("source_duration"),
                   "edited_duration": retime_map.get("edited_duration"),
                   "keep_segments": retime_map.get("segments")}
    validate_retime_map(retime_map, pseudo_plan)
    transcript_duration = _finite_time(transcript.get("duration"), "Transcript duration")
    source_duration = _finite_time(retime_map["source_duration"], "Source duration")
    edited_duration = _finite_time(retime_map["edited_duration"], "Edited duration")
    if (not project_name or retime_map["project_name"] != project_name or
            retimed["project_name"] != project_name or
            abs(transcript_duration - source_duration) > TIMING_TOLERANCE or
            abs(retimed["source_duration"] - source_duration) > TIMING_TOLERANCE or
            abs(retimed["duration"] - edited_duration) > TIMING_TOLERANCE or
            retimed["base_audio"]["keep_segments"] != retime_map["segments"] or
            retimed["base_video"]["keep_segments"] != retime_map["segments"]):
        raise ValueError("Caption inputs disagree on project, duration, or canonical keep segments")
    if not isinstance(transcript.get("segments"), list) or not transcript["segments"]:
        raise ValueError("Transcript needs timed segments for captions")


def map_caption_time(source_time: float, retime_map: dict[str, Any]) -> float:
    """Collapse cut interiors to their edited boundary; canonical mapping stays strict."""
    instant = _finite_time(source_time, "Caption source timestamp")
    source_duration = retime_map["source_duration"]
    if instant < 0 or instant > source_duration + TIMING_TOLERANCE:
        raise ValueError("Caption source timestamp outside source duration")
    instant = min(instant, source_duration)
    for segment in retime_map["segments"]:
        if instant < segment["source_start"]:
            return segment["edited_start"]
        if instant <= segment["source_end"]:
            return round(segment["edited_start"] + instant - segment["source_start"], 3)
    return retime_map["edited_duration"]


def _plain(value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError("Caption token text must be a string")
    result = " ".join(value.split())
    if not result:
        raise ValueError("Caption token text is empty")
    return result


def _unpunctuated(value: str) -> str:
    return "".join(char for char in value if not unicodedata.category(char).startswith("P")).casefold()


def _tokens(transcript: dict[str, Any], retime_map: dict[str, Any]) -> list[CaptionToken]:
    result: list[CaptionToken] = []
    previous_end = 0.0
    for segment in transcript["segments"]:
        if not isinstance(segment, dict):
            raise ValueError("Malformed transcript segment")
        entries = segment.get("words") or [segment]
        if not isinstance(entries, list) or not entries:
            raise ValueError("Malformed transcript word list")
        segment_words = segment.get("text", "")
        text_parts = segment_words.split() if isinstance(segment_words, str) else []
        use_segment_punctuation = (len(text_parts) == len(entries) and all(
            _unpunctuated(part) == _unpunctuated(_plain(word.get("text", "")))
            for part, word in zip(text_parts, entries) if isinstance(word, dict)))
        for index, word in enumerate(entries):
            if not isinstance(word, dict):
                raise ValueError("Malformed transcript word")
            start = _finite_time(word.get("start"), "Word start")
            end = _finite_time(word.get("end"), "Word end")
            if start < 0 or end <= start or start < previous_end - TIMING_TOLERANCE:
                raise ValueError("Caption word timestamps are invalid or unordered")
            text = _plain(text_parts[index] if use_segment_punctuation else word.get("text", ""))
            mapped_start = map_caption_time(start, retime_map)
            mapped_end = map_caption_time(end, retime_map)
            result.append(CaptionToken(mapped_start, mapped_end, text))
            previous_end = end
    return result


def retimed_caption_tokens(transcript: dict[str, Any], retime_map: dict[str, Any]) -> list[CaptionToken]:
    """Return validated transcript tokens on the caption-specific collapsed time axis."""
    return _tokens(transcript, retime_map)


def _cue_text(tokens: list[CaptionToken]) -> str:
    return re.sub(r"\s+([,.;:!?…])", r"\1", " ".join(token.text for token in tokens))


def _raw_cues(tokens: list[CaptionToken]) -> list[dict[str, Any]]:
    cues: list[dict[str, Any]] = []
    pending: list[CaptionToken] = []
    def flush() -> None:
        if pending:
            cues.append({"start": pending[0].start, "end": pending[-1].end,
                         "text": _cue_text(pending)})
            pending.clear()
    for token in tokens:
        if pending and len(pending) >= 2 and (
            token.end - pending[0].start > MAX_CUE_SECONDS or
            len(_cue_text(pending + [token])) > MAX_CUE_CHARACTERS
        ):
            flush()
        pending.append(token)
        if len(pending) >= 2 and token.text.endswith(STRONG_PUNCTUATION):
            flush()
    flush()
    return cues


def _repair_cues(cues: list[dict[str, Any]], duration: float) -> list[dict[str, Any]]:
    """A collapsed zero-length cue joins its neighbor; a lone cue gets at most 1 ms."""
    result: list[dict[str, Any]] = []
    prefix = ""
    for cue in cues:
        start = max(0.0, cue["start"], result[-1]["end"] if result else 0.0)
        end = min(duration, cue["end"])
        text = f"{prefix} {cue['text']}".strip() if prefix else cue["text"]
        prefix = ""
        if end <= start:
            if result:
                result[-1]["text"] += " " + text
            else:
                prefix = text
            continue
        result.append({"start": round(start, 3), "end": round(end, 3), "text": text})
    if prefix:
        start = max(0.0, duration - MIN_REPAIR_SECONDS)
        result.append({"start": round(start, 3), "end": duration, "text": prefix})
    for index, cue in enumerate(result, 1):
        cue["index"] = index
    return result


def validate_caption_timeline(document: dict[str, Any], project_name: str,
                              retimed: dict[str, Any]) -> dict[str, Any]:
    _schema_validate(document, SCHEMA)
    if (document["project_name"] != project_name or
            abs(document["duration"] - retimed["duration"]) > TIMING_TOLERANCE):
        raise ValueError("Caption timeline project or duration mismatch")
    duration = _finite_time(document["duration"], "Caption duration")
    prior_end = 0.0
    for index, cue in enumerate(document["cues"], 1):
        start = _finite_time(cue["start"], "Cue start")
        end = _finite_time(cue["end"], "Cue end")
        if (cue["index"] != index or start < prior_end or start < 0 or
                end <= start or end > duration or
                not cue["text"].strip() or "\n" in cue["text"] or "\r" in cue["text"]):
            raise ValueError(f"Invalid caption cue {index}")
        prior_end = end
    return document


def build_caption_timeline(project_name: str, transcript: dict[str, Any],
                           retime_map: dict[str, Any], retimed: dict[str, Any],
                           project_language: str | None = None) -> dict[str, Any]:
    validate_caption_inputs(project_name, transcript, retime_map, retimed)
    tokens = retimed_caption_tokens(transcript, retime_map)
    if not tokens:
        raise ValueError("Transcript has no caption tokens")
    duration = retimed["duration"]
    cues = _repair_cues(_raw_cues(tokens), duration)
    language = transcript.get("language")
    if not isinstance(language, str) or not language.strip():
        language = project_language if isinstance(project_language, str) and project_language.strip() not in {"", "auto"} else "und"
    result = {"schema_version": SCHEMA_VERSION, "project_name": project_name,
              "source_transcript": "transcript.json", "retime_source": "retime_map.json",
              "timeline_source": "retimed_edit_timeline.json", "duration": duration,
              "language": language.strip(), "cues": cues}
    return validate_caption_timeline(result, project_name, retimed)
