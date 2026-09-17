"""Plan cuts from validated acoustic silences and one canonical time map."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Literal

from jsonschema import Draft202012Validator

from . import SCHEMA_VERSION
from .clip_planner import TIMING_TOLERANCE
from .paths import ROOT
from .silence_detector import validate_noise_db, validate_silences

PLAN_SCHEMA = ROOT / "config" / "speech_edit_plan.schema.json"
MAP_SCHEMA = ROOT / "config" / "retime_map.schema.json"
BOUNDARY_PADDING_MS = 40


def _milliseconds(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{label} must be finite numeric timing")
    return round(value * 1000)


def _seconds(value: int) -> float:
    return value / 1000


def validate_parameters(threshold: float, keep: float) -> tuple[int, int]:
    threshold_ms = _milliseconds(threshold, "Pause threshold")
    keep_ms = _milliseconds(keep, "Pause keep")
    if threshold <= 0 or threshold_ms <= 0:
        raise ValueError("Pause threshold must be positive")
    if keep < 0 or keep_ms < 0 or keep >= threshold or keep_ms >= threshold_ms:
        raise ValueError("Pause keep must be non-negative and below threshold")
    return threshold_ms, keep_ms


def validate_inputs(project: dict[str, Any], transcript: dict[str, Any],
                    timeline: dict[str, Any]) -> tuple[int, list[tuple[int, int]]]:
    _schema_validate(timeline, ROOT / "config" / "edit_timeline.schema.json")
    name = project.get("project_name")
    if not isinstance(name, str) or not name or timeline.get("project_name") != name:
        raise ValueError("Project and edit timeline project_name mismatch")
    if timeline.get("mode") != project.get("mode") or project.get("mode") not in {"TALKING_HEAD", "VOICEOVER"}:
        raise ValueError("Project and edit timeline mode mismatch")
    duration = _milliseconds(timeline.get("duration"), "Edit timeline duration")
    transcript_duration = _milliseconds(transcript.get("duration"), "Transcript duration")
    if duration <= 0 or transcript_duration <= 0 or abs(duration - transcript_duration) > TIMING_TOLERANCE * 1000:
        raise ValueError("Transcript and edit timeline duration mismatch")
    base_audio = timeline["base_audio"]
    if base_audio["source_path"] != project.get("source_media"):
        raise ValueError("Edit timeline primary audio source mismatch")
    if (_milliseconds(base_audio["timeline_start"], "Audio start") != 0 or
            abs(_milliseconds(base_audio["timeline_end"], "Audio end") - duration) > TIMING_TOLERANCE * 1000 or
            _milliseconds(base_audio["source_in"], "Audio source in") != 0 or
            abs(_milliseconds(base_audio["source_out"], "Audio source out") - duration) > TIMING_TOLERANCE * 1000):
        raise ValueError("Edit timeline primary audio does not cover project duration")
    for overlay in timeline["visual_overlays"]:
        start = _milliseconds(overlay["timeline_start"], "Overlay start")
        end = _milliseconds(overlay["timeline_end"], "Overlay end")
        if start < 0 or end <= start or end > duration:
            raise ValueError("Edit timeline overlay outside project duration")
    segments = transcript.get("segments")
    if not isinstance(segments, list) or not segments:
        raise ValueError("Transcript must contain timed word segments")
    words: list[tuple[int, int]] = []
    for segment in segments:
        if not isinstance(segment, dict) or not isinstance(segment.get("words"), list) or not segment["words"]:
            raise ValueError("Every transcript segment needs word timestamps")
        segment_start = _milliseconds(segment.get("start"), "Segment start")
        segment_end = _milliseconds(segment.get("end"), "Segment end")
        if segment_start < 0 or segment_end <= segment_start or segment_end > duration + TIMING_TOLERANCE * 1000:
            raise ValueError("Transcript segment timestamp outside project duration or empty")
        for word in segment["words"]:
            if not isinstance(word, dict):
                raise ValueError("Malformed word timestamp")
            start = _milliseconds(word.get("start"), "Word start")
            end = _milliseconds(word.get("end"), "Word end")
            if start < 0 or end <= start or end > duration + TIMING_TOLERANCE * 1000:
                raise ValueError("Word timestamp outside project duration or empty")
            if start < segment_start - TIMING_TOLERANCE * 1000 or end > segment_end + TIMING_TOLERANCE * 1000:
                raise ValueError("Word timestamp outside transcript segment")
            if words and start < words[-1][1]:
                raise ValueError("Word timestamps overlap or are unordered")
            words.append((start, end))
    return duration, words


def _schema_validate(document: dict[str, Any], path: Path) -> None:
    schema = json.loads(path.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    errors = list(Draft202012Validator(schema).iter_errors(document))
    if errors:
        error = errors[0]
        raise ValueError(f"Invalid {path.name} at {'.'.join(map(str, error.path)) or 'root'}: {error.message}")


def _build_keeps(duration: int, cuts: list[dict[str, Any]]) -> list[dict[str, float]]:
    keeps = []
    source_cursor = edited_cursor = 0
    for cut in cuts:
        cut_start = _milliseconds(cut["source_start"], "Cut start")
        if cut_start > source_cursor:
            length = cut_start - source_cursor
            keeps.append({"source_start": _seconds(source_cursor), "source_end": _seconds(cut_start),
                          "source_duration": _seconds(length), "edited_start": _seconds(edited_cursor),
                          "edited_end": _seconds(edited_cursor + length)})
            edited_cursor += length
        source_cursor = _milliseconds(cut["source_end"], "Cut end")
    if source_cursor < duration:
        length = duration - source_cursor
        keeps.append({"source_start": _seconds(source_cursor), "source_end": _seconds(duration),
                      "source_duration": _seconds(length), "edited_start": _seconds(edited_cursor),
                      "edited_end": _seconds(edited_cursor + length)})
    return keeps


def validate_speech_edit_plan(document: dict[str, Any], project: dict[str, Any],
                              transcript: dict[str, Any], timeline: dict[str, Any],
                              silences: list[tuple[float, float]]) -> dict[str, Any]:
    _schema_validate(document, PLAN_SCHEMA)
    duration, words = validate_inputs(project, transcript, timeline)
    if document["project_name"] != project["project_name"] or _milliseconds(document["source_duration"], "Source duration") != duration:
        raise ValueError("Speech edit plan project or duration mismatch")
    threshold_ms, keep_ms = validate_parameters(document["pause_threshold_seconds"],
                                                 document["pause_keep_seconds"])
    padding_ms = _milliseconds(document["speech_boundary_padding_seconds"], "Boundary padding")
    if padding_ms != BOUNDARY_PADDING_MS:
        raise ValueError("Speech boundary padding mismatch")
    detection = document["detection"]
    validate_noise_db(detection["noise_db"])
    validate_silences(silences, _seconds(duration))
    first_word_start, last_word_end = words[0][0], words[-1][1]
    validated_intervals = {(_milliseconds(start, "Silence start"), _milliseconds(end, "Silence end"))
                           for start, end in silences}
    previous_end = 0
    for index, cut in enumerate(document["cuts"], 1):
        start, end = (_milliseconds(cut[key], key) for key in ("source_start", "source_end"))
        if cut["id"] != index or start < previous_end or end <= start or end > duration:
            raise ValueError("Speech cuts must be ordered, disjoint, and inside the source")
        if _milliseconds(cut["removed_duration"], "Removed duration") != end - start:
            raise ValueError("Speech cut duration mismatch")
        silence_start = _milliseconds(cut["detected_silence_start"], "Detected silence start")
        silence_end = _milliseconds(cut["detected_silence_end"], "Detected silence end")
        if ((silence_start, silence_end) not in validated_intervals or
                silence_start < first_word_start or silence_end > last_word_end or
                silence_end - silence_start < threshold_ms or
                start < silence_start + padding_ms or end > silence_end - padding_ms or
                (start - silence_start) + (silence_end - end) < keep_ms):
            raise ValueError("Speech cut is not safely inside an internal detected silence")
        expected = sorted({overlay["shot_id"] for overlay in timeline["visual_overlays"]
                           if overlay["visual_type"] == "B_ROLL" and
                           start < _milliseconds(overlay["timeline_end"], "Overlay end") and
                           end > _milliseconds(overlay["timeline_start"], "Overlay start")})
        if cut["affected_overlay_shot_ids"] != expected:
            raise ValueError("Speech cut overlay diagnostics mismatch")
        previous_end = end
    removed = sum(_milliseconds(cut["removed_duration"], "Removed duration") for cut in document["cuts"])
    if (_milliseconds(document["total_removed_duration"], "Total removed") != removed or
            _milliseconds(document["edited_duration"], "Edited duration") != duration - removed):
        raise ValueError("Speech edit plan duration totals mismatch")
    if document["keep_segments"] != _build_keeps(duration, document["cuts"]):
        raise ValueError("Keep segments do not exactly cover the complement of cuts")
    return document


class SpeechEditPlanner:
    def plan(self, project: dict[str, Any], transcript: dict[str, Any],
             timeline: dict[str, Any], silences: list[tuple[float, float]],
             threshold: float = 1.0, keep: float = 0.25,
             noise_db: float = -35.0) -> dict[str, Any]:
        threshold_ms, keep_ms = validate_parameters(threshold, keep)
        duration, words = validate_inputs(project, transcript, timeline)
        validate_noise_db(noise_db)
        validate_silences(silences, _seconds(duration))
        first_word_start, last_word_end = words[0][0], words[-1][1]
        cuts = []
        for raw_start, raw_end in silences:
            if raw_end - raw_start < threshold:
                continue
            silence_start = _milliseconds(raw_start, "Silence start")
            silence_end = _milliseconds(raw_end, "Silence end")
            gap = silence_end - silence_start
            if (silence_start < first_word_start or silence_end > last_word_end or
                    gap < threshold_ms):
                continue
            retained = max(keep_ms, 2 * BOUNDARY_PADDING_MS)
            if gap <= retained:
                continue
            left = retained // 2
            start, end = silence_start + left, silence_end - (retained - left)
            affected = sorted({overlay["shot_id"] for overlay in timeline["visual_overlays"]
                               if overlay["visual_type"] == "B_ROLL" and
                               start < _milliseconds(overlay["timeline_end"], "Overlay end") and
                               end > _milliseconds(overlay["timeline_start"], "Overlay start")})
            cuts.append({"id": len(cuts) + 1, "source_start": _seconds(start),
                         "source_end": _seconds(end), "removed_duration": _seconds(end - start),
                         "detected_silence_start": round(raw_start, 6),
                         "detected_silence_end": round(raw_end, 6),
                         "reason": "Long detected audio silence", "affected_overlay_shot_ids": affected})
        removed = sum(_milliseconds(cut["removed_duration"], "Removed duration") for cut in cuts)
        result = {"schema_version": SCHEMA_VERSION, "project_name": project["project_name"],
                  "source_duration": _seconds(duration),
                  "pause_threshold_seconds": _seconds(threshold_ms),
                  "pause_keep_seconds": _seconds(keep_ms),
                  "speech_boundary_padding_seconds": _seconds(BOUNDARY_PADDING_MS),
                  "detection": {"method": "ffmpeg_silencedetect", "noise_db": float(noise_db)},
                  "cuts": cuts, "keep_segments": _build_keeps(duration, cuts),
                  "total_removed_duration": _seconds(removed),
                  "edited_duration": _seconds(duration - removed)}
        return validate_speech_edit_plan(result, project, transcript, timeline, silences)


def validate_retime_map(document: dict[str, Any], plan: dict[str, Any]) -> dict[str, Any]:
    _schema_validate(document, MAP_SCHEMA)
    if (document["project_name"] != plan["project_name"] or
            document["source_duration"] != plan["source_duration"] or
            document["edited_duration"] != plan["edited_duration"] or
            document["segments"] != plan["keep_segments"]):
        raise ValueError("Retime map does not match canonical speech edit plan")
    segments = document["segments"]
    if not segments or segments[0]["source_start"] != 0 or segments[-1]["source_end"] != document["source_duration"]:
        raise ValueError("Retime map must preserve project start and tail")
    edited_cursor = 0
    previous_source_end = 0
    for segment in segments:
        start, end, edited_start, edited_end = (_milliseconds(segment[key], key) for key in
                                                ("source_start", "source_end", "edited_start", "edited_end"))
        if (start < previous_source_end or end <= start or edited_start != edited_cursor or
                edited_end - edited_start != end - start):
            raise ValueError("Retime map segments are not monotonic or duration preserving")
        previous_source_end, edited_cursor = end, edited_end
    if edited_cursor != _milliseconds(document["edited_duration"], "Edited duration"):
        raise ValueError("Retime map edited duration mismatch")
    return document


def build_retime_map(plan: dict[str, Any]) -> dict[str, Any]:
    result = {"schema_version": SCHEMA_VERSION, "project_name": plan["project_name"],
              "source_duration": plan["source_duration"], "edited_duration": plan["edited_duration"],
              "segments": plan["keep_segments"]}
    return validate_retime_map(result, plan)


def map_source_time(time: float, retime_map: dict[str, Any],
                    bias: Literal["error", "left", "right"] = "error") -> float:
    """Map kept time; for cut interiors, explicitly opt into either cut boundary."""
    if bias not in {"error", "left", "right"}:
        raise ValueError("Bias must be error, left, or right")
    if isinstance(time, bool) or not isinstance(time, (int, float)) or not math.isfinite(time):
        raise ValueError("Source timestamp must be finite numeric timing")
    if time < 0 or time > retime_map["source_duration"]:
        raise ValueError("Source timestamp outside project duration")
    segments = retime_map["segments"]
    for index, segment in enumerate(segments):
        start, end = segment["source_start"], segment["source_end"]
        if start <= time <= end:
            if time == start:
                return segment["edited_start"]
            if time == end:
                return segment["edited_end"]
            return segment["edited_start"] + (time - start)
        if index and segments[index - 1]["source_end"] < time < start:
            if bias == "error":
                raise ValueError("Source timestamp is inside a removed region; choose left or right bias")
            return segment["edited_start"]
    raise ValueError("Source timestamp is not covered by the retime map")
