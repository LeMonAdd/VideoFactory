"""Deterministic, media-free Smart Emphasis candidates and plan validation."""

from __future__ import annotations

import json
import math
import unicodedata
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from . import SCHEMA_VERSION
from .caption_timeline import (build_caption_timeline, retimed_caption_tokens,
                               validate_caption_inputs, validate_caption_timeline)
from .clip_planner import TIMING_TOLERANCE
from .paths import ROOT

CANDIDATES_SCHEMA = ROOT / "config" / "emphasis_candidates.schema.json"
PLAN_SCHEMA = ROOT / "config" / "emphasis_plan.schema.json"
MAX_WORDS = 5
MAX_CHARACTERS = 48
MIN_DISPLAY_SECONDS = 1.5
MAX_DISPLAY_SECONDS = 3.0
LEAD_SECONDS = 0.2
HOLD_SECONDS = 0.4
_WEAK_BOUNDARY_WORDS = {
    "en": frozenset({"a", "an", "and", "but", "for", "in", "it", "of", "on", "or",
                     "that", "the", "these", "those", "to", "we", "which", "who", "with"}),
    "ru": frozenset({"а", "в", "для", "за", "и", "из", "их", "к", "когда", "которые",
                     "который", "которая", "которое", "на", "но", "о", "об", "он",
                     "она", "они", "от", "по", "с", "у", "что", "чтобы", "это"}),
}
_BOUNDARY_PUNCTUATION = (",", ";", ":", ".", "?", "!", "…")


def _validate_schema(document: dict[str, Any], path: Path) -> None:
    schema = json.loads(path.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    errors = list(Draft202012Validator(schema).iter_errors(document))
    if errors:
        first = errors[0]
        raise ValueError(f"Invalid {path.name} at {'.'.join(map(str, first.path)) or 'root'}: {first.message}")


def _time(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{label} must be finite")
    return float(value)


def validate_selection_options(max_count: int, min_gap_seconds: float,
                               timeout_seconds: float = 300) -> None:
    if isinstance(max_count, bool) or not isinstance(max_count, int) or max_count < 0:
        raise ValueError("--emphasis-max-count must be a non-negative integer")
    if _time(min_gap_seconds, "Emphasis minimum gap") < 0:
        raise ValueError("--emphasis-min-gap-seconds must be finite and non-negative")
    if _time(timeout_seconds, "Emphasis timeout") <= 0:
        raise ValueError("--emphasis-timeout must be finite and positive")


def automatic_max_count(duration: float) -> int:
    value = _time(duration, "Emphasis duration")
    if value <= 0:
        raise ValueError("Emphasis duration must be positive")
    return min(12, max(1, round(value / 20)))


def normalized_text(text: str) -> str:
    """Unicode-safe duplicate key: casefold, discard punctuation, normalize spaces."""
    return " ".join("".join(char if not unicodedata.category(char).startswith("P") else " "
                            for char in text.casefold()).split())


def _weak_boundary_token(text: str, language: str) -> bool:
    language_code = language.casefold().split("-", 1)[0]
    return normalized_text(text) in _WEAK_BOUNDARY_WORDS.get(language_code, ())


def _boundary_score(tokens: list[Any], cue_first: int, cue_last: int,
                    start: int, end: int, language: str) -> float:
    """Auditable cue/connector bonuses and weak phrase-edge penalties."""
    score = 0.0
    if start == cue_first:
        score += 0.7
    elif (tokens[start - 1].text.endswith(_BOUNDARY_PUNCTUATION) or
          _weak_boundary_token(tokens[start - 1].text, language)):
        score += 0.45
    if end == cue_last:
        score += 0.7
    elif (tokens[end - 1].text.endswith(_BOUNDARY_PUNCTUATION) or
          tokens[end].text.startswith(_BOUNDARY_PUNCTUATION) or
          _weak_boundary_token(tokens[end].text, language)):
        score += 0.45
    if _weak_boundary_token(tokens[start].text, language):
        score -= 1.25
    if _weak_boundary_token(tokens[end - 1].text, language):
        score -= 1.25
    return round(score, 2)


def _phrase_text(tokens: list[Any]) -> str:
    import re
    return re.sub(r"\s+([,.;:!?…])", r"\1", " ".join(token.text for token in tokens))


def _cue_token_ranges(tokens: list[Any], caption_timeline: dict[str, Any]) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    cursor = 0
    for cue in caption_timeline["cues"]:
        start = cursor
        while cursor < len(tokens):
            cursor += 1
            phrase = _phrase_text(tokens[start:cursor])
            if phrase == cue["text"]:
                ranges.append((start, cursor))
                break
            if not cue["text"].startswith(phrase):
                raise ValueError("Caption cue text does not match contiguous transcript tokens")
        else:
            raise ValueError("Caption cue text does not match transcript tokens")
    if cursor != len(tokens):
        raise ValueError("Caption timeline omitted transcript tokens")
    return ranges


def visible_aroll_regions(retimed: dict[str, Any]) -> list[dict[str, Any]]:
    """Subtract B-roll coverage within each distinct canonical primary keep segment."""
    duration = _time(retimed["duration"], "Final duration")
    overlays = []
    for overlay in retimed["visual_overlays"]:
        start = _time(overlay["timeline_start"], "B-roll start")
        end = _time(overlay["timeline_end"], "B-roll end")
        if overlay["visual_type"] != "B_ROLL" or not 0 <= start < end <= duration:
            raise ValueError("Invalid B-roll overlay for emphasis planning")
        overlays.append((start, end))
    regions: list[dict[str, Any]] = []
    for index, keep in enumerate(retimed["base_video"]["keep_segments"], 1):
        start = _time(keep["edited_start"], "Keep start")
        end = _time(keep["edited_end"], "Keep end")
        cursor = start
        for covered_start, covered_end in sorted(overlays):
            if covered_end <= cursor or covered_start >= end:
                continue
            if covered_start > cursor:
                regions.append({"keep_segment_index": index, "start": round(cursor, 3),
                                "end": round(min(covered_start, end), 3)})
            cursor = max(cursor, min(covered_end, end))
            if cursor >= end:
                break
        if cursor < end:
            regions.append({"keep_segment_index": index, "start": round(cursor, 3),
                            "end": round(end, 3)})
    return regions


def _display_window(speech_start: float, speech_end: float,
                    region: dict[str, Any]) -> tuple[float, float] | None:
    if (speech_start < region["start"] or speech_end > region["end"] or
            speech_end <= speech_start or speech_end - speech_start > MAX_DISPLAY_SECONDS):
        return None
    target = min(MAX_DISPLAY_SECONDS, max(MIN_DISPLAY_SECONDS,
                                         speech_end - speech_start + LEAD_SECONDS + HOLD_SECONDS))
    if region["end"] - region["start"] < target - 0.0005:
        return None
    # Center the readable window on the phrase, then clamp to the visible region.
    center = (speech_start + speech_end + HOLD_SECONDS - LEAD_SECONDS) / 2
    start = max(region["start"], min(center - target / 2, region["end"] - target))
    end = start + target
    if start > speech_start or end < speech_end:
        return None
    return round(start, 3), round(end, 3)


def _candidate_document(project_name: str, transcript: dict[str, Any],
                        retime_map: dict[str, Any], caption_timeline: dict[str, Any],
                        retimed: dict[str, Any]) -> dict[str, Any]:
    validate_caption_inputs(project_name, transcript, retime_map, retimed)
    validate_caption_timeline(caption_timeline, project_name, retimed)
    if caption_timeline != build_caption_timeline(project_name, transcript, retime_map, retimed,
                                                  project_language=caption_timeline["language"]):
        raise ValueError("Saved caption timeline does not match transcript and retime map")
    if abs(caption_timeline["duration"] - retime_map["edited_duration"]) > TIMING_TOLERANCE:
        raise ValueError("Caption and retime durations disagree")
    tokens = retimed_caption_tokens(transcript, retime_map)
    ranges = _cue_token_ranges(tokens, caption_timeline)
    regions = visible_aroll_regions(retimed)
    candidates: list[dict[str, Any]] = []
    for cue, (first, last) in zip(caption_timeline["cues"], ranges):
        for start_index in range(first, last):
            for end_index in range(start_index + 1, min(last, start_index + MAX_WORDS) + 1):
                phrase_tokens = tokens[start_index:end_index]
                text = _phrase_text(phrase_tokens)
                meaningful = "".join(char for char in text if char.isalnum())
                if len(text) > MAX_CHARACTERS or len(meaningful) < 2:
                    continue
                speech_start, speech_end = phrase_tokens[0].start, phrase_tokens[-1].end
                if not cue["start"] <= speech_start < speech_end <= cue["end"]:
                    continue
                region = next((item for item in regions if all(
                    item["start"] <= token.start <= token.end <= item["end"]
                    for token in phrase_tokens)), None)
                if region is None:
                    continue
                window = _display_window(speech_start, speech_end, region)
                if window is None:
                    continue
                candidates.append({
                    "id": f"emphasis_candidate_{len(candidates) + 1:03d}",
                    "cue_index": cue["index"], "token_start_index": start_index,
                    "token_end_index": end_index, "text": text,
                    "word_count": end_index - start_index,
                    "boundary_score": _boundary_score(tokens, first, last, start_index,
                                                      end_index, caption_timeline["language"]),
                    "speech_start": speech_start, "speech_end": speech_end,
                    "eligible_visual_start": region["start"],
                    "eligible_visual_end": region["end"],
                    "timeline_start": window[0], "timeline_end": window[1],
                })
    return {"schema_version": SCHEMA_VERSION, "project_name": project_name,
            "duration": retimed["duration"], "language": caption_timeline["language"],
            "source_transcript": "transcript.json", "source_retime_map": "retime_map.json",
            "source_caption_timeline": "caption_timeline.json",
            "source_retimed_edit_timeline": "retimed_edit_timeline.json",
            "caption_cues": caption_timeline["cues"],
            "visible_regions": regions, "candidates": candidates}


def build_emphasis_candidates(project_name: str, transcript: dict[str, Any],
                              retime_map: dict[str, Any], caption_timeline: dict[str, Any],
                              retimed: dict[str, Any]) -> dict[str, Any]:
    document = _candidate_document(project_name, transcript, retime_map, caption_timeline, retimed)
    _validate_schema(document, CANDIDATES_SCHEMA)
    return document


def validate_emphasis_candidates(document: dict[str, Any], project_name: str,
                                 transcript: dict[str, Any], retime_map: dict[str, Any],
                                 caption_timeline: dict[str, Any], retimed: dict[str, Any]) -> dict[str, Any]:
    _validate_schema(document, CANDIDATES_SCHEMA)
    if document != _candidate_document(project_name, transcript, retime_map, caption_timeline, retimed):
        raise ValueError("Emphasis candidates differ from deterministic transcript and timeline derivation")
    return document


def build_emphasis_plan(project_name: str, candidates: dict[str, Any],
                        candidate_ids: list[str], provider: str, max_count: int,
                        min_gap_seconds: float, input_sha256: dict[str, str]) -> dict[str, Any]:
    validate_selection_options(max_count, min_gap_seconds)
    effective_max = max_count or automatic_max_count(candidates["duration"])
    by_id = {candidate["id"]: candidate for candidate in candidates["candidates"]}
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("Duplicate emphasis candidate IDs")
    if len(candidate_ids) > effective_max:
        raise ValueError("Too many emphasis candidates selected")
    items = []
    for index, candidate_id in enumerate(candidate_ids, 1):
        if candidate_id not in by_id:
            raise ValueError(f"Unknown emphasis candidate ID: {candidate_id}")
        candidate = by_id[candidate_id]
        items.append({"index": index, "candidate_id": candidate_id,
                      **{key: candidate[key] for key in
                         ("text", "speech_start", "speech_end", "timeline_start", "timeline_end")}})
    document = {"schema_version": SCHEMA_VERSION, "project_name": project_name,
                "provider": provider, "duration": candidates["duration"],
                "source_candidates": "emphasis_candidates.json", "max_count": effective_max,
                "min_gap_seconds": min_gap_seconds, "input_sha256": input_sha256,
                "items": items}
    return validate_emphasis_plan(document, candidates)


def validate_emphasis_plan(document: dict[str, Any], candidates: dict[str, Any]) -> dict[str, Any]:
    _validate_schema(document, PLAN_SCHEMA)
    if (document["project_name"] != candidates["project_name"] or
            abs(document["duration"] - candidates["duration"]) > TIMING_TOLERANCE or
            len(document["items"]) > document["max_count"]):
        raise ValueError("Emphasis plan project, duration, or maximum count mismatch")
    min_gap = _time(document["min_gap_seconds"], "Emphasis minimum gap")
    if min_gap < 0:
        raise ValueError("Emphasis minimum gap must be non-negative")
    lookup = {candidate["id"]: candidate for candidate in candidates["candidates"]}
    used_ids: set[str] = set()
    used_text: set[str] = set()
    previous_end = -min_gap
    for index, item in enumerate(document["items"], 1):
        candidate = lookup.get(item["candidate_id"])
        if candidate is None or item["candidate_id"] in used_ids:
            raise ValueError("Unknown or repeated emphasis candidate ID")
        expected = {"index": index, "candidate_id": candidate["id"],
                    **{key: candidate[key] for key in
                       ("text", "speech_start", "speech_end", "timeline_start", "timeline_end")}}
        if item != expected:
            raise ValueError("Emphasis item text, timing, or index differs from candidate")
        if (item["timeline_start"] - previous_end < min_gap - 0.0005 or
                item["timeline_end"] - item["timeline_start"] > MAX_DISPLAY_SECONDS + 0.0005 or
                not candidate["eligible_visual_start"] <= item["timeline_start"] <
                item["timeline_end"] <= candidate["eligible_visual_end"] <= document["duration"]):
            raise ValueError("Emphasis timing violates visible region or minimum gap")
        duplicate_key = normalized_text(item["text"])
        if duplicate_key in used_text:
            raise ValueError("Duplicate normalized emphasis text")
        used_ids.add(item["candidate_id"])
        used_text.add(duplicate_key)
        previous_end = item["timeline_end"]
    return document
