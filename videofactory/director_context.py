"""Build a small transcript-only request for editorial providers."""

from __future__ import annotations

import math
from typing import Any

from . import SCHEMA_VERSION

EDITORIAL_RULES = [
    "Treat related narration as ideas, not raw Whisper segment boundaries.",
    "For TALKING_HEAD, keep the presenter visible for opening hooks, personal opinions, emotion, transitions, conclusions, and calls to action.",
    "Use B_ROLL for concrete places, food, objects, transport, buildings, technology, nature, history, or processes.",
    "Use IMAGE for specific historical photos, documents, or archive stills.",
    "Use GRAPHIC for statistics, numbers, dates, comparisons, lists, charts, maps, or short quotes.",
    "Do not mechanically alternate A_ROLL and B_ROLL or hide the presenter for nearly the entire video.",
    "Ordinary supporting shots should usually last about 3–8 seconds; A_ROLL may last longer.",
    "Avoid unnecessary one-second cuts; use word timestamps for sensible boundaries within a segment.",
    "B_ROLL and IMAGE visual_query values must be concise English descriptions of concrete visuals.",
    "Do not invent narration for silent gaps. Leave gaps uncovered; VideoFactory fills them later.",
    "Do not select or claim any actual media asset. Plan editorial intent only.",
]


class DirectorContextBuilder:
    def build(self, project_name: str, mode: str, transcript: dict[str, Any]) -> dict[str, Any]:
        if transcript.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("Director context requires a supported transcript schema_version")
        if mode not in {"TALKING_HEAD", "VOICEOVER"}:
            raise ValueError(f"Unsupported project mode: {mode}")
        duration = transcript.get("duration")
        if not isinstance(duration, (int, float)) or isinstance(duration, bool) or not math.isfinite(duration) or duration <= 0:
            raise ValueError("Transcript has no valid positive duration")
        segments = transcript.get("segments")
        if not isinstance(segments, list) or not segments:
            raise ValueError("Transcript has no timed segments")
        compact_segments = []
        previous_end = 0.0
        for item in segments:
            if not isinstance(item, dict):
                raise ValueError("Transcript contains a non-object segment")
            start, end = item.get("start"), item.get("end")
            if (not isinstance(start, (int, float)) or isinstance(start, bool)
                    or not isinstance(end, (int, float)) or isinstance(end, bool)
                    or not math.isfinite(start) or not math.isfinite(end)
                    or start < previous_end - 0.01 or start < 0 or end <= start
                    or end > duration + 0.01 or not isinstance(item.get("text"), str)):
                raise ValueError("Transcript contains invalid or overlapping timed segments")
            segment = {"start": item["start"], "end": item["end"], "text": item["text"]}
            if item.get("words"):
                if not isinstance(item["words"], list):
                    raise ValueError("Transcript words must be a list")
                words = []
                for word in item["words"]:
                    if not isinstance(word, dict):
                        raise ValueError("Transcript contains a non-object word")
                    word_start, word_end = word.get("start"), word.get("end")
                    if (not isinstance(word_start, (int, float)) or isinstance(word_start, bool)
                            or not isinstance(word_end, (int, float)) or isinstance(word_end, bool)
                            or not math.isfinite(word_start) or not math.isfinite(word_end)
                            or word_start < start - 0.01 or word_end > end + 0.01
                            or word_end <= word_start or not isinstance(word.get("text"), str)):
                        raise ValueError("Transcript contains invalid word timestamps")
                    words.append({"start": word_start, "end": word_end, "text": word["text"]})
                segment["words"] = words
            compact_segments.append(segment)
            previous_end = end
        return {
            "project_name": project_name,
            "mode": mode,
            "language": transcript.get("language") or "und",
            "duration": duration,
            "transcript": compact_segments,
            "editorial_rules": EDITORIAL_RULES,
            "response_shape": {
                "schema_version": 1,
                "project_name": project_name,
                "shots": [
                    {"id": 1, "start": 0.0, "end": 3.0,
                     "visual_type": "A_ROLL" if mode == "TALKING_HEAD" else "GRAPHIC",
                     "visual_query": None,
                     "reason": "Brief editorial rationale"}
                ],
            },
        }
