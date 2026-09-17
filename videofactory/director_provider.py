"""Offline rules and one-shot Codex CLI editorial providers."""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from jsonschema import Draft202012Validator

from .paths import ROOT

CODEX_OUTPUT_SCHEMA_PATH = ROOT / "config" / "codex_director_output.schema.json"


@dataclass(frozen=True)
class DirectorResult:
    plan: dict[str, Any]
    provider: str
    model: str | None
    invocation_count: int
    elapsed_seconds: float


class DirectorProvider(Protocol):
    def create(self, context: dict[str, Any]) -> DirectorResult: ...


def _idea_windows(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Group short segments and split long timed speech near word endings."""
    pieces: list[dict[str, Any]] = []
    for segment in segments:
        start, end = float(segment["start"]), float(segment["end"])
        words = segment.get("words", [])
        if end - start <= 8 or not words:
            pieces.append({"start": start, "end": end, "text": segment["text"]})
            continue
        current_start = start
        current_words: list[dict[str, Any]] = []
        for word in words:
            current_words.append(word)
            word_end = float(word["end"])
            if word_end - current_start >= 5 and end - word_end >= 2.5:
                pieces.append({"start": current_start, "end": word_end,
                               "text": " ".join(item["text"] for item in current_words)})
                current_start = word_end
                current_words = []
        if current_start < end:
            pieces.append({"start": current_start, "end": end,
                           "text": " ".join(item["text"] for item in current_words)})
    groups: list[dict[str, Any]] = []
    for piece in pieces:
        if (groups and groups[-1]["end"] >= piece["start"] - 0.25
                and (groups[-1]["end"] - groups[-1]["start"] < 3
                     or piece["end"] - piece["start"] < 3)
                and piece["end"] - groups[-1]["start"] <= 10):
            groups[-1]["end"] = piece["end"]
            groups[-1]["text"] += " " + piece["text"]
        else:
            groups.append(piece)
    return groups


class RuleBasedDirector:
    """Deterministic editorial fallback; never contacts a model or network."""

    _VISUAL_QUERIES = (
        (("токио", "tokyo"), "modern Tokyo skyline and busy city streets"),
        (("поезд", "транспорт", "train", "transport"), "commuters at a busy train station"),
        (("еда", "кухн", "food"), "Japanese street food prepared at a market"),
        (("улиц", "город", "city", "street"), "busy urban streets and buildings"),
        (("природ", "nature"), "Japanese mountain landscape and forest"),
        (("технолог", "technology"), "modern technology devices in use"),
    )

    def create(self, context: dict[str, Any]) -> DirectorResult:
        started = time.monotonic()
        windows = [
            window for window in _idea_windows(context["transcript"])
            if window["end"] - window["start"] >= 1
        ]
        if not windows:
            windows = [{"start": 0.0, "end": float(context["duration"]),
                        "text": " ".join(segment["text"] for segment in context["transcript"])}]
        shots: list[dict[str, Any]] = []
        for index, window in enumerate(windows):
            text = window["text"].lower()
            is_anchor = context["mode"] == "TALKING_HEAD" and (
                index == 0 or index == len(windows) - 1
                or any(token in text for token in ("я думаю", "я считаю", "мне кажется", "i think", "in my opinion"))
            )
            query = next((description for tokens, description in self._VISUAL_QUERIES
                          if any(token in text for token in tokens)), None)
            if is_anchor:
                visual_type, query, reason = "A_ROLL", None, "Presenter connection anchors this idea"
            elif any(token in text for token in ("фото", "архив", "документ", "photograph", "archive")):
                visual_type, query, reason = "IMAGE", "historical photograph or archive document", "Narration refers to a still source"
            elif any(token in text for token in ("процент", "миллион", "статист", "сравнен", "percent", "million")):
                visual_type, query, reason = "GRAPHIC", None, "Narration contains a statistic or comparison"
            elif query:
                visual_type, reason = "B_ROLL", "Narration describes a concrete visual subject"
            elif context["mode"] == "TALKING_HEAD":
                visual_type, query, reason = "A_ROLL", None, "Keep presenter visible without a concrete visual match"
            else:
                visual_type, query, reason = "GRAPHIC", None, "Voiceover has no concrete visual match"
            shots.append({
                "id": index + 1, "start": round(window["start"], 3),
                "end": round(window["end"], 3), "visual_type": visual_type,
                "visual_query": query, "reason": reason,
            })
        return DirectorResult(
            {"schema_version": 1, "project_name": context["project_name"], "shots": shots},
            "rule", None, 0, round(time.monotonic() - started, 3),
        )


class CodexDirector:
    def __init__(self, model: str | None = None, timeout_seconds: float = 300,
                 schema_path: Path = CODEX_OUTPUT_SCHEMA_PATH) -> None:
        if timeout_seconds <= 0:
            raise ValueError("Director timeout must be positive")
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.schema_path = schema_path

    def create(self, context: dict[str, Any]) -> DirectorResult:
        executable = shutil.which("codex")
        if executable is None:
            raise RuntimeError("Codex CLI is unavailable; install it or use --director rule")
        prompt = (
            "You are VideoFactory's editorial director. Use ONLY the transcript context below. "
            "Do not inspect files, run commands, search for media, or select asset paths. "
            "Return one JSON object matching the supplied schema. Give concise English visual_query "
            "values for B_ROLL and IMAGE. Ground every visual_query in the actual narration: "
            "never invent a specific event, ceremony, person, object, location, food, historical event, "
            "or activity unless it is explicitly stated or directly entailed. Do not narrow a generic "
            "concept into a culturally plausible specific example. Improve search wording without "
            "changing the narration's semantic scope; include only entities and actions explicitly "
            "mentioned or strongly and directly implied. For 'old Japanese traditions', prefer "
            "'traditional Japanese cultural customs', not 'Japanese tea ceremony' or "
            "'Shinto shrine ritual' unless those are named. If narration is too abstract for faithful "
            "B_ROLL or IMAGE, prefer A_ROLL in TALKING_HEAD rather than inventing concrete footage. "
            "Leave silent gaps uncovered. Do not include metadata.\n\n"
            + json.dumps(context, ensure_ascii=False, separators=(",", ":"))
        )
        with tempfile.TemporaryDirectory(prefix="videofactory-director-") as temporary:
            result_path = Path(temporary) / "director_result.json"
            command = [
                executable, "exec", "--ephemeral", "--sandbox", "read-only",
                "--output-schema", str(self.schema_path.resolve()),
                "--output-last-message", str(result_path),
                "-C", temporary, "--skip-git-repo-check",
            ]
            if self.model:
                command.extend(["-m", self.model])
            command.append("-")
            started = time.monotonic()
            try:
                result = subprocess.run(
                    command, input=prompt, capture_output=True, text=True,
                    timeout=self.timeout_seconds, check=False,
                )
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError(f"Codex Director timed out after {self.timeout_seconds:g}s") from exc
            except OSError as exc:
                raise RuntimeError(f"Could not start Codex Director: {exc}") from exc
            elapsed = round(time.monotonic() - started, 3)
            if result.returncode != 0:
                detail = (result.stderr or result.stdout or "No diagnostic output").strip()[-2000:]
                raise RuntimeError(f"Codex Director failed (exit {result.returncode}): {detail}")
            if not result_path.is_file():
                raise RuntimeError("Codex Director wrote no final JSON result")
            try:
                plan = json.loads(result_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError(f"Codex Director returned invalid JSON: {exc}") from exc
            schema = json.loads(self.schema_path.read_text(encoding="utf-8"))
            Draft202012Validator.check_schema(schema)
            errors = list(Draft202012Validator(schema).iter_errors(plan))
            if errors:
                raise ValueError(f"Codex Director returned invalid structure: {errors[0].message}")
        return DirectorResult(plan, "codex", self.model, 1, elapsed)
