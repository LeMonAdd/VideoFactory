"""Offline and opt-in Codex candidate selectors; neither fetches media."""

from __future__ import annotations

import json
import math
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from jsonschema import Draft202012Validator

from .paths import ROOT

CODEX_SELECTION_SCHEMA = ROOT / "config" / "codex_source_selection_output.schema.json"
_STOPWORDS = {"and", "or", "the", "a", "an", "of", "in", "at", "on", "to", "for", "with", "using"}


@dataclass(frozen=True)
class SelectorResult:
    selection: dict[str, Any]
    provider: str
    model: str | None
    invocation_count: int
    elapsed_seconds: float


class SourceSelector(Protocol):
    def select(self, context: dict[str, Any]) -> SelectorResult: ...


def _query_terms(query: str) -> set[str]:
    return {word for word in re.findall(r"[a-z0-9]+", query.casefold()) if word not in _STOPWORDS}


class RuleBasedSourceSelector:
    """Select only explicit tag matches or strong page-slug matches."""

    def select(self, context: dict[str, Any]) -> SelectorResult:
        started = time.monotonic()
        used: set[str] = set()
        selections = []
        for shot in context["shots"]:
            shot_id, visual_type = shot["shot_id"], shot["visual_type"]
            base = {"shot_id": shot_id, "candidate_id": None, "confidence": None,
                    "refined_query": None}
            if visual_type in {"A_ROLL", "GRAPHIC"} or (visual_type == "IMAGE" and shot["search_status"] == "SKIPPED"):
                selections.append(base | {"status": "SKIPPED", "reason": "No candidate selection needed for this shot"})
                continue
            required = _query_terms(shot["visual_query"] or "")
            ranked = []
            for candidate in shot["candidates"]:
                duration = candidate["duration"]
                if visual_type == "B_ROLL" and (duration is None or not math.isfinite(duration) or
                                                duration + 0.001 < shot["shot_duration"]):
                    continue
                # Local candidate queries were explicitly tagged by the user. Pexels slugs are only clues.
                is_local = candidate["provider"] == "local"
                title_terms = _query_terms(candidate["page_title_clue"] or "")
                if not is_local and (not required or not required <= title_terms):
                    continue
                enough_resolution = (candidate["width"] or 0) >= 1920 and (candidate["height"] or 0) >= 1080
                ranked.append((
                    candidate["orientation"] == "landscape", enough_resolution,
                    candidate["candidate_id"] not in used, candidate["candidate_id"], candidate,
                ))
            if not ranked:
                selections.append(base | {"status": "NO_SUITABLE_CANDIDATE",
                                          "reason": "No candidate has sufficient duration and explicit relevance evidence"})
                continue
            best = max(ranked, key=lambda row: row[:-1])[-1]
            used.add(best["candidate_id"])
            selections.append(base | {"status": "SELECTED", "candidate_id": best["candidate_id"],
                                      "confidence": 0.8 if best["provider"] == "local" else 0.65,
                                      "reason": "Explicit local tag match" if best["provider"] == "local" else
                                                "Page title clue matches all query terms; verify footage before use"})
        return SelectorResult({"schema_version": 1, "project_name": context["project_name"],
                               "selections": selections}, "rule", None, 0,
                              round(time.monotonic() - started, 3))


class CodexSourceSelector:
    """One sandboxed Codex CLI call over compact source metadata only."""

    def __init__(self, model: str | None = None, timeout_seconds: float = 300,
                 schema_path: Path = CODEX_SELECTION_SCHEMA) -> None:
        if timeout_seconds <= 0:
            raise ValueError("Selector timeout must be positive")
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.schema_path = schema_path

    def select(self, context: dict[str, Any]) -> SelectorResult:
        executable = shutil.which("codex")
        if executable is None:
            raise RuntimeError("Codex CLI is unavailable; install it or use --selector rule")
        prompt = (
            "Select a candidate for each source-search shot, or reject all as NO_SUITABLE_CANDIDATE. "
            "Return only JSON matching the supplied schema, one selection per shot. "
            "A_ROLL and GRAPHIC must be SKIPPED; IMAGE with skipped search must be SKIPPED. "
            "Use only candidate IDs supplied for that shot. Never invent IDs. "
            "Semantic relevance comes first. A page URL slug is a clue, not guaranteed truth. "
            "Do not infer ethnicity, nationality, location, or activity from creator names, IDs, or generic footage. "
            "For Japanese breakfast narration, generic family breakfast footage does not establish Japan. "
            "Reject if metadata lacks evidence of required entities or actions. Rejection is valid. "
            "A selected video must be at least as long as the target shot; do not assume looping or speed changes. "
            "Then prefer landscape and at least 1920x1080; do not favor 4K merely for being 4K. "
            "Repeated queries may use different equally relevant clips, but relevance matters more than variety. "
            "refined_query is allowed only when status is NO_SUITABLE_CANDIDATE; it may be a "
            "grounded string or null for that status. For status SELECTED you MUST return "
            "refined_query: null. For status SKIPPED you MUST return refined_query: null. "
            "A refined_query is not creative generation or a new search instruction: it is only a "
            "grounded search-query rewrite of that shot's visual_query and narration_context. "
            "It may simplify or reorder wording and use ordinary search-friendly synonyms while "
            "preserving explicit entities, locations, and actions. Do not add facts merely because "
            "they are plausible in the real world. Never add a new person type, family relationship, "
            "demographic group, food, object, ritual, tradition, custom, location, time of day, event, "
            "cultural practice, or activity unless explicitly supported by that shot's visual_query "
            "or narration_context. Candidate metadata, including page titles, URLs, creator names, "
            "and IDs, must not introduce new concepts into refined_query. For 'Japanese people eating "
            "and preparing breakfast', 'breakfast preparation in Japan' is allowed; 'Japanese family "
            "eating breakfast', 'traditional Japanese breakfast', 'Japanese breakfast with rice and "
            "miso soup', and 'traditional Japanese breakfast ritual' are not allowed unless the "
            "visual_query or narration_context explicitly supports those details. If no useful "
            "grounded refinement exists, set refined_query to null. "
            "Do not inspect files, run commands, access the network, or include metadata.\n\n"
            + json.dumps(context, ensure_ascii=False, separators=(",", ":"))
        )
        with tempfile.TemporaryDirectory(prefix="videofactory-selector-") as temporary:
            result_path = Path(temporary) / "selection_result.json"
            command = [executable, "exec", "--ephemeral", "--sandbox", "read-only",
                       "--output-schema", str(self.schema_path.resolve()),
                       "--output-last-message", str(result_path),
                       "-C", temporary, "--skip-git-repo-check"]
            if self.model:
                command.extend(["-m", self.model])
            command.append("-")
            started = time.monotonic()
            try:
                process = subprocess.run(command, input=prompt, capture_output=True, text=True,
                                         timeout=self.timeout_seconds, check=False)
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError(f"Codex Selector timed out after {self.timeout_seconds:g}s") from exc
            except OSError as exc:
                raise RuntimeError(f"Could not start Codex Selector: {exc}") from exc
            elapsed = round(time.monotonic() - started, 3)
            if process.returncode != 0:
                detail = (process.stderr or process.stdout or "No diagnostic output").strip()[-2000:]
                raise RuntimeError(f"Codex Selector failed (exit {process.returncode}): {detail}")
            if not result_path.is_file():
                raise RuntimeError("Codex Selector wrote no final JSON result")
            try:
                selection = json.loads(result_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError(f"Codex Selector returned invalid JSON: {exc}") from exc
            schema = json.loads(self.schema_path.read_text(encoding="utf-8"))
            Draft202012Validator.check_schema(schema)
            errors = list(Draft202012Validator(schema).iter_errors(selection))
            if errors:
                raise ValueError(f"Codex Selector returned invalid structure: {errors[0].message}")
        return SelectorResult(selection, "codex", self.model, 1, elapsed)
