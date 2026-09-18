"""Deterministic and opt-in sandboxed selection of existing emphasis IDs."""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Protocol

from jsonschema import Draft202012Validator

from .emphasis import automatic_max_count, normalized_text, validate_selection_options
from .paths import ROOT

CODEX_SCHEMA = ROOT / "config" / "codex_emphasis_output.schema.json"


class EmphasisSelector(Protocol):
    def select(self, candidates: dict[str, Any], max_count: int,
               min_gap_seconds: float) -> list[str]: ...


def _quality(candidate: dict[str, Any]) -> float:
    text = candidate["text"]
    letters = "".join(char for char in text if char.isalnum())
    if len(letters) < 4:
        return -100
    count = candidate["word_count"]
    score = {1: 2.8, 2: 4.0, 3: 3.8, 4: 2.0, 5: 0.8}[count]
    # A small uncapped-to-36 lexical difference breaks nearby two-word ties;
    # the word-count base still keeps long fragments from winning by length.
    score += min(len(letters), 36) / 16
    score += candidate["boundary_score"]
    if any(char.isdigit() for char in text):
        score += 0.7
    if text.endswith((".", "?", "!", "…")):
        score += 0.2
    if len(text) > 34:
        score -= 1
    return score


class RuleBasedEmphasisSelector:
    """Prefer concise phrases, then spread sparse selections across time."""

    def select(self, candidates: dict[str, Any], max_count: int,
               min_gap_seconds: float) -> list[str]:
        validate_selection_options(max_count, min_gap_seconds)
        limit = max_count or automatic_max_count(candidates["duration"])
        selected: list[dict[str, Any]] = []
        pool = [candidate for candidate in candidates["candidates"] if _quality(candidate) >= 3.0]
        while pool and len(selected) < limit:
            used_text = {normalized_text(item["text"]) for item in selected}
            used_cues = {item["cue_index"] for item in selected}
            available = [item for item in pool if normalized_text(item["text"]) not in used_text
                         and item["cue_index"] not in used_cues
                         and all(item["timeline_end"] + min_gap_seconds <= chosen["timeline_start"] or
                                 chosen["timeline_end"] + min_gap_seconds <= item["timeline_start"]
                                 for chosen in selected)]
            if not available:
                break
            def rank(item: dict[str, Any]) -> tuple[float, float, str]:
                spread = min((abs((item["timeline_start"] + item["timeline_end"] -
                                   chosen["timeline_start"] - chosen["timeline_end"]) / 2)
                              for chosen in selected), default=20.0)
                return (_quality(item) + min(spread, 20) / 15,
                        -item["timeline_start"], item["id"])
            best = max(available, key=rank)
            selected.append(best)
            pool.remove(best)
        return [item["id"] for item in sorted(selected, key=lambda item: item["timeline_start"])]


class CodexEmphasisSelector:
    """One noninteractive read-only Codex call over compact candidate metadata."""

    def __init__(self, model: str | None = None, timeout_seconds: float = 300,
                 schema_path: Path = CODEX_SCHEMA) -> None:
        validate_selection_options(0, 0, timeout_seconds)
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.schema_path = schema_path

    def select(self, candidates: dict[str, Any], max_count: int,
               min_gap_seconds: float) -> list[str]:
        validate_selection_options(max_count, min_gap_seconds, self.timeout_seconds)
        limit = max_count or automatic_max_count(candidates["duration"])
        executable = shutil.which("codex")
        if executable is None:
            raise RuntimeError("Codex CLI is unavailable; use --emphasis-selector rule")
        context = {"duration": candidates["duration"], "language": candidates["language"],
                   "max_count": limit, "min_gap_seconds": min_gap_seconds,
                   "caption_cues": candidates["caption_cues"],
                   "candidates": [{key: candidate[key] for key in
                                   ("id", "cue_index", "text", "speech_start", "speech_end",
                                    "timeline_start", "timeline_end")}
                                  for candidate in candidates["candidates"]]}
        prompt = (
            "Select sparse visual text emphasis for a modern long-form YouTube talking-head video. "
            "This is not Shorts captioning: do not select every sentence. Select phrases that "
            "communicate a clear concept when seen alone on screen without the surrounding sentence. "
            "Prefer named entities, places, people or organizations, concrete nouns, short noun or "
            "topic phrases, numbers actually spoken, and concrete actions or memorable concepts. "
            "A strong shorter phrase is better than a longer phrase that depends on context. "
            "Avoid incomplete grammatical or subordinate-clause fragments, relative or connective "
            "phrases, pronoun-dependent comparisons, generic judgments without their subject, and "
            "vague verbs missing the important noun or topic. Perfect grammar is not required. "
            "Avoid duplicate or repeated ideas. If only weak candidates remain, return fewer items "
            "rather than filler; an empty list is valid. Respect the maximum count, minimum gap, "
            "and supplied display windows. Never translate, rewrite, or join non-contiguous words. "
            "Return only candidate IDs from the supplied set; do not invent text, timing, or IDs. "
            "Do not inspect files, run commands, or access the network. Return JSON matching the schema.\n\n"
            + json.dumps(context, ensure_ascii=False, separators=(",", ":"))
        )
        with tempfile.TemporaryDirectory(prefix="videofactory-emphasis-") as temporary:
            result_path = Path(temporary) / "emphasis_result.json"
            command = [executable, "exec", "--ephemeral", "--sandbox", "read-only",
                       "--output-schema", str(self.schema_path.resolve()),
                       "--output-last-message", str(result_path), "-C", temporary,
                       "--skip-git-repo-check"]
            if self.model:
                command.extend(["-m", self.model])
            command.append("-")
            try:
                result = subprocess.run(command, input=prompt, capture_output=True,
                                        text=True, timeout=self.timeout_seconds, check=False)
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError(f"Codex Emphasis timed out after {self.timeout_seconds:g}s") from exc
            except OSError as exc:
                raise RuntimeError(f"Could not start Codex Emphasis: {exc}") from exc
            if result.returncode != 0:
                tail = (result.stderr or result.stdout or "No diagnostic output").strip()[-2000:]
                raise RuntimeError(f"Codex Emphasis failed (exit {result.returncode}): {tail}")
            if not result_path.is_file():
                raise RuntimeError("Codex Emphasis wrote no final JSON result")
            try:
                response = json.loads(result_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError("Codex Emphasis returned invalid JSON") from exc
        schema = json.loads(self.schema_path.read_text(encoding="utf-8"))
        Draft202012Validator.check_schema(schema)
        errors = list(Draft202012Validator(schema).iter_errors(response))
        if errors:
            raise ValueError(f"Codex Emphasis returned invalid structure: {errors[0].message}")
        return response["candidate_ids"]
