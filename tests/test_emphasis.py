"""V5B planning tests: synthetic JSON only, no media or real Codex process."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from jsonschema import Draft202012Validator

from videofactory import cli
from videofactory.caption_timeline import build_caption_timeline
from videofactory.caption_timeline import map_caption_time
from videofactory.emphasis import (automatic_max_count, build_emphasis_candidates,
                                   build_emphasis_plan, normalized_text,
                                   validate_emphasis_candidates, validate_emphasis_plan,
                                   validate_selection_options, visible_aroll_regions)
from videofactory.emphasis_selector import (CodexEmphasisSelector, RuleBasedEmphasisSelector,
                                            _quality)
from videofactory.models import read_json, write_json


HASHES = {name: "a" * 64 for name in ("transcript.json", "retime_map.json",
                                      "caption_timeline.json", "retimed_edit_timeline.json",
                                      "emphasis_candidates.json")}


def fixture(tmp_path: Path):
    project = tmp_path / "projects" / "demo"
    project.mkdir(parents=True)
    keeps = [{"source_start": start, "source_end": end, "source_duration": end - start,
              "edited_start": start, "edited_end": end}
             for start, end in ((0, 20), (20, 40), (40, 60))]
    mapping = {"schema_version": 1, "project_name": "demo", "source_duration": 60,
               "edited_duration": 60, "segments": keeps}
    primary = {"role": "PRIMARY", "source_path": "primary.mov", "keep_segments": keeps}
    def overlay(start, end, shot):
        return {"shot_id": shot, "segment_index": 1, "candidate_id": "candidate",
                "visual_type": "B_ROLL", "source_path": "media/broll/a.mp4",
                "original_timeline_start": start, "original_timeline_end": end,
                "timeline_start": start, "timeline_end": end,
                "source_in": 0, "source_out": end - start,
                "source_audio": False, "audio_policy": "MUTE_SOURCE"}
    retimed = {"schema_version": 1, "project_name": "demo", "mode": "TALKING_HEAD",
               "source_duration": 60, "duration": 60, "retime_source": "retime_map.json",
               "base_video": primary, "base_audio": primary,
               "visual_overlays": [overlay(10, 15, 1), overlay(35, 39, 2)]}
    words = []
    for index in range(24):
        start = 1 + index * 2.4
        text = ("новый" if index % 2 == 0 else "город.") if index < 4 else (
            "старые" if index % 2 == 0 else "традиции.")
        words.append({"start": round(start, 3), "end": round(start + 0.8, 3), "text": text})
    transcript = {"schema_version": 1, "duration": 60, "language": "ru",
                  "segments": [{"start": words[0]["start"], "end": words[-1]["end"],
                                "text": " ".join(word["text"] for word in words), "words": words}]}
    caption = build_caption_timeline("demo", transcript, mapping, retimed)
    for name, data in (("transcript.json", transcript), ("retime_map.json", mapping),
                       ("caption_timeline.json", caption), ("retimed_edit_timeline.json", retimed)):
        write_json(project / name, data)
    return project, transcript, mapping, caption, retimed


def candidates(tmp_path):
    project, transcript, mapping, caption, retimed = fixture(tmp_path)
    document = build_emphasis_candidates("demo", transcript, mapping, caption, retimed)
    return project, transcript, mapping, caption, retimed, document


def phrase_candidates(words: list[str], language: str):
    keep = {"source_start": 0, "source_end": 20, "source_duration": 20,
            "edited_start": 0, "edited_end": 20}
    mapping = {"schema_version": 1, "project_name": "phrases", "source_duration": 20,
               "edited_duration": 20, "segments": [keep]}
    primary = {"role": "PRIMARY", "source_path": "primary.mov", "keep_segments": [keep]}
    retimed = {"schema_version": 1, "project_name": "phrases", "mode": "TALKING_HEAD",
               "source_duration": 20, "duration": 20, "retime_source": "retime_map.json",
               "base_video": primary, "base_audio": primary, "visual_overlays": []}
    timed = [{"start": round(1 + i * 0.45, 3), "end": round(1.35 + i * 0.45, 3),
              "text": word} for i, word in enumerate(words)]
    transcript = {"schema_version": 1, "duration": 20, "language": language,
                  "segments": [{"start": timed[0]["start"], "end": timed[-1]["end"],
                                "text": " ".join(words), "words": timed}]}
    caption = build_caption_timeline("phrases", transcript, mapping, retimed)
    return build_emphasis_candidates("phrases", transcript, mapping, caption, retimed)


def by_text(document):
    return {item["text"]: item for item in document["candidates"]}


def test_boundary_metadata_and_lexical_phrase_ranking_are_deterministic():
    words = ["Today", "new", "public", "transit", "but", "tomorrow", "changes."]
    document = phrase_candidates(words, "en")
    assert document == phrase_candidates(words, "en")
    entries = by_text(document)
    assert entries["public transit"]["boundary_score"] > entries["new public"]["boundary_score"]
    assert _quality(entries["public transit"]) > _quality(entries["new public"])
    assert entries["but tomorrow"]["boundary_score"] < entries["tomorrow changes."]["boundary_score"]
    assert entries["transit but"]["boundary_score"] < entries["public transit"]["boundary_score"]
    assert _quality(entries["public transit"]) > _quality(entries["transit but"])
    assert [item["id"] for item in document["candidates"]] == [
        f"emphasis_candidate_{i:03d}" for i in range(1, len(document["candidates"]) + 1)]


def test_weak_boundaries_support_russian_and_unknown_languages_neutrally():
    english_words = ["Today", "new", "public", "transit", "but", "tomorrow", "changes."]
    english = phrase_candidates(english_words, "en")
    unknown = phrase_candidates(english_words, "und")
    en_entries, unknown_entries = by_text(english), by_text(unknown)
    assert en_entries["but tomorrow"]["boundary_score"] < unknown_entries["but tomorrow"]["boundary_score"]
    assert en_entries["transit but"]["boundary_score"] < unknown_entries["transit but"]["boundary_score"]
    assert [item["id"] for item in english["candidates"]] == [
        item["id"] for item in unknown["candidates"]]
    assert [(item["speech_start"], item["speech_end"], item["timeline_start"], item["timeline_end"])
            for item in english["candidates"]] == [
        (item["speech_start"], item["speech_end"], item["timeline_start"], item["timeline_end"])
        for item in unknown["candidates"]]
    russian_words = ["Люди", "любят", "метро", "но", "иногда", "гуляют."]
    russian = by_text(phrase_candidates(russian_words, "ru"))
    neutral = by_text(phrase_candidates(russian_words, "und"))
    assert russian["любят метро"]["boundary_score"] > neutral["любят метро"]["boundary_score"]
    assert russian["но иногда"]["boundary_score"] < neutral["но иногда"]["boundary_score"]
    assert russian["метро но"]["boundary_score"] < neutral["метро но"]["boundary_score"]


def test_candidates_are_deterministic_transcript_spans_and_visible(tmp_path):
    project, transcript, mapping, caption, retimed, document = candidates(tmp_path)
    assert document == build_emphasis_candidates("demo", transcript, mapping, caption, retimed)
    assert validate_emphasis_candidates(document, "demo", transcript, mapping, caption, retimed) == document
    assert [item["id"] for item in document["candidates"]] == [
        f"emphasis_candidate_{index:03d}" for index in range(1, len(document["candidates"]) + 1)]
    assert document["visible_regions"] == [
        {"keep_segment_index": 1, "start": 0, "end": 10},
        {"keep_segment_index": 1, "start": 15, "end": 20},
        {"keep_segment_index": 2, "start": 20, "end": 35},
        {"keep_segment_index": 2, "start": 39, "end": 40},
        {"keep_segment_index": 3, "start": 40, "end": 60}]
    words = transcript["segments"][0]["words"]
    assert document["candidates"]
    for item in document["candidates"]:
        span = words[item["token_start_index"]:item["token_end_index"]]
        assert normalized_text(item["text"]) == normalized_text(" ".join(w["text"] for w in span))
        assert 1 <= item["word_count"] <= 5 and len(item["text"]) <= 48
        assert item["cue_index"] in [cue["index"] for cue in caption["cues"]]
        assert item["eligible_visual_start"] <= item["speech_start"] < item["speech_end"] <= item["eligible_visual_end"]
        assert item["eligible_visual_start"] <= item["timeline_start"] < item["timeline_end"] <= item["eligible_visual_end"]
        assert 1.499 <= item["timeline_end"] - item["timeline_start"] <= 3.001
        assert not any(item["timeline_start"] < o["timeline_end"] and
                       item["timeline_end"] > o["timeline_start"] for o in retimed["visual_overlays"])
    assert read_json(project / "caption_timeline.json") == caption


def test_cue_boundary_broll_and_keep_boundary_are_conservative(tmp_path):
    _, transcript, mapping, caption, retimed, document = candidates(tmp_path)
    for item in document["candidates"]:
        cue = caption["cues"][item["cue_index"] - 1]
        assert cue["start"] <= item["speech_start"] < item["speech_end"] <= cue["end"]
        assert not (item["speech_start"] < 20 < item["speech_end"])
        assert not (item["timeline_start"] < 20 < item["timeline_end"])
    hidden_word_indexes = [i for i, word in enumerate(transcript["segments"][0]["words"])
                           if 10 <= word["start"] < word["end"] <= 15]
    assert not any(item["token_start_index"] in hidden_word_indexes
                   for item in document["candidates"])
    bad = copy.deepcopy(document)
    bad["candidates"][0]["text"] = "invented"
    with pytest.raises(ValueError, match="differ"):
        validate_emphasis_candidates(bad, "demo", transcript, mapping, caption, retimed)
    bad = copy.deepcopy(document)
    bad["candidates"][0]["timeline_start"] += 0.1
    with pytest.raises(ValueError, match="differ"):
        validate_emphasis_candidates(bad, "demo", transcript, mapping, caption, retimed)
    bad = copy.deepcopy(caption)
    bad["duration"] = 59
    with pytest.raises(ValueError):
        build_emphasis_candidates("demo", transcript, mapping, bad, retimed)
    bad = copy.deepcopy(retimed)
    bad["project_name"] = "other"
    with pytest.raises(ValueError):
        build_emphasis_candidates("demo", transcript, mapping, caption, bad)


def test_candidate_speech_uses_caption_collapse_after_a_cut(tmp_path):
    _, transcript, mapping, _, retimed = fixture(tmp_path)
    # Source time 30-31 is removed; edited time and B-roll placement stay frozen.
    mapping["source_duration"] = 61
    mapping["segments"] = [
        {"source_start": 0, "source_end": 30, "source_duration": 30,
         "edited_start": 0, "edited_end": 30},
        {"source_start": 31, "source_end": 61, "source_duration": 30,
         "edited_start": 30, "edited_end": 60}]
    retimed["source_duration"] = 61
    retimed["base_video"]["keep_segments"] = mapping["segments"]
    transcript["duration"] = 61
    for word in transcript["segments"][0]["words"]:
        if word["start"] >= 30:
            word["start"] = round(word["start"] + 1, 3)
            word["end"] = round(word["end"] + 1, 3)
    transcript["segments"][0]["end"] += 1
    caption = build_caption_timeline("demo", transcript, mapping, retimed)
    document = build_emphasis_candidates("demo", transcript, mapping, caption, retimed)
    source_words = transcript["segments"][0]["words"]
    after_cut = [item for item in document["candidates"]
                 if source_words[item["token_start_index"]]["start"] >= 31]
    assert after_cut
    for item in after_cut:
        assert item["speech_start"] == map_caption_time(
            source_words[item["token_start_index"]]["start"], mapping)


def test_fully_hidden_video_produces_no_forced_emphasis(tmp_path):
    _, transcript, mapping, caption, retimed = fixture(tmp_path)
    retimed["visual_overlays"] = [retimed["visual_overlays"][0] | {
        "timeline_start": 0, "timeline_end": 60,
        "original_timeline_start": 0, "original_timeline_end": 60,
        "source_out": 60}]
    document = build_emphasis_candidates("demo", transcript, mapping, caption, retimed)
    assert document["visible_regions"] == [] and document["candidates"] == []
    ids = RuleBasedEmphasisSelector().select(document, automatic_max_count(60), 5)
    assert ids == []
    assert build_emphasis_plan("demo", document, ids, "rule", 0, 5, HASHES)["items"] == []


@pytest.mark.parametrize("max_count,gap", [(-1, 5), (1, -1), (1, float("nan")),
                                            (1, float("inf"))])
def test_invalid_selection_options(max_count, gap):
    with pytest.raises(ValueError):
        validate_selection_options(max_count, gap)


def test_rule_sparsity_duplicates_and_plan_integrity(tmp_path):
    _, _, _, _, _, document = candidates(tmp_path)
    selector = RuleBasedEmphasisSelector()
    assert automatic_max_count(60) == 3
    assert automatic_max_count(240) == 12
    ids = selector.select(document, automatic_max_count(60), 5)
    assert ids == selector.select(document, 3, 5)
    assert ids == selector.select(document, 0, 5)
    assert 0 < len(ids) <= 3
    plan = build_emphasis_plan("demo", document, ids, "rule", 0, 5, HASHES)
    assert plan == build_emphasis_plan("demo", document, ids, "rule", 0, 5, HASHES)
    assert validate_emphasis_plan(plan, document) == plan
    assert len({normalized_text(item["text"]) for item in plan["items"]}) == len(plan["items"])
    assert [item["index"] for item in plan["items"]] == list(range(1, len(plan["items"]) + 1))
    assert [item["timeline_start"] for item in plan["items"]] == sorted(
        item["timeline_start"] for item in plan["items"])
    assert all(b["timeline_start"] - a["timeline_end"] >= 5
               for a, b in zip(plan["items"], plan["items"][1:]))
    assert len(selector.select(document, 1, 0)) <= 1
    with pytest.raises(ValueError, match="Unknown"):
        build_emphasis_plan("demo", document, ["fake"], "codex", 3, 5, HASHES)
    with pytest.raises(ValueError, match="Duplicate"):
        build_emphasis_plan("demo", document, [ids[0], ids[0]], "codex", 3, 5, HASHES)
    with pytest.raises(ValueError, match="Too many"):
        build_emphasis_plan("demo", document, ids[:2], "codex", 1, 5, HASHES)
    bad = copy.deepcopy(plan)
    bad["items"][0]["text"] = "invented"
    with pytest.raises(ValueError, match="differs"):
        validate_emphasis_plan(bad, document)
    # A second occurrence of the same spoken phrase may never be selected twice.
    duplicate_pair = next((a, b) for a in document["candidates"]
                          for b in document["candidates"]
                          if a["id"] != b["id"] and normalized_text(a["text"]) == normalized_text(b["text"])
                          and b["timeline_start"] - a["timeline_end"] >= 5)
    with pytest.raises(ValueError, match="Duplicate normalized"):
        build_emphasis_plan("demo", document, [duplicate_pair[0]["id"], duplicate_pair[1]["id"]],
                            "codex", 3, 5, HASHES)
    bad = copy.deepcopy(plan)
    bad["items"][0]["timeline_end"] += 0.1
    with pytest.raises(ValueError, match="differs"):
        validate_emphasis_plan(bad, document)
    if len(plan["items"]) >= 2:
        bad = copy.deepcopy(plan)
        bad["items"][0]["index"] = 2
        with pytest.raises(ValueError, match="differs"):
            validate_emphasis_plan(bad, document)


def test_codex_uses_safe_command_and_rejects_bad_ids(tmp_path, monkeypatch):
    _, _, _, _, _, document = candidates(tmp_path)
    import videofactory.emphasis_selector as module
    monkeypatch.setattr(module.shutil, "which", lambda name: "/mock/codex")
    seen = {}
    def fake_run(command, **kwargs):
        seen["command"] = command
        seen["kwargs"] = kwargs
        path = Path(command[command.index("--output-last-message") + 1])
        path.write_text(json.dumps({"candidate_ids": ["unknown"]}))
        return SimpleNamespace(returncode=0, stderr="", stdout="")
    monkeypatch.setattr(module.subprocess, "run", fake_run)
    ids = CodexEmphasisSelector(model="test-model", timeout_seconds=12).select(document, 3, 5)
    assert ids == ["unknown"]
    command = seen["command"]
    assert command[:5] == ["/mock/codex", "exec", "--ephemeral", "--sandbox", "read-only"]
    assert "--output-schema" in command and "--skip-git-repo-check" in command
    assert command[-1] == "-" and command[command.index("-m") + 1] == "test-model"
    assert seen["kwargs"]["timeout"] == 12 and seen["kwargs"]["check"] is False
    assert "shell" not in seen["kwargs"]
    assert "source_path" not in seen["kwargs"]["input"]
    assert "caption_cues" in seen["kwargs"]["input"]
    editorial_prompt = seen["kwargs"]["input"]
    assert "clear concept when seen alone on screen" in editorial_prompt
    assert "concrete nouns" in editorial_prompt and "topic phrases" in editorial_prompt
    assert "subordinate-clause fragments" in editorial_prompt
    assert "pronoun-dependent comparisons" in editorial_prompt
    assert "return fewer items rather than filler" in editorial_prompt
    assert "Return only candidate IDs" in editorial_prompt
    with pytest.raises(ValueError, match="Unknown"):
        build_emphasis_plan("demo", document, ids, "codex", 3, 5, HASHES)
    with pytest.raises(ValueError, match="Duplicate"):
        build_emphasis_plan("demo", document, ["emphasis_candidate_001"] * 2,
                            "codex", 3, 5, HASHES)
    with pytest.raises(ValueError, match="Too many"):
        build_emphasis_plan("demo", document,
                            [item["id"] for item in document["candidates"][:4]],
                            "codex", 3, 5, HASHES)


def test_codex_emphasis_schema_is_minimal_structured_output():
    from videofactory.emphasis_selector import CODEX_SCHEMA
    schema = json.loads(CODEX_SCHEMA.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    assert schema["type"] == "object"
    assert schema["additionalProperties"] is False
    assert schema["required"] == ["candidate_ids"]
    assert schema["properties"] == {
        "candidate_ids": {"type": "array", "items": {"type": "string"}}}
    assert "schema_version" not in schema["properties"]
    assert "uniqueItems" not in json.dumps(schema)
    validator = Draft202012Validator(schema)
    assert validator.is_valid({"candidate_ids": []})
    assert validator.is_valid({"candidate_ids": ["one", "one"]})
    assert not validator.is_valid({"candidate_ids": [1]})
    assert not validator.is_valid({"candidate_ids": [], "text": "untrusted"})


@pytest.mark.parametrize("returned,match", [
    (["unknown"], "Unknown"),
    (["emphasis_candidate_001", "emphasis_candidate_001"], "Duplicate"),
    ([f"emphasis_candidate_{index:03d}" for index in range(1, 5)], "Too many"),
])
def test_mocked_codex_ids_are_rejected_by_application(tmp_path, monkeypatch, returned, match):
    _, _, _, _, _, document = candidates(tmp_path)
    import videofactory.emphasis_selector as module
    monkeypatch.setattr(module.shutil, "which", lambda name: "/mock/codex")
    def fake_run(command, **kwargs):
        result_path = Path(command[command.index("--output-last-message") + 1])
        result_path.write_text(json.dumps({"candidate_ids": returned}), encoding="utf-8")
        return SimpleNamespace(returncode=0, stderr="", stdout="")
    monkeypatch.setattr(module.subprocess, "run", fake_run)
    ids = CodexEmphasisSelector().select(document, 3, 5)
    assert ids == returned
    with pytest.raises(ValueError, match=match):
        build_emphasis_plan("demo", document, ids, "codex", 3, 5, HASHES)


def test_cli_rule_route_and_guards(tmp_path, monkeypatch, capsys):
    project, *_ = fixture(tmp_path)
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    monkeypatch.setattr(cli, "default_whisper_model", lambda: "unused-local-model")
    monkeypatch.setattr(cli, "CodexEmphasisSelector", lambda *args, **kwargs:
                        (_ for _ in ()).throw(AssertionError("Codex instantiated in rule mode")))
    import subprocess
    def forbidden(*args, **kwargs):
        raise AssertionError("planning invoked a subprocess")
    monkeypatch.setattr(subprocess, "run", forbidden)
    assert cli.main(["--project", "demo", "--plan-emphasis", "--emphasis-selector", "rule"]) == 0
    assert "Selected emphasis:" in capsys.readouterr().out
    plan = read_json(project / "emphasis_plan.json")
    generated = read_json(project / "emphasis_candidates.json")
    assert plan["provider"] == "rule" and len(plan["items"]) <= 3
    for filename, digest in plan["input_sha256"].items():
        assert digest == hashlib.sha256((project / filename).read_bytes()).hexdigest()
    original = (project / "emphasis_plan.json").read_bytes()
    assert cli.main(["--project", "demo", "--plan-emphasis", "--emphasis-max-count", "1"]) == 0
    assert len(read_json(project / "emphasis_plan.json")["items"]) <= 1
    assert cli.main(["--project", "demo", "--plan-emphasis", "--emphasis-selector", "rule"]) == 0
    assert (project / "emphasis_plan.json").read_bytes() == original
    assert generated == read_json(project / "emphasis_candidates.json")
    for flag in (["--video", "x.mov"], ["--voice", "x.wav"],
                 ["--fixture-transcript", "fixture.json"]):
        with pytest.raises(SystemExit):
            cli.main(["--project", "demo", "--plan-emphasis", *flag])
    for flag in (["--emphasis-selector", "rule"], ["--emphasis-model", "x"],
                 ["--emphasis-max-count", "0"], ["--emphasis-min-gap-seconds", "5"],
                 ["--emphasis-timeout", "300"]):
        with pytest.raises(SystemExit):
            cli.main(["--project", "demo", *flag])
