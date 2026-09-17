from __future__ import annotations

import copy

import pytest

from videofactory.director_context import DirectorContextBuilder
from videofactory.director_plan import (
    align_plan_to_words, load_schema, snap_boundary, validate_plan,
)
from videofactory.director_provider import DirectorResult, RuleBasedDirector
from videofactory.director_scenes import convert_plan_to_scenes
from videofactory.director_service import create_director_plan


def transcript():
    return {
        "schema_version": 1, "language": "ru", "duration": 12.0,
        "segments": [
            {"start": 2.0, "end": 6.0, "text": "Я расскажу о Токио",
             "words": [
                 {"start": 2.0, "end": 3.0, "text": "Я"},
                 {"start": 3.0, "end": 4.0, "text": "расскажу"},
                 {"start": 4.0, "end": 5.0, "text": "о"},
                 {"start": 5.0, "end": 6.0, "text": "Токио"},
             ]},
            {"start": 6.0, "end": 10.0, "text": "Токио огромный город",
             "words": [
                 {"start": 6.0, "end": 7.0, "text": "Токио"},
                 {"start": 7.0, "end": 8.0, "text": "огромный"},
                 {"start": 8.0, "end": 10.0, "text": "город"},
             ]},
        ],
    }


def plan():
    return {
        "schema_version": 1, "project_name": "demo",
        "shots": [
            {"id": 1, "start": 2.0, "end": 6.0, "visual_type": "A_ROLL",
             "visual_query": None, "reason": "Presenter opens"},
            {"id": 2, "start": 6.0, "end": 10.0, "visual_type": "B_ROLL",
             "visual_query": "modern Tokyo skyline and busy city streets",
             "reason": "Narration describes the city"},
        ],
    }


def test_schema_is_strict_and_accepts_valid_plan():
    schema = load_schema()
    assert schema["additionalProperties"] is False
    assert schema["properties"]["shots"]["items"]["additionalProperties"] is False
    assert validate_plan(plan(), "demo", 12, "TALKING_HEAD")["shots"][0]["id"] == 1


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda p: p.update(extra=True), "Additional properties"),
        (lambda p: p["shots"][0].update(visual_type="VIDEO"), "not one of"),
        (lambda p: p["shots"][1].update(start=5.0), "overlaps"),
        (lambda p: p["shots"][1].update(end=13.0), "exceeds"),
        (lambda p: p["shots"][1].update(visual_query=None), "requires a visual_query"),
        (lambda p: p["shots"][1].update(end=6.5), "shorter than"),
        (lambda p: p["shots"][0].update(visual_query="presenter"), "visual_query=null"),
    ],
)
def test_invalid_director_plans_are_rejected(change, message):
    candidate = copy.deepcopy(plan())
    change(candidate)
    with pytest.raises(ValueError, match=message):
        validate_plan(candidate, "demo", 12, "TALKING_HEAD")


def test_plan_rejects_wrong_project_and_voiceover_aroll():
    with pytest.raises(ValueError, match="project_name"):
        validate_plan(plan(), "other", 12, "TALKING_HEAD")
    with pytest.raises(ValueError, match="VOICEOVER"):
        validate_plan(plan(), "demo", 12, "VOICEOVER")


def test_boundary_snap_and_deep_word_cut_rejection():
    words = transcript()["segments"][0]["words"]
    assert snap_boundary(5.12, words) == 5.0
    assert snap_boundary(1.0, words) == 1.0
    with pytest.raises(ValueError, match="inside a spoken word"):
        snap_boundary(5.5, words)


def test_plan_alignment_preserves_shared_cut():
    candidate = plan()
    candidate["shots"][0]["end"] = 6.12
    candidate["shots"][1]["start"] = 6.12
    aligned = align_plan_to_words(candidate, transcript())
    assert aligned["shots"][0]["end"] == aligned["shots"][1]["start"] == 6.0
    assert candidate["shots"][0]["end"] == 6.12


def test_context_has_only_transcript_and_editorial_fields():
    context = DirectorContextBuilder().build("demo", "TALKING_HEAD", transcript())
    assert set(context) == {
        "project_name", "mode", "language", "duration", "transcript",
        "editorial_rules", "response_shape",
    }
    assert context["transcript"][0]["words"][0]["text"] == "Я"
    assert "source_media" not in str(context)


def test_rule_director_is_deterministic_and_not_mechanical_alternation():
    context = DirectorContextBuilder().build("demo", "TALKING_HEAD", transcript())
    first = RuleBasedDirector().create(context)
    second = RuleBasedDirector().create(context)
    assert first.plan == second.plan
    assert first.invocation_count == 0
    assert [shot["visual_type"] for shot in first.plan["shots"]] == ["A_ROLL", "A_ROLL"]
    validate_plan(first.plan, "demo", 12, "TALKING_HEAD")


def test_rule_voiceover_never_requests_aroll():
    context = DirectorContextBuilder().build("demo", "VOICEOVER", transcript())
    result = RuleBasedDirector().create(context)
    assert all(shot["visual_type"] != "A_ROLL" for shot in result.plan["shots"])
    validate_plan(result.plan, "demo", 12, "VOICEOVER")


def test_context_rejects_invalid_word_timing_before_provider_call():
    invalid = transcript()
    invalid["segments"][0]["words"][0]["end"] = 99
    with pytest.raises(ValueError, match="word timestamps"):
        DirectorContextBuilder().build("demo", "TALKING_HEAD", invalid)


def test_untrusted_provider_plan_is_validated_before_metadata():
    class BadProvider:
        def create(self, context):
            bad = plan()
            bad["shots"][1]["visual_type"] = "UNKNOWN"
            return DirectorResult(bad, "codex", None, 1, 1.0)

    with pytest.raises(ValueError):
        create_director_plan("demo", "TALKING_HEAD", transcript(), BadProvider())


def test_scene_conversion_fills_gaps_and_preserves_unresolved_query():
    sources = {"schema_version": 1, "sources": [
        {"id": "source_aroll", "kind": "A_ROLL", "path": "/tmp/head.mov"},
    ]}
    scenes = convert_plan_to_scenes(plan(), transcript(), sources, "TALKING_HEAD")["scenes"]
    assert [(scene["start"], scene["end"]) for scene in scenes] == [
        (0.0, 2.0), (2.0, 6.0), (6.0, 10.0), (10.0, 12.0)
    ]
    assert scenes[0]["narration"] == ""
    assert scenes[0]["visual_type"] == scenes[-1]["visual_type"] == "A_ROLL"
    assert scenes[2]["requested_visual_type"] == "B_ROLL"
    assert scenes[2]["visual_type"] == "A_ROLL"
    assert scenes[2]["resolution_status"] == "unresolved_local_asset"
    assert scenes[2]["visual_query"] == "modern Tokyo skyline and busy city streets"


def test_scene_conversion_resolves_explicit_local_query_only():
    sources = {"schema_version": 1, "sources": [
        {"id": "source_aroll", "kind": "A_ROLL", "path": "/tmp/head.mov"},
        {"id": "broll_001", "kind": "B_ROLL", "path": "/tmp/tokyo.mp4",
         "visual_queries": ["modern Tokyo skyline and busy city streets"]},
    ]}
    scenes = convert_plan_to_scenes(plan(), transcript(), sources, "TALKING_HEAD")["scenes"]
    assert scenes[2]["visual_type"] == "B_ROLL"
    assert scenes[2]["source_id"] == "broll_001"
    assert scenes[2]["resolution_status"] == "resolved_local"


def test_voiceover_unresolved_request_becomes_graphic():
    voice_plan = {
        "schema_version": 1, "project_name": "demo",
        "shots": [{"id": 1, "start": 2.0, "end": 10.0,
                   "visual_type": "B_ROLL", "visual_query": "busy Tokyo streets",
                   "reason": "Concrete place"}],
    }
    scenes = convert_plan_to_scenes(
        voice_plan, transcript(), {"schema_version": 1, "sources": []}, "VOICEOVER"
    )["scenes"]
    assert all(scene["visual_type"] == "GRAPHIC" for scene in scenes)
    assert scenes[1]["resolution_status"] == "unresolved_local_asset"
