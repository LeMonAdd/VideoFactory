"""V3C planning tests use tiny local files and no external services."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from videofactory import cli
from videofactory.clip_planner import build_clip_plan, resolve_project_media, validate_clip_plan
from videofactory.edit_timeline import build_edit_timeline, validate_edit_timeline
from videofactory.models import read_json, write_json
from videofactory.paths import ROOT
from videofactory.source_models import SourceCandidate, CandidateFile


def fixture(tmp_path: Path, mode: str = "TALKING_HEAD", asset_duration: float = 20.0):
    project_dir = tmp_path / "projects" / "demo"
    media_dir = project_dir / "media" / "broll"
    media_dir.mkdir(parents=True)
    media = media_dir / "pexels_video_42.mp4"
    media.write_bytes(b"tiny local video fixture")
    primary = tmp_path / ("primary.mov" if mode == "TALKING_HEAD" else "voice.wav")
    primary.write_bytes(b"primary fixture")
    project = {"schema_version": 1, "project_name": "demo", "mode": mode,
               "source_media": str(primary)}
    query = "Tokyo city crowds"
    candidate = SourceCandidate(
        candidate_id="pexels:video:42", provider="pexels", provider_asset_id="42",
        media_type="VIDEO", query=query, width=1920, height=1080, duration=20,
        orientation="landscape", files=[CandidateFile("https://example.test/42.mp4", 1920, 1080,
                                                        None, "video/mp4")],
    ).to_dict()
    shots = [
        {"id": 1, "start": 2, "end": 4, "visual_type": "A_ROLL" if mode == "TALKING_HEAD" else "GRAPHIC", "visual_query": None, "reason": "Opening"},
        {"id": 2, "start": 4, "end": 8, "visual_type": "B_ROLL", "visual_query": query, "reason": "City"},
        {"id": 3, "start": 8, "end": 10, "visual_type": "B_ROLL", "visual_query": "Japanese breakfast", "reason": "Food"},
        {"id": 7, "start": 10, "end": 14, "visual_type": "B_ROLL", "visual_query": query, "reason": "City again"},
        {"id": 8, "start": 14, "end": 16, "visual_type": "GRAPHIC", "visual_query": None, "reason": "Numbers"},
    ]
    plan = {"schema_version": 1, "project_name": "demo", "shots": shots}
    requests = []
    for shot in shots:
        found = shot["id"] in (2, 7)
        requests.append({"shot_id": shot["id"], "visual_type": shot["visual_type"],
                         "visual_query": shot["visual_query"],
                         "media_type": "VIDEO" if shot["visual_type"] == "B_ROLL" else None,
                         "status": "FOUND" if found else "NO_RESULTS" if shot["id"] == 3 else "SKIPPED",
                         "candidates": [copy.deepcopy(candidate)] if found else [],
                         "error": None, "reused_from_shot_id": 2 if shot["id"] == 7 else None})
    sources = {"schema_version": 1, "project_name": "demo", "provider": "pexels",
               "sources": [], "requests": requests}
    selections = [{"shot_id": shot["id"],
                   "status": "SELECTED" if shot["id"] in (2, 7) else
                   "NO_SUITABLE_CANDIDATE" if shot["id"] == 3 else "SKIPPED",
                   "candidate_id": "pexels:video:42" if shot["id"] in (2, 7) else None,
                   "reason": "Fixture decision", "confidence": None, "refined_query": None}
                  for shot in shots]
    selection = {"schema_version": 1, "project_name": "demo", "provider": "rule",
                 "selections": selections, "metadata": {"provider": "rule", "model": None,
                                                       "invocation_count": 0, "elapsed_seconds": 0}}
    asset = {"candidate_id": "pexels:video:42", "provider": "pexels", "provider_asset_id": "42",
             "media_type": "VIDEO", "status": "DOWNLOADED", "shot_ids": [2, 7],
             "relative_path": "media/broll/pexels_video_42.mp4", "source_page_url": None,
             "download_url": "https://example.test/42.mp4", "creator": None, "creator_url": None,
             "license_name": None, "license_url": None, "candidate_width": 1920,
             "candidate_height": 1080, "candidate_duration": 20,
             "chosen_file": {"width": 1920, "height": 1080, "quality": None,
                             "file_type": "video/mp4"},
             "width": 1920, "height": 1080, "duration": asset_duration,
             "file_size_bytes": media.stat().st_size,
             "sha256": hashlib.sha256(media.read_bytes()).hexdigest(), "content_type": "video/mp4"}
    manifest = {"schema_version": 1, "project_name": "demo", "assets": [asset]}
    return project_dir, project, plan, sources, selection, manifest


def test_schemas_and_complete_layered_timeline(tmp_path):
    project_dir, project, plan, sources, selection, manifest = fixture(tmp_path)
    for name in ("clip_plan", "edit_timeline"):
        schema = json.loads((ROOT / "config" / f"{name}.schema.json").read_text())
        Draft202012Validator.check_schema(schema)
        assert schema["additionalProperties"] is False
    clips = build_clip_plan(project_dir, "TALKING_HEAD", 20, plan, sources, selection, manifest)
    timeline = build_edit_timeline(project, 20, plan, clips, project_dir)
    assert len(clips["clips"]) == len(plan["shots"])
    assert [item["status"] for item in clips["clips"]] == [
        "SKIPPED", "PLANNED", "A_ROLL_FALLBACK", "PLANNED", "SKIPPED"]
    assert clips["clips"][2]["resolved_visual_type"] == "A_ROLL"
    assert clips["clips"][0]["resolved_visual_type"] == "A_ROLL"
    assert timeline["base_video"]["timeline_start"] == 0
    assert timeline["base_video"]["timeline_end"] == 20
    assert timeline["base_audio"]["audio_policy"] == "KEEP_PRIMARY_CONTINUOUS"
    assert timeline["base_audio"]["timeline_start"] == 0
    assert timeline["base_audio"]["timeline_end"] == 20
    assert timeline["uncovered_visual"] == "PRIMARY"
    assert [item["shot_id"] for item in timeline["visual_overlays"]] == [2, 7, 8]
    assert all(item["source_audio"] is False for item in timeline["visual_overlays"])
    assert all(item["audio_policy"] == "MUTE_SOURCE" for item in timeline["visual_overlays"])
    assert timeline["visual_overlays"][2]["visual_type"] == "GRAPHIC"
    assert timeline["visual_overlays"][2]["source_path"] is None
    validate_clip_plan(clips, plan, "TALKING_HEAD", 20, project_dir)
    validate_edit_timeline(timeline, project, plan, clips, project_dir)


def test_repeated_asset_has_distinct_deterministic_ranges(tmp_path):
    project_dir, _, plan, sources, selection, manifest = fixture(tmp_path)
    first = build_clip_plan(project_dir, "TALKING_HEAD", 20, plan, sources, selection, manifest)
    second = build_clip_plan(project_dir, "TALKING_HEAD", 20, plan, sources, selection, manifest)
    assert first == second
    a, b = first["clips"][1], first["clips"][3]
    assert a["candidate_id"] == b["candidate_id"] == "pexels:video:42"
    assert a["relative_path"] == b["relative_path"]
    assert 0 <= a["source_in"] < a["source_out"] <= 20
    assert 0 <= b["source_in"] < b["source_out"] <= 20
    assert a["source_out"] <= b["source_in"]
    assert not a["source_overlap"] and not b["source_overlap"]
    for clip in (a, b):
        assert abs((clip["source_out"] - clip["source_in"]) - clip["timeline_duration"]) < 0.003
        assert clip["audio_policy"] == "MUTE_SOURCE"


def test_margin_shrinks_for_nearly_full_asset(tmp_path):
    project_dir, _, plan, sources, selection, manifest = fixture(tmp_path, asset_duration=8.2)
    clips = build_clip_plan(project_dir, "TALKING_HEAD", 20, plan, sources, selection, manifest,
                            source_margin_seconds=0.5)["clips"]
    assert 0 < clips[1]["source_in"] < 0.5
    assert clips[1]["source_out"] <= clips[3]["source_in"]
    assert clips[3]["source_out"] <= 8.2


def test_overlap_fallback_when_distinct_windows_cannot_fit(tmp_path):
    project_dir, _, plan, sources, selection, manifest = fixture(tmp_path, asset_duration=6)
    clips = build_clip_plan(project_dir, "TALKING_HEAD", 20, plan, sources, selection, manifest)["clips"]
    assert clips[1]["status"] == clips[3]["status"] == "PLANNED"
    assert clips[1]["source_out"] > clips[3]["source_in"]
    assert clips[3]["source_overlap"] is True
    assert "overlaps" in clips[3]["reason"]


def test_short_asset_or_missing_manifest_entry_falls_back(tmp_path):
    project_dir, _, plan, sources, selection, manifest = fixture(tmp_path, asset_duration=3)
    clips = build_clip_plan(project_dir, "TALKING_HEAD", 20, plan, sources, selection, manifest)["clips"]
    assert clips[1]["status"] == clips[3]["status"] == "A_ROLL_FALLBACK"
    assert clips[1]["source_in"] is None
    manifest["assets"] = []
    clips = build_clip_plan(project_dir, "TALKING_HEAD", 20, plan, sources, selection, manifest)["clips"]
    assert clips[1]["status"] == "A_ROLL_FALLBACK"
    assert "no download manifest entry" in clips[1]["reason"]


def test_missing_file_hash_mismatch_and_unsafe_paths_fail(tmp_path):
    project_dir, _, plan, sources, selection, manifest = fixture(tmp_path)
    media = project_dir / manifest["assets"][0]["relative_path"]
    media.unlink()
    with pytest.raises(FileNotFoundError, match="Downloaded media is missing"):
        build_clip_plan(project_dir, "TALKING_HEAD", 20, plan, sources, selection, manifest)
    media.write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash differs"):
        build_clip_plan(project_dir, "TALKING_HEAD", 20, plan, sources, selection, manifest)
    with pytest.raises(ValueError, match="Unsafe project media path"):
        resolve_project_media(project_dir, "media/broll/../../other.mp4")
    with pytest.raises(ValueError, match="Unsafe project media path"):
        resolve_project_media(project_dir, "/tmp/other.mp4")


def test_symlink_escape_rejected(tmp_path):
    project_dir, _, _, _, _, _ = fixture(tmp_path)
    outside = tmp_path / "outside.mp4"
    outside.write_bytes(b"outside")
    link = project_dir / "media" / "broll" / "link.mp4"
    link.symlink_to(outside)
    with pytest.raises(ValueError, match="symlink"):
        resolve_project_media(project_dir, "media/broll/link.mp4")


def test_selection_and_project_integrity_errors(tmp_path):
    project_dir, _, plan, sources, selection, manifest = fixture(tmp_path)
    selection["selections"][1]["candidate_id"] = "pexels:video:999"
    with pytest.raises(ValueError, match="not available"):
        build_clip_plan(project_dir, "TALKING_HEAD", 20, plan, sources, selection, manifest)
    selection["selections"][1]["candidate_id"] = "pexels:video:42"
    manifest["project_name"] = "other"
    with pytest.raises(ValueError, match="project_name"):
        build_clip_plan(project_dir, "TALKING_HEAD", 20, plan, sources, selection, manifest)


def test_clip_and_overlay_validation_reject_bad_ranges(tmp_path):
    project_dir, project, plan, sources, selection, manifest = fixture(tmp_path)
    clips = build_clip_plan(project_dir, "TALKING_HEAD", 20, plan, sources, selection, manifest)
    timeline = build_edit_timeline(project, 20, plan, clips, project_dir)
    clips["clips"][1]["source_out"] = 99
    with pytest.raises(ValueError, match="Invalid source range"):
        validate_clip_plan(clips, plan, "TALKING_HEAD", 20, project_dir)
    clips = build_clip_plan(project_dir, "TALKING_HEAD", 20, plan, sources, selection, manifest)
    timeline["visual_overlays"][0]["source_audio"] = True
    with pytest.raises(ValueError, match="False|muted"):
        validate_edit_timeline(timeline, project, plan, clips, project_dir)
    timeline = build_edit_timeline(project, 20, plan, clips, project_dir)
    timeline["visual_overlays"][1]["timeline_start"] = 5
    with pytest.raises(ValueError, match="Invalid visual overlay"):
        validate_edit_timeline(timeline, project, plan, clips, project_dir)


def test_voiceover_keeps_audio_and_uses_graphic_fallback(tmp_path):
    project_dir, project, plan, sources, selection, manifest = fixture(tmp_path, mode="VOICEOVER")
    clips = build_clip_plan(project_dir, "VOICEOVER", 20, plan, sources, selection, manifest)
    timeline = build_edit_timeline(project, 20, plan, clips, project_dir)
    assert timeline["base_video"] is None
    assert timeline["base_audio"]["timeline_end"] == 20
    assert timeline["uncovered_visual"] == "GRAPHIC_PLACEHOLDER"
    assert clips["clips"][2]["status"] == "GRAPHIC_FALLBACK"
    assert clips["clips"][2]["resolved_visual_type"] == "GRAPHIC"
    assert [item["shot_id"] for item in timeline["visual_overlays"]] == [1, 2, 3, 7, 8]


def test_cli_builds_only_planning_json_without_rendering(monkeypatch, tmp_path, capsys):
    project_dir, project, plan, sources, selection, manifest = fixture(tmp_path)
    config = tmp_path / "config" / "transcription.json"
    config.parent.mkdir(parents=True)
    write_json(config, {"schema_version": 1, "default_whisper_model": "local/test-model"})
    for filename, data in (("project.json", project), ("director_plan.json", plan),
                           ("sources.json", sources), ("source_selection.json", selection),
                           ("download_manifest.json", manifest)):
        write_json(project_dir / filename, data)
    before = {item.name: item.read_bytes() for item in project_dir.glob("*.json")}
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    monkeypatch.setattr(cli, "probe", lambda _path: {"streams": [
        {"codec_type": "video"}, {"codec_type": "audio"}],
        "format": {"duration": "20"}})
    monkeypatch.setattr(cli, "render", lambda *_args, **_kwargs: pytest.fail("renderer called"))
    monkeypatch.setattr(cli, "PexelsSourceProvider", lambda *_args, **_kwargs: pytest.fail("network provider called"))
    monkeypatch.setattr(cli, "CodexSourceSelector", lambda *_args, **_kwargs: pytest.fail("Codex called"))
    assert cli.main(["--project", "demo", "--build-edit-timeline"]) == 0
    assert all((project_dir / name).read_bytes() == data for name, data in before.items())
    assert {item.name for item in project_dir.glob("*.json")} == set(before) | {"clip_plan.json", "edit_timeline.json"}
    saved_clip = read_json(project_dir / "clip_plan.json")
    saved_edit = read_json(project_dir / "edit_timeline.json")
    assert saved_clip == build_clip_plan(project_dir, "TALKING_HEAD", 20, plan, sources, selection, manifest)
    assert saved_edit == build_edit_timeline(project, 20, plan, saved_clip, project_dir)
    assert "[6/6] Validating and saving edit timeline" in capsys.readouterr().out
    assert not (tmp_path / "output").exists()
    with pytest.raises(SystemExit):
        cli.main(["--project", "demo", "--build-edit-timeline", "--source-margin-seconds", "-1"])
