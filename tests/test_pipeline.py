from pathlib import Path

import pytest

from videofactory import SCHEMA_VERSION
from videofactory.asset_manager import discover_assets
from videofactory.models import normalize_transcript, read_json, write_json
from videofactory.paths import ProjectPaths, safe_project_name
from videofactory.renderer import build_render_command
from videofactory.scene_planner import plan_scenes
from videofactory.timeline import build_timeline, validate_timeline
from videofactory.transcriber import FixtureTranscriber


def transcript():
    return normalize_transcript({
        "language": "en",
        "segments": [
            {"start": i * 4, "end": (i + 1) * 4, "text": f"Part {i}"}
            for i in range(5)
        ],
    }, 20)


def sources():
    return {
        "schema_version": SCHEMA_VERSION,
        "sources": [
            {"id": "source_aroll", "kind": "A_ROLL", "path": "/tmp/head.mp4", "duration": 20.0},
            {"id": "broll_001", "kind": "B_ROLL", "path": "/tmp/blue.mp4", "duration": 5.0},
            {"id": "broll_002", "kind": "B_ROLL", "path": "/tmp/green.mp4", "duration": 5.0},
        ],
    }


def settings():
    return {"width": 1920, "height": 1080, "fps": 30}


@pytest.mark.parametrize("name", ["../bad", "bad/name", "", ".hidden", "a b", "a" * 65])
def test_reject_unsafe_project_names(name):
    with pytest.raises(ValueError):
        safe_project_name(name)


def test_project_paths_stay_under_root(tmp_path):
    paths = ProjectPaths(tmp_path, "demo_1")
    paths.create()
    assert paths.project == tmp_path / "projects" / "demo_1"
    assert paths.temp.is_dir() and paths.output.is_dir()


def test_json_schema_roundtrip_and_rejection(tmp_path):
    path = tmp_path / "project.json"
    write_json(path, {"schema_version": 1, "project_name": "demo"})
    assert read_json(path)["project_name"] == "demo"
    with pytest.raises(ValueError):
        write_json(path, {"project_name": "missing-version"})


def test_normalization_preserves_word_timing():
    result = normalize_transcript({
        "language": "en",
        "segments": [{"start": 0, "end": 2, "text": " hello ",
                      "words": [{"start": 0.2, "end": 0.7, "word": "hello"}]}],
    }, 2)
    assert result["segments"][0]["text"] == "hello"
    assert result["segments"][0]["words"][0] == {"start": 0.2, "end": 0.7, "text": "hello"}


def test_fixture_transcriber_needs_no_mlx(tmp_path):
    fixture = tmp_path / "fixture.json"
    fixture.write_text('{"language":"en","segments":[{"start":0,"end":2,"text":"Hi"}]}')
    result = FixtureTranscriber(fixture).transcribe(tmp_path / "unused.wav", 2)
    assert result["segments"][0]["text"] == "Hi"


def test_scene_planner_alternates_aroll_and_local_broll():
    scenes = plan_scenes(transcript(), sources(), "TALKING_HEAD")["scenes"]
    assert [scene["visual_type"] for scene in scenes] == [
        "A_ROLL", "B_ROLL", "A_ROLL", "B_ROLL", "A_ROLL"
    ]
    assert [scene["source_id"] for scene in scenes[1::2]] == ["broll_001", "broll_002"]
    assert [(scene["start"], scene["end"]) for scene in scenes] == [
        (0, 4), (4, 8), (8, 12), (12, 16), (16, 20)
    ]


def test_voiceover_without_assets_uses_graphics():
    empty = {"schema_version": 1, "sources": []}
    scenes = plan_scenes(transcript(), empty, "VOICEOVER")
    assert all(scene["visual_type"] == "GRAPHIC" for scene in scenes["scenes"])


def test_long_transcript_segment_still_allows_periodic_broll():
    long_transcript = normalize_transcript({
        "language": "en",
        "segments": [{"start": 0, "end": 20,
                      "text": "one two three four five six seven eight nine ten"}],
    }, 20)
    scenes = plan_scenes(long_transcript, sources(), "TALKING_HEAD")["scenes"]
    assert [scene["visual_type"] for scene in scenes] == [
        "A_ROLL", "B_ROLL", "A_ROLL", "B_ROLL", "A_ROLL"
    ]
    assert " ".join(scene["narration"] for scene in scenes) == (
        "one two three four five six seven eight nine ten"
    )


def test_timeline_preserves_continuous_original_audio():
    scenes = plan_scenes(transcript(), sources(), "TALKING_HEAD")
    timeline = build_timeline(scenes, sources(), Path("/tmp/head.mp4"), "TALKING_HEAD", settings())
    assert timeline["audio"]["source_path"] == str(Path("/tmp/head.mp4").resolve())
    assert [event["source_in"] for event in timeline["events"]] == [0, 0, 8, 0, 16]
    assert all(event["audio_behavior"] == "original_narration_continuous"
               for event in timeline["events"])
    validate_timeline(timeline)


def test_timeline_rejects_gap():
    scenes = plan_scenes(transcript(), sources(), "TALKING_HEAD")
    timeline = build_timeline(scenes, sources(), Path("/tmp/head.mp4"), "TALKING_HEAD", settings())
    timeline["events"][1]["start"] = 4.5
    with pytest.raises(ValueError, match="gap"):
        validate_timeline(timeline)


def test_local_asset_discovery_probes_only_supported_files(tmp_path, monkeypatch):
    folder = tmp_path / "assets" / "broll"
    folder.mkdir(parents=True)
    (folder / "blue.mp4").touch()
    (folder / "notes.txt").touch()
    monkeypatch.setattr("videofactory.asset_manager.probe",
                        lambda path: {"streams": [{"codec_type": "video", "width": 320,
                                                    "height": 180, "codec_name": "h264"}],
                                      "format": {"duration": "5"}})
    found = discover_assets(tmp_path, "VOICEOVER")
    assert len(found["sources"]) == 1
    assert found["sources"][0]["id"] == "broll_001"
    assert found["sources"][0]["source_url"] is None


def test_render_command_normalizes_and_keeps_one_audio_track(tmp_path):
    scenes = plan_scenes(transcript(), sources(), "TALKING_HEAD")
    timeline = build_timeline(scenes, sources(), Path("/tmp/head.mp4"), "TALKING_HEAD", settings())
    command = build_render_command(timeline, tmp_path / "draft.mp4")
    filters = command[command.index("-filter_complex") + 1]
    assert "scale=1920:1080:force_original_aspect_ratio=increase" in filters
    assert "concat=n=5:v=1:a=0" in filters
    assert "atrim=duration=20.000" in filters
    assert command.count("-stream_loop") == 2
