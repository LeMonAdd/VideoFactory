"""V3D graph and publication tests; FFmpeg and ffprobe are always mocked."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from test_edit_timeline import fixture as planning_fixture
from videofactory import cli, edit_renderer
from videofactory.clip_planner import build_clip_plan
from videofactory.edit_timeline import build_edit_timeline
from videofactory.models import read_json, write_json
from videofactory.paths import ROOT


def render_fixture(tmp_path: Path, *, mode: str = "TALKING_HEAD"):
    project_dir, project, plan, sources, selection, manifest = planning_fixture(tmp_path, mode)
    if mode == "TALKING_HEAD":
        plan["shots"] = [shot for shot in plan["shots"] if shot["id"] != 8]
        sources["requests"] = [row for row in sources["requests"] if row["shot_id"] != 8]
        selection["selections"] = [row for row in selection["selections"] if row["shot_id"] != 8]
    clips = build_clip_plan(project_dir, mode, 20, plan, sources, selection, manifest)
    timeline = build_edit_timeline(project, 20, plan, clips, project_dir)
    write_json(project_dir / "download_manifest.json", manifest)
    return project_dir, project, plan, clips, timeline


def fake_probe(path: Path):
    if path.name.endswith(".part.mp4") or path.name == "edited_draft.mp4":
        return {"streams": [
            {"index": 0, "codec_type": "video", "codec_name": "h264", "width": 1920,
             "height": 1080, "avg_frame_rate": "30/1"},
            {"index": 1, "codec_type": "audio", "codec_name": "aac", "duration": "20.0"},
        ], "format": {"duration": "20.0"}}
    if path.name == "primary.mov":
        return {"streams": [
            {"index": 0, "codec_type": "video", "codec_name": "hevc", "width": 1080, "height": 1920},
            {"index": 1, "codec_type": "audio", "codec_name": "aac", "duration": "20.0"},
            {"index": 2, "codec_type": "data"},
        ], "format": {"duration": "20.0"}}
    return {"streams": [{"index": 0, "codec_type": "video", "width": 1920,
                         "height": 1080}, {"index": 1, "codec_type": "audio"}],
            "format": {"duration": "20.0"}}


def patch_media(monkeypatch):
    monkeypatch.setattr(edit_renderer, "probe", fake_probe)
    commands = []
    def fake_run(command):
        commands.append(command)
        Path(command[-1]).write_bytes(b"synthetic edited mp4")
    monkeypatch.setattr(edit_renderer, "run_command", fake_run)
    return commands


def saved_manifest(project_dir: Path):
    return read_json(project_dir / "download_manifest.json")


def test_graph_preserves_primary_video_and_audio_with_repeated_source(tmp_path, monkeypatch):
    project_dir, project, plan, clips, timeline = render_fixture(tmp_path)
    patch_media(monkeypatch)
    primary_info, visual_sources = edit_renderer.inspect_render_inputs(
        project, timeline, clips, plan, read_json(project_dir / "download_manifest.json"), project_dir)
    command = edit_renderer.build_edit_render_command(
        project, timeline, project_dir, tmp_path / "edited_draft.mp4.part.mp4",
        primary_info, visual_sources)
    assert command.count("-i") == 2  # primary plus one physical B-roll input
    assert command[command.index("-i") + 1] == project["source_media"]
    graph = command[command.index("-filter_complex") + 1]
    assert "[0:v:0]setpts=PTS-STARTPTS,fps=30" in graph
    assert "[1:v:0]split=2[src1_0][src1_1]" in graph
    first, second = timeline["visual_overlays"]
    assert f"[src1_0]trim=start={first['source_in']:.3f}:end={first['source_out']:.3f}" in graph
    assert f"[src1_1]trim=start={second['source_in']:.3f}:end={second['source_out']:.3f}" in graph
    assert "setpts=PTS+4.000/TB[shot0]" in graph
    assert "setpts=PTS+10.000/TB[shot1]" in graph
    assert graph.index("[base][shot0]overlay") < graph.index("[layer0][shot1]overlay")
    assert "enable='gte(t,4.000)*lt(t,8.000)'" in graph
    assert "enable='gte(t,10.000)*lt(t,14.000)'" in graph
    assert graph.count("repeatlast=0:eof_action=pass:shortest=0") == 2
    assert graph.count("scale=1920:1080:force_original_aspect_ratio=increase") == 3
    assert graph.count("crop=1920:1080,setsar=1,format=yuv420p") == 3
    assert command[command.index("-map") + 1] == "[vout]"
    assert command[command.index("-map", command.index("-map") + 1) + 1] == "0:1"
    assert "1:1" not in command and "1:a" not in " ".join(command)
    assert "-af" in command and "atrim=duration=20.000,asetpts=PTS-STARTPTS" in command[command.index("-af") + 1]
    assert command[command.index("-c:v") + 1] == "libx264"
    assert command[command.index("-c:a") + 1] == "aac"
    assert command[command.index("-pix_fmt") + 1] == "yuv420p"
    assert command[command.index("-r") + 1] == "30"
    assert command[command.index("-t") + 1] == "20.000"
    assert command[command.index("-map_metadata") + 1] == "-1"
    assert command[command.index("-map_metadata:s:v:0") + 1] == "-1"
    assert command[command.index("-map_metadata:s:a:0") + 1] == "-1"
    assert command[command.index("-map_chapters") + 1] == "-1"
    assert "-dn" in command and "-sn" in command and "+faststart" in command
    assert isinstance(command, list) and all(isinstance(arg, str) for arg in command)


def test_saved_source_ranges_are_used_without_retiming(tmp_path, monkeypatch):
    project_dir, project, plan, clips, timeline = render_fixture(tmp_path)
    timeline["visual_overlays"][0]["source_in"] = 1.25
    timeline["visual_overlays"][0]["source_out"] = 5.25
    clips["clips"][1]["source_in"] = 1.25
    clips["clips"][1]["source_out"] = 5.25
    patch_media(monkeypatch)
    info, sources = edit_renderer.inspect_render_inputs(
        project, timeline, clips, plan, read_json(project_dir / "download_manifest.json"), project_dir)
    command = edit_renderer.build_edit_render_command(project, timeline, project_dir,
                                                       tmp_path / "out.mp4", info, sources)
    graph = command[command.index("-filter_complex") + 1]
    assert "trim=start=1.250:end=5.250" in graph
    assert "setpts=PTS+4.000/TB" in graph
    assert "setpts=PTS+10.000/TB" in graph
    assert "setpts=PTS-STARTPTS" in graph


def test_render_manifest_and_atomic_publication_preserve_v1_draft(tmp_path, monkeypatch):
    project_dir, project, plan, clips, timeline = render_fixture(tmp_path)
    output = tmp_path / "output" / "demo"
    output.mkdir(parents=True)
    (output / "draft.mp4").write_bytes(b"old V1 draft")
    commands = patch_media(monkeypatch)
    manifest = edit_renderer.EditRenderer().render(project_dir, output, project, timeline, clips, plan,
                                                   saved_manifest(project_dir))
    assert len(commands) == 1
    assert commands[0][-1].endswith("edited_draft.mp4.part.mp4")
    assert (output / "edited_draft.mp4").read_bytes() == b"synthetic edited mp4"
    assert not (output / "edited_draft.mp4.part.mp4").exists()
    assert (output / "draft.mp4").read_bytes() == b"old V1 draft"
    assert manifest["broll_overlay_count"] == 2 and manifest["metadata_stripped"] is True
    assert manifest["video_streams"] == manifest["audio_streams"] == 1
    assert manifest["duration"] == 20 and manifest["encoder"] == "libx264"
    assert manifest["file_size_bytes"] == len(b"synthetic edited mp4")
    assert len(manifest["sha256"]) == 64
    assert read_json(project_dir / "render_manifest.json") == manifest
    schema = json.loads((ROOT / "config" / "render_manifest.schema.json").read_text())
    Draft202012Validator.check_schema(schema)


def test_existing_edited_draft_protected_and_overwrite_replaces_only_edited(tmp_path, monkeypatch):
    project_dir, project, plan, clips, timeline = render_fixture(tmp_path)
    output = tmp_path / "output" / "demo"
    output.mkdir(parents=True)
    (output / "edited_draft.mp4").write_bytes(b"previous edit")
    (output / "draft.mp4").write_bytes(b"V1 draft")
    commands = patch_media(monkeypatch)
    renderer = edit_renderer.EditRenderer()
    with pytest.raises(FileExistsError, match="--overwrite-render"):
        renderer.render(project_dir, output, project, timeline, clips, plan, saved_manifest(project_dir))
    assert commands == [] and (output / "edited_draft.mp4").read_bytes() == b"previous edit"
    renderer.render(project_dir, output, project, timeline, clips, plan, saved_manifest(project_dir),
                    encoder="h264_videotoolbox", overwrite=True)
    assert commands[0][commands[0].index("-c:v") + 1] == "h264_videotoolbox"
    assert commands[0][commands[0].index("-b:v") + 1] == "8M"
    assert (output / "edited_draft.mp4").read_bytes() == b"synthetic edited mp4"
    assert (output / "draft.mp4").read_bytes() == b"V1 draft"


def test_ffmpeg_failure_cleans_temp_without_manifest(tmp_path, monkeypatch):
    project_dir, project, plan, clips, timeline = render_fixture(tmp_path)
    output = tmp_path / "output" / "demo"
    monkeypatch.setattr(edit_renderer, "probe", fake_probe)
    def fail(command):
        Path(command[-1]).write_bytes(b"partial")
        raise RuntimeError("mock FFmpeg stderr")
    monkeypatch.setattr(edit_renderer, "run_command", fail)
    with pytest.raises(RuntimeError, match="mock FFmpeg stderr"):
        edit_renderer.EditRenderer().render(project_dir, output, project, timeline, clips, plan,
                                            saved_manifest(project_dir))
    assert not (output / "edited_draft.mp4.part.mp4").exists()
    assert not (output / "edited_draft.mp4").exists()
    assert not (project_dir / "render_manifest.json").exists()


def test_validation_failure_cleans_temp_and_preserves_existing_edit(tmp_path, monkeypatch):
    project_dir, project, plan, clips, timeline = render_fixture(tmp_path)
    output = tmp_path / "output" / "demo"
    output.mkdir(parents=True)
    (output / "edited_draft.mp4").write_bytes(b"good earlier edit")
    patch_media(monkeypatch)
    def bad_probe(path):
        info = fake_probe(path)
        if path.name.endswith(".part.mp4"):
            info["streams"][1]["duration"] = "4.0"
        return info
    monkeypatch.setattr(edit_renderer, "probe", bad_probe)
    with pytest.raises(ValueError, match="audio does not cover"):
        edit_renderer.EditRenderer().render(project_dir, output, project, timeline, clips, plan,
                                            saved_manifest(project_dir),
                                            overwrite=True)
    assert not (output / "edited_draft.mp4.part.mp4").exists()
    assert (output / "edited_draft.mp4").read_bytes() == b"good earlier edit"
    assert not (project_dir / "render_manifest.json").exists()


@pytest.mark.parametrize("change", ["missing_primary", "missing_broll", "unsafe_path", "overlap", "no_audio", "voiceover"])
def test_invalid_inputs_fail_before_ffmpeg(tmp_path, monkeypatch, change):
    project_dir, project, plan, clips, timeline = render_fixture(tmp_path,
        mode="VOICEOVER" if change == "voiceover" else "TALKING_HEAD")
    commands = patch_media(monkeypatch)
    if change == "missing_primary":
        Path(project["source_media"]).unlink()
    elif change == "missing_broll":
        (project_dir / timeline["visual_overlays"][0]["source_path"]).unlink()
    elif change == "unsafe_path":
        timeline["visual_overlays"][0]["source_path"] = "media/broll/../../other.mp4"
        clips["clips"][1]["relative_path"] = "media/broll/../../other.mp4"
    elif change == "overlap":
        timeline["visual_overlays"][1]["timeline_start"] = 7
    elif change == "no_audio":
        def mute_probe(path):
            info = fake_probe(path)
            if path.name == "primary.mov":
                info["streams"] = [item for item in info["streams"] if item["codec_type"] != "audio"]
            return info
        monkeypatch.setattr(edit_renderer, "probe", mute_probe)
    with pytest.raises((ValueError, FileNotFoundError)):
        edit_renderer.EditRenderer().render(project_dir, tmp_path / "output", project,
                                            timeline, clips, plan, saved_manifest(project_dir))
    assert commands == []


def test_graphic_overlay_rejected(tmp_path, monkeypatch):
    project_dir, project, plan, sources, selection, manifest = planning_fixture(tmp_path)
    clips = build_clip_plan(project_dir, "TALKING_HEAD", 20, plan, sources, selection, manifest)
    timeline = build_edit_timeline(project, 20, plan, clips, project_dir)
    commands = patch_media(monkeypatch)
    with pytest.raises(ValueError, match="GRAPHIC"):
        edit_renderer.EditRenderer().render(project_dir, tmp_path / "output", project,
                                            timeline, clips, plan, manifest)
    assert commands == []


def test_paths_with_spaces_and_no_broll_graph(tmp_path, monkeypatch):
    root = tmp_path / "project with spaces"
    project_dir, project, plan, sources, selection, manifest = planning_fixture(root)
    plan["shots"] = [shot for shot in plan["shots"] if shot["id"] not in {2, 7, 8}]
    sources["requests"] = [row for row in sources["requests"] if row["shot_id"] not in {2, 7, 8}]
    selection["selections"] = [row for row in selection["selections"] if row["shot_id"] not in {2, 7, 8}]
    clips = build_clip_plan(project_dir, "TALKING_HEAD", 20, plan, sources, selection, manifest)
    timeline = build_edit_timeline(project, 20, plan, clips, project_dir)
    commands = patch_media(monkeypatch)
    edit_renderer.EditRenderer().render(project_dir, root / "output" / "demo", project,
                                        timeline, clips, plan, manifest)
    assert len(commands) == 1 and commands[0].count("-i") == 1
    assert project["source_media"] in commands[0]
    assert "[base]null[vout]" in commands[0][commands[0].index("-filter_complex") + 1]


def test_output_validation_rejects_extra_streams_and_bad_duration(tmp_path, monkeypatch):
    path = tmp_path / "out.mp4"
    path.write_bytes(b"fixture")
    monkeypatch.setattr(edit_renderer, "probe", lambda _path: fake_probe(Path("edited_draft.mp4")) |
                        {"streams": fake_probe(Path("edited_draft.mp4"))["streams"] + [{"codec_type": "data"}]})
    with pytest.raises(ValueError, match="no data streams"):
        edit_renderer.validate_edited_output(path, 20)
    monkeypatch.setattr(edit_renderer, "probe", lambda _path: fake_probe(Path("edited_draft.mp4")) |
                        {"format": {"duration": "18.0"}})
    with pytest.raises(ValueError, match="duration differs"):
        edit_renderer.validate_edited_output(path, 20)


@pytest.mark.parametrize("field,value", [
    ("codec_name", "hevc"), ("width", 1280), ("height", 720),
    ("avg_frame_rate", "24/1"),
])
def test_output_validation_rejects_wrong_video_standard(tmp_path, monkeypatch, field, value):
    path = tmp_path / "out.mp4"
    path.write_bytes(b"fixture")
    def changed_probe(_path):
        info = fake_probe(Path("edited_draft.mp4"))
        info["streams"][0][field] = value
        return info
    monkeypatch.setattr(edit_renderer, "probe", changed_probe)
    with pytest.raises(ValueError, match="unexpected codec or resolution|frame rate or duration"):
        edit_renderer.validate_edited_output(path, 20)


def test_cli_render_edit_uses_only_saved_state_and_mocked_media(tmp_path, monkeypatch, capsys):
    project_dir, project, plan, clips, timeline = render_fixture(tmp_path)
    config = tmp_path / "config" / "transcription.json"
    config.parent.mkdir(parents=True)
    write_json(config, {"schema_version": 1, "default_whisper_model": "local/model"})
    for name, document in (("project.json", project), ("director_plan.json", plan),
                           ("clip_plan.json", clips), ("edit_timeline.json", timeline)):
        write_json(project_dir / name, document)
    before = {path.name: path.read_bytes() for path in project_dir.glob("*.json")}
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    commands = patch_media(monkeypatch)
    assert cli.main(["--project", "demo", "--render-edit", "--encoder", "libx264"]) == 0
    assert len(commands) == 1
    assert all((project_dir / name).read_bytes() == data for name, data in before.items())
    assert (tmp_path / "output" / "demo" / "edited_draft.mp4").exists()
    assert (project_dir / "render_manifest.json").exists()
    output = capsys.readouterr().out
    assert "[1/5] Loading edit timeline" in output and "[5/5] Validating and saving" in output
    assert "overlay shot 2" in output and "overlay shot 7" in output


def test_cli_render_edit_requires_download_manifest(tmp_path, monkeypatch):
    project_dir, project, plan, clips, timeline = render_fixture(tmp_path)
    for name, document in (("project.json", project), ("director_plan.json", plan),
                           ("clip_plan.json", clips), ("edit_timeline.json", timeline)):
        write_json(project_dir / name, document)
    (project_dir / "download_manifest.json").unlink()
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    commands = patch_media(monkeypatch)
    config = tmp_path / "config" / "transcription.json"
    config.parent.mkdir(parents=True)
    write_json(config, {"schema_version": 1, "default_whisper_model": "local/model"})
    with pytest.raises(FileNotFoundError, match="download_manifest.json"):
        cli.run_render_edit(cli.parser().parse_args(["--project", "demo", "--render-edit"]))
    assert commands == []


@pytest.mark.parametrize("change,expected", [
    ("missing_candidate", "lacks candidate"),
    ("candidate_mismatch", "lacks candidate"),
    ("relative_path", "path mismatch|mapping differs"),
    ("shot_ids", "mapping differs"),
    ("modified_file", "hash differs"),
])
def test_manifest_disagreement_fails_before_ffmpeg(tmp_path, monkeypatch, change, expected):
    project_dir, project, plan, clips, timeline = render_fixture(tmp_path)
    manifest = saved_manifest(project_dir)
    if change == "missing_candidate":
        manifest["assets"] = []
    elif change == "candidate_mismatch":
        clips["clips"][1]["candidate_id"] = "pexels:video:99"
    elif change == "relative_path":
        manifest["assets"][0]["relative_path"] = "media/broll/pexels_video_99.mp4"
    elif change == "shot_ids":
        manifest["assets"][0]["shot_ids"] = [7]
    elif change == "modified_file":
        (project_dir / manifest["assets"][0]["relative_path"]).write_bytes(b"modified")
    commands = patch_media(monkeypatch)
    with pytest.raises(ValueError, match=expected):
        edit_renderer.EditRenderer().render(project_dir, tmp_path / "output", project,
                                            timeline, clips, plan, manifest)
    assert commands == []


def test_overwrite_help_describes_selected_render_mode():
    help_text = " ".join(cli.parser().format_help().split())
    assert "Replace an existing generated output for the selected render mode" in help_text
    assert "Replace an existing generated draft.mp4" not in help_text
