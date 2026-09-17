"""V3B2 download tests: all HTTP and media probing are mocked."""

from __future__ import annotations

import copy
import hashlib
import json
import ssl
from pathlib import Path
from types import SimpleNamespace
from urllib.error import HTTPError

import certifi
import pytest

from videofactory import cli, source_downloader
from videofactory.models import read_json, write_json
from videofactory.source_downloader import (DownloadError, SelectedMediaDownloader,
                                            checked_https_url, choose_video_variant,
                                            max_download_bytes, selected_assets, validate_manifest)
from videofactory.source_models import SourceCandidate, CandidateFile

MP4_BYTES = b"\x00\x00\x00\x18ftypisom" + b"test-video-body"
URL = "https://videos.pexels.com/video-files/31177207/clip.mp4"


def fixture_documents():
    query = "Tokyo city crowds"
    candidate = SourceCandidate(
        candidate_id="pexels:video:31177207", provider="pexels", provider_asset_id="31177207",
        media_type="VIDEO", query=query,
        page_url="https://www.pexels.com/video/tokyo-crowds-31177207/",
        creator="Creator", creator_url="https://www.pexels.com/@creator/",
        license="Pexels License", license_url="https://www.pexels.com/license/",
        width=3840, height=2160, duration=24, orientation="landscape",
        files=[CandidateFile(URL, 1920, 1080, "hd", "video/mp4")],
    ).to_dict()
    shots = [
        {"id": 1, "start": 0, "end": 2, "visual_type": "A_ROLL", "visual_query": None, "reason": "Intro"},
        {"id": 2, "start": 2, "end": 7, "visual_type": "B_ROLL", "visual_query": query, "reason": "City"},
        {"id": 3, "start": 7, "end": 10, "visual_type": "B_ROLL", "visual_query": "Japanese breakfast", "reason": "Food"},
        {"id": 7, "start": 10, "end": 15, "visual_type": "B_ROLL", "visual_query": query, "reason": "City again"},
        {"id": 8, "start": 15, "end": 17, "visual_type": "GRAPHIC", "visual_query": None, "reason": "Numbers"},
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
    selections = [
        {"shot_id": shot["id"], "status": "SELECTED" if shot["id"] in (2, 7) else
         "NO_SUITABLE_CANDIDATE" if shot["id"] == 3 else "SKIPPED",
         "candidate_id": "pexels:video:31177207" if shot["id"] in (2, 7) else None,
         "reason": "Fixture decision", "confidence": None, "refined_query": None}
        for shot in shots
    ]
    selection = {"schema_version": 1, "project_name": "demo", "provider": "rule",
                 "selections": selections, "metadata": {"provider": "rule", "model": None,
                                                       "invocation_count": 0, "elapsed_seconds": 0}}
    return plan, sources, selection


class FakeResponse:
    def __init__(self, body=MP4_BYTES, content_type="video/mp4", content_length=None,
                 final_url=URL, status=200):
        self.body = body
        self.position = 0
        self.headers = {"Content-Type": content_type}
        if content_length is not None:
            self.headers["Content-Length"] = str(content_length)
        self.final_url = final_url
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def geturl(self):
        return self.final_url

    def read(self, size):
        data = self.body[self.position:self.position + size]
        self.position += len(data)
        return data


def good_probe(_path):
    return {"streams": [{"codec_type": "video", "width": 1920, "height": 1080}],
            "format": {"duration": "24.0"}}


def test_variant_ranking_prefers_1080p_over_4k_and_720p():
    files = [
        {"url": "https://v.example/720.mp4", "width": 1280, "height": 720, "file_type": "video/mp4"},
        {"url": "https://v.example/4k.mp4", "width": 3840, "height": 2160, "file_type": "video/mp4"},
        {"url": "https://v.example/1080.mp4", "width": 1920, "height": 1080, "file_type": "video/mp4"},
        {"url": "https://v.example/not-mp4.webm", "width": 1920, "height": 1080, "file_type": "video/webm"},
    ]
    assert choose_video_variant(files)["url"].endswith("1080.mp4")
    assert choose_video_variant([files[0], files[1]])["url"].endswith("4k.mp4")
    assert choose_video_variant([files[0]])["url"].endswith("720.mp4")
    with pytest.raises(DownloadError, match="no usable MP4"):
        choose_video_variant([files[3]])


@pytest.mark.parametrize("bad", ["http://example.com/file.mp4", "file:///tmp/file.mp4",
                                    "ftp://example.com/file.mp4", "https://user:pass@example.com/file.mp4",
                                    "https://", "https://host:bad/file.mp4", "https://host/file.mp4\nX:1"])
def test_url_validation_rejects_unsafe_urls(bad):
    with pytest.raises(DownloadError):
        checked_https_url(bad)
    assert checked_https_url(URL) == URL


def test_selected_only_and_deduplicated_download_with_provenance(monkeypatch, tmp_path):
    plan, sources, selection = fixture_documents()
    project = tmp_path / "demo"
    project.mkdir()
    calls = []
    def fake_open(request, timeout, context):
        calls.append((request, timeout, context))
        return FakeResponse(content_length=len(MP4_BYTES))
    monkeypatch.setattr(source_downloader, "urlopen", fake_open)
    monkeypatch.setattr(source_downloader, "probe", good_probe)
    monkeypatch.setenv("PEXELS_API_KEY", "private-test-key")
    downloader = SelectedMediaDownloader()
    assert len(selected_assets(plan, sources, selection)) == 1
    manifest = downloader.run(project, plan, sources, selection)
    assert len(calls) == 1
    request, timeout, context = calls[0]
    assert request.full_url == URL and request.get_header("User-agent") == "VideoFactory/1.0"
    assert request.get_header("Authorization") is None
    assert isinstance(context, ssl.SSLContext) and context.verify_mode == ssl.CERT_REQUIRED
    assert context.check_hostname and timeout == 30
    assert certifi.where()
    asset = manifest["assets"][0]
    assert asset["status"] == "DOWNLOADED" and asset["shot_ids"] == [2, 7]
    assert asset["relative_path"] == "media/broll/pexels_video_31177207.mp4"
    assert asset["sha256"] == hashlib.sha256(MP4_BYTES).hexdigest()
    assert asset["file_size_bytes"] == len(MP4_BYTES)
    assert asset["source_page_url"] == "https://www.pexels.com/video/tokyo-crowds-31177207/"
    assert asset["creator"] == "Creator" and asset["license_name"] == "Pexels License"
    assert asset["license_url"] == "https://www.pexels.com/license/"
    assert asset["chosen_file"]["width"] == 1920 and asset["candidate_width"] == 3840
    assert asset["width"] == 1920 and asset["duration"] == 24
    assert "private-test-key" not in json.dumps(manifest)
    final = project / asset["relative_path"]
    assert final.read_bytes() == MP4_BYTES and not final.with_suffix(".mp4.part").exists()
    assert read_json(project / "download_manifest.json") == manifest
    validate_manifest(manifest, "demo")


def test_downloader_uses_certifi_bundle(monkeypatch):
    observed = []
    real_create_context = ssl.create_default_context
    def tracked_context(*, cafile):
        observed.append(cafile)
        return real_create_context(cafile=cafile)
    monkeypatch.setattr(source_downloader.ssl, "create_default_context", tracked_context)
    downloader = SelectedMediaDownloader()
    assert observed == [certifi.where()]
    assert downloader._ssl_context.verify_mode == ssl.CERT_REQUIRED
    assert downloader._ssl_context.check_hostname is True


def test_valid_existing_file_reused_without_network(monkeypatch, tmp_path):
    plan, sources, selection = fixture_documents()
    project = tmp_path / "demo"
    project.mkdir()
    monkeypatch.setattr(source_downloader, "urlopen", lambda *_args, **_kwargs: FakeResponse())
    monkeypatch.setattr(source_downloader, "probe", good_probe)
    first = SelectedMediaDownloader().run(project, plan, sources, selection)
    monkeypatch.setattr(source_downloader, "urlopen", lambda *_args, **_kwargs: pytest.fail("network used"))
    second = SelectedMediaDownloader().run(project, plan, sources, selection)
    assert first["assets"][0]["status"] == "DOWNLOADED"
    assert second["assets"][0]["status"] == "REUSED"
    assert second["assets"][0]["shot_ids"] == [2, 7]


def test_corrupt_existing_file_and_unmanifested_file_rejected(monkeypatch, tmp_path):
    plan, sources, selection = fixture_documents()
    project = tmp_path / "demo"
    project.mkdir()
    monkeypatch.setattr(source_downloader, "urlopen", lambda *_args, **_kwargs: FakeResponse())
    monkeypatch.setattr(source_downloader, "probe", good_probe)
    downloader = SelectedMediaDownloader()
    prior = downloader.run(project, plan, sources, selection)
    final = project / prior["assets"][0]["relative_path"]
    final.write_bytes(b"corrupt")
    monkeypatch.setattr(source_downloader, "urlopen", lambda *_args, **_kwargs: pytest.fail("network used"))
    with pytest.raises(DownloadError, match="hash differs"):
        downloader.run(project, plan, sources, selection)
    assert final.read_bytes() == b"corrupt"
    (project / "download_manifest.json").unlink()
    with pytest.raises(DownloadError, match="no matching manifest"):
        downloader.run(project, plan, sources, selection)


@pytest.mark.parametrize("response,limit,message", [
    (FakeResponse(content_length=999999999), 1, "Content-Length"),
    (FakeResponse(body=MP4_BYTES * 100000), 0.0001, "stream exceeds"),
    (FakeResponse(body=b""), 1, "empty"),
    (FakeResponse(body=b"<html>error</html>"), 1, "HTML/JSON"),
    (FakeResponse(body=b'{"error":"bad"}', content_type="application/octet-stream"), 1, "HTML/JSON"),
    (FakeResponse(body=b"not an mp4", content_type="application/octet-stream"), 1, "does not look like MP4"),
    (FakeResponse(body=MP4_BYTES, content_type="text/html"), 1, "content type"),
    (FakeResponse(body=MP4_BYTES, final_url="http://example.com/clip.mp4"), 1, "HTTPS"),
    (FakeResponse(body=MP4_BYTES, status=500), 1, "HTTP 500"),
])
def test_failed_download_cleans_part_and_preserves_manifest(monkeypatch, tmp_path, response, limit, message):
    plan, sources, selection = fixture_documents()
    project = tmp_path / "demo"
    project.mkdir()
    old = {"schema_version": 1, "project_name": "demo", "assets": []}
    write_json(project / "download_manifest.json", old)
    monkeypatch.setattr(source_downloader, "urlopen", lambda *_args, **_kwargs: response)
    monkeypatch.setattr(source_downloader, "probe", good_probe)
    with pytest.raises(DownloadError, match=message) as caught:
        SelectedMediaDownloader(max_download_mb=limit).run(project, plan, sources, selection)
    assert "pexels:video:31177207" in str(caught.value)
    assert not list(project.rglob("*.part"))
    assert not list(project.rglob("*.mp4"))
    assert read_json(project / "download_manifest.json") == old


def test_probe_failure_cleans_part(monkeypatch, tmp_path):
    plan, sources, selection = fixture_documents()
    project = tmp_path / "demo"
    project.mkdir()
    monkeypatch.setattr(source_downloader, "urlopen", lambda *_args, **_kwargs: FakeResponse())
    monkeypatch.setattr(source_downloader, "probe", lambda _path: {"streams": [], "format": {"duration": "10"}})
    with pytest.raises(DownloadError, match="ffprobe validation failed"):
        SelectedMediaDownloader().run(project, plan, sources, selection)
    assert not list(project.rglob("*.part")) and not list(project.rglob("*.mp4"))


def test_second_asset_failure_cleans_first_staged_asset(monkeypatch, tmp_path):
    plan, sources, selection = fixture_documents()
    second = copy.deepcopy(sources["requests"][1]["candidates"][0])
    second["candidate_id"] = "pexels:video:42"
    second["provider_asset_id"] = "42"
    second["query"] = "Japanese breakfast"
    second["files"][0]["url"] = "https://videos.pexels.com/video-files/42/clip.mp4"
    sources["requests"][2].update(status="FOUND", candidates=[second])
    selection["selections"][2].update(status="SELECTED", candidate_id="pexels:video:42")
    responses = iter((FakeResponse(), FakeResponse(body=b"<html>error</html>")))
    monkeypatch.setattr(source_downloader, "urlopen", lambda *_args, **_kwargs: next(responses))
    monkeypatch.setattr(source_downloader, "probe", good_probe)
    project = tmp_path / "demo"
    project.mkdir()
    with pytest.raises(DownloadError, match="pexels:video:42"):
        SelectedMediaDownloader().run(project, plan, sources, selection)
    assert not list(project.rglob("*.part")) and not list(project.rglob("*.mp4"))
    assert not (project / "download_manifest.json").exists()


def test_bad_candidate_url_and_invented_id_blocked_before_network(monkeypatch, tmp_path):
    plan, sources, selection = fixture_documents()
    project = tmp_path / "demo"
    project.mkdir()
    monkeypatch.setattr(source_downloader, "urlopen", lambda *_args, **_kwargs: pytest.fail("network used"))
    selection["selections"][1]["candidate_id"] = "pexels:video:999"
    with pytest.raises(ValueError, match="not available"):
        SelectedMediaDownloader().run(project, plan, sources, selection)
    selection["selections"][1]["candidate_id"] = "pexels:video:31177207"
    for request in sources["requests"]:
        for item in request["candidates"]:
            item["files"][0]["url"] = "file:///tmp/secret.mp4"
    with pytest.raises(DownloadError, match="HTTPS"):
        SelectedMediaDownloader().run(project, plan, sources, selection)


def test_no_selected_assets_writes_empty_manifest_without_network(monkeypatch, tmp_path):
    plan, sources, selection = fixture_documents()
    for item in selection["selections"]:
        if item["status"] == "SELECTED":
            item.update(status="NO_SUITABLE_CANDIDATE", candidate_id=None)
    monkeypatch.setattr(source_downloader, "urlopen", lambda *_args, **_kwargs: pytest.fail("network used"))
    project = tmp_path / "demo"
    project.mkdir()
    assert SelectedMediaDownloader().run(project, plan, sources, selection)["assets"] == []


def test_symlinked_media_directory_is_rejected(monkeypatch, tmp_path):
    plan, sources, selection = fixture_documents()
    project = tmp_path / "demo"
    project.mkdir()
    external = tmp_path / "elsewhere"
    external.mkdir()
    (project / "media").symlink_to(external, target_is_directory=True)
    monkeypatch.setattr(source_downloader, "urlopen", lambda *_args, **_kwargs: pytest.fail("network used"))
    with pytest.raises(DownloadError, match="symbolic link"):
        SelectedMediaDownloader().run(project, plan, sources, selection)
    assert list(external.iterdir()) == []


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf"), 0.00000001])
def test_invalid_max_size(value):
    with pytest.raises(ValueError, match="max-download-mb"):
        max_download_bytes(value)


def test_cli_download_mode_is_explicit_and_offline_in_test(monkeypatch, tmp_path, capsys):
    config = tmp_path / "config" / "transcription.json"
    config.parent.mkdir(parents=True)
    write_json(config, {"schema_version": 1, "default_whisper_model": "local/test-model"})
    project = tmp_path / "projects" / "demo"
    project.mkdir(parents=True)
    plan, sources, selection = fixture_documents()
    for filename, data in (("director_plan.json", plan), ("sources.json", sources),
                           ("source_selection.json", selection)):
        write_json(project / filename, data)
    original = {path.name: path.read_bytes() for path in project.iterdir()}
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    monkeypatch.setattr(source_downloader, "urlopen", lambda *_args, **_kwargs: FakeResponse())
    monkeypatch.setattr(source_downloader, "probe", good_probe)
    args = cli.parser().parse_args(["--project", "demo", "--download-sources", "--max-download-mb", "5"])
    assert args.download_sources and args.max_download_mb == 5
    assert cli.main(["--project", "demo", "--download-sources", "--max-download-mb", "5"]) == 0
    assert all((project / name).read_bytes() == data for name, data in original.items())
    assert read_json(project / "download_manifest.json")["assets"][0]["shot_ids"] == [2, 7]
    out = capsys.readouterr().out
    assert "[1/5] Loading source candidates" in out and "[5/5] Saving download_manifest.json" in out
    assert out.count("-> downloading") == 1
    with pytest.raises(SystemExit):
        cli.main(["--project", "demo", "--download-sources", "--max-download-mb", "0"])


@pytest.mark.parametrize("missing", ["sources.json", "source_selection.json"])
def test_cli_missing_input_fails_without_network(monkeypatch, tmp_path, missing, capsys):
    config = tmp_path / "config" / "transcription.json"
    config.parent.mkdir(parents=True)
    write_json(config, {"schema_version": 1, "default_whisper_model": "local/test-model"})
    project = tmp_path / "projects" / "demo"
    project.mkdir(parents=True)
    plan, sources, selection = fixture_documents()
    write_json(project / "director_plan.json", plan)
    if missing != "sources.json":
        write_json(project / "sources.json", sources)
    if missing != "source_selection.json":
        write_json(project / "source_selection.json", selection)
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    monkeypatch.setattr(source_downloader, "urlopen", lambda *_args, **_kwargs: pytest.fail("network used"))
    assert cli.main(["--project", "demo", "--download-sources"]) == 1
    assert missing in capsys.readouterr().err
    assert not (project / "download_manifest.json").exists()
