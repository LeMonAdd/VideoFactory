"""V3A source discovery tests use only mocked HTTP and synthetic project state."""

from __future__ import annotations

import io
import json
import ssl
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urlsplit

import pytest
import certifi

from videofactory import cli
from videofactory.models import read_json, write_json
from videofactory.source_finder import find_sources
from videofactory.source_models import validate_sources_document
from videofactory.source_provider import (LocalSourceProvider, PexelsSourceProvider,
                                          SourceSearchError)


VIDEO = {"id": 123, "width": 1920, "height": 1080, "duration": 12,
         "url": "https://www.pexels.com/video/example-123/", "image": "https://images.pexels.com/preview.jpg",
         "user": {"name": "Creator", "url": "https://www.pexels.com/@creator/"},
         "video_files": [{"link": "https://videos.pexels.com/file.mp4", "width": 1920,
                          "height": 1080, "quality": "hd", "file_type": "video/mp4"}]}


class Response:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return None

    def read(self, *_):
        return self.payload


def plan():
    return {"shots": [
        {"id": 1, "visual_type": "A_ROLL", "visual_query": None},
        {"id": 2, "visual_type": "B_ROLL", "visual_query": " Tokyo city "},
        {"id": 3, "visual_type": "GRAPHIC", "visual_query": None},
        {"id": 4, "visual_type": "B_ROLL", "visual_query": "tokyo  city"},
        {"id": 5, "visual_type": "IMAGE", "visual_query": "Tokyo image"},
    ]}


def test_pexels_request_and_normalization(monkeypatch):
    seen = []
    contexts = []
    create_context = ssl.create_default_context
    def tracked_context(*, cafile):
        contexts.append(cafile)
        return create_context(cafile=cafile)
    monkeypatch.setattr("videofactory.source_provider.ssl.create_default_context", tracked_context)
    def fake_open(request, timeout, context):
        seen.append((request, timeout, context))
        return Response(json.dumps({"videos": [VIDEO]}).encode())
    monkeypatch.setattr("videofactory.source_provider.urlopen", fake_open)
    provider = PexelsSourceProvider("private-token")
    result = provider.search("Tokyo commuters", "VIDEO", "landscape", 5)
    request, timeout, context = seen[0]
    assert request.get_header("Authorization") == "private-token" and timeout == 10
    assert request.get_header("User-agent") == "VideoFactory/1.0"
    assert context is provider._ssl_context and isinstance(context, ssl.SSLContext)
    assert context.verify_mode == ssl.CERT_REQUIRED and context.check_hostname is True
    assert contexts == [certifi.where()]
    assert parse_qs(urlsplit(request.full_url).query) == {
        "query": ["Tokyo commuters"], "orientation": ["landscape"], "per_page": ["5"]}
    candidate = result[0]
    assert candidate.candidate_id == "pexels:video:123"
    assert candidate.creator == "Creator" and candidate.creator_url.endswith("@creator/")
    assert candidate.page_url == VIDEO["url"] and candidate.license == "Pexels License"
    assert candidate.license_url == "https://www.pexels.com/license/"
    assert candidate.files[0].url == VIDEO["video_files"][0]["link"]
    assert candidate.orientation == "landscape" and candidate.duration == 12


def test_pexels_missing_optional_fields_and_empty_results(monkeypatch):
    monkeypatch.setattr("videofactory.source_provider.urlopen", lambda *_args, **_kwargs: Response(b'{"videos":[{"id":17},{}]}'))
    item = PexelsSourceProvider("token").search("query", "VIDEO", "landscape", 5)[0]
    assert item.page_url is None and item.creator is None and item.files == []
    assert item.width is None and item.duration is None
    monkeypatch.setattr("videofactory.source_provider.urlopen", lambda *_args, **_kwargs: Response(b'{"videos":[]}'))
    assert PexelsSourceProvider("token").search("query", "VIDEO", "landscape", 5) == []


def test_pexels_requires_key_without_http(monkeypatch):
    monkeypatch.delenv("PEXELS_API_KEY", raising=False)
    monkeypatch.setattr("videofactory.source_provider.urlopen", lambda *_args, **_kwargs: pytest.fail("HTTP called"))
    with pytest.raises(SourceSearchError, match="PEXELS_API_KEY"):
        PexelsSourceProvider()


@pytest.mark.parametrize("code,expected", [(401, "authorization"), (403, "authorization"), (429, "rate limit")])
def test_http_errors_sanitize_secret(monkeypatch, code, expected):
    def fail(*_args, **_kwargs):
        raise HTTPError("https://secret-token.example/", code, "secret-token", {}, io.BytesIO(b"secret-token"))
    monkeypatch.setattr("videofactory.source_provider.urlopen", fail)
    with pytest.raises(SourceSearchError) as caught:
        PexelsSourceProvider("secret-token").search("query", "VIDEO", "landscape", 5)
    assert expected in str(caught.value).lower() and "secret-token" not in str(caught.value)


@pytest.mark.parametrize("failure", [URLError("secret-token"), TimeoutError("secret-token")])
def test_network_errors_sanitize_secret(monkeypatch, failure):
    monkeypatch.setattr("videofactory.source_provider.urlopen", lambda *_args, **_kwargs: (_ for _ in ()).throw(failure))
    with pytest.raises(SourceSearchError) as caught:
        PexelsSourceProvider("secret-token").search("query", "VIDEO", "landscape", 5)
    assert "secret-token" not in str(caught.value)


@pytest.mark.parametrize("raw", [b"not json", b"{}", b'{"videos":{}}'])
def test_malformed_response(monkeypatch, raw):
    monkeypatch.setattr("videofactory.source_provider.urlopen", lambda *_args, **_kwargs: Response(raw))
    with pytest.raises(SourceSearchError):
        PexelsSourceProvider("token").search("query", "VIDEO", "landscape", 5)


def test_finder_deduplicates_and_skips_non_video(monkeypatch):
    calls = []
    def fake_open(request, timeout, context):
        calls.append(request)
        return Response(json.dumps({"videos": [VIDEO]}).encode())
    monkeypatch.setattr("videofactory.source_provider.urlopen", fake_open)
    doc = find_sources(plan(), "demo", PexelsSourceProvider("token"))
    assert len(calls) == 1
    assert [row["status"] for row in doc["requests"]] == ["SKIPPED", "FOUND", "SKIPPED", "FOUND", "SKIPPED"]
    assert doc["requests"][3]["reused_from_shot_id"] == 2
    assert doc["requests"][3]["candidates"][0]["query"] == "tokyo  city"
    assert doc["requests"][1]["candidates"][0]["provider_asset_id"] == "123"
    assert doc["sources"] == []


def test_finder_no_results_and_validation(monkeypatch):
    monkeypatch.setattr("videofactory.source_provider.urlopen", lambda *_args, **_kwargs: Response(b'{"videos":[]}'))
    doc = find_sources(plan(), "demo", PexelsSourceProvider("token"))
    assert doc["requests"][1]["status"] == "NO_RESULTS"
    doc["requests"][1]["status"] = "FOUND"
    with pytest.raises(ValueError, match="no candidates"):
        validate_sources_document(doc, "demo")


def test_local_provider_only_explicit_query(monkeypatch, tmp_path):
    from videofactory import source_provider
    monkeypatch.setattr(source_provider, "discover_assets", lambda *_args, **_kwargs: {"sources": [
        {"id": "broll_001", "kind": "B_ROLL", "path": "/tmp/tokyo.mp4", "width": 1920,
         "height": 1080, "duration": 3, "visual_queries": ["Tokyo city"]},
        {"id": "broll_002", "kind": "B_ROLL", "path": "/tmp/other.mp4", "width": 1920,
         "height": 1080, "duration": 3, "visual_queries": []},
    ]})
    provider = LocalSourceProvider(tmp_path)
    assert len(provider.search("tokyo CITY", "VIDEO", "landscape", 5)) == 1
    assert provider.search("other", "VIDEO", "landscape", 5) == []


def test_cli_find_sources_only_writes_sources(tmp_path, monkeypatch, capsys):
    config = tmp_path / "config" / "transcription.json"
    config.parent.mkdir(parents=True)
    write_json(config, {"schema_version": 1, "default_whisper_model": "local/test-model"})
    project = tmp_path / "projects" / "demo"
    project.mkdir(parents=True)
    write_json(project / "project.json", {"schema_version": 1, "project_name": "demo", "mode": "TALKING_HEAD"})
    write_json(project / "transcript.json", {"schema_version": 1, "duration": 6, "segments": []})
    write_json(project / "director_plan.json", {"schema_version": 1, "project_name": "demo", "shots": [
        {"id": 1, "start": 0, "end": 2, "visual_type": "A_ROLL", "visual_query": None, "reason": "Intro"},
        {"id": 2, "start": 2, "end": 4, "visual_type": "B_ROLL", "visual_query": "Tokyo city", "reason": "City"},
    ]})
    write_json(project / "sources.json", {"schema_version": 1, "sources": [{"id": "source_aroll"}]})
    original = {path.name: path.read_bytes() for path in project.iterdir()}
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    monkeypatch.setattr("videofactory.source_provider.urlopen", lambda *_args, **_kwargs: Response(b'{"videos":[]}'))
    monkeypatch.setenv("PEXELS_API_KEY", "private-token")
    assert cli.main(["--project", "demo", "--find-sources", "--source-provider", "pexels", "--source-limit", "5"]) == 0
    assert all((project / name).read_bytes() == data for name, data in original.items() if name != "sources.json")
    saved = read_json(project / "sources.json")
    assert saved["sources"] == [{"id": "source_aroll"}]
    assert saved["requests"][1]["status"] == "NO_RESULTS"
    assert "[1/4] Loading director plan" in capsys.readouterr().out
    assert "private-token" not in json.dumps(saved)


def test_cli_missing_plan_and_parser_options(tmp_path, monkeypatch, capsys):
    config = tmp_path / "config" / "transcription.json"
    config.parent.mkdir(parents=True)
    write_json(config, {"schema_version": 1, "default_whisper_model": "local/test-model"})
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    args = cli.parser().parse_args(["--project", "demo", "--find-sources", "--source-provider", "local", "--source-limit", "3"])
    assert args.find_sources and args.source_provider == "local" and args.source_limit == 3
    assert cli.main(["--project", "demo", "--find-sources", "--source-provider", "local"]) == 1
    assert "project.json" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        cli.main(["--project", "demo", "--find-sources"])
