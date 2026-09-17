"""Download selected Pexels MP4s once, validate them, and persist provenance."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import ssl
from pathlib import Path
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit, urlunsplit
from urllib.request import Request, urlopen

import certifi
from jsonschema import Draft202012Validator

from . import SCHEMA_VERSION
from .media_probe import duration as probe_duration
from .media_probe import probe, stream
from .models import read_json, write_json
from .paths import ROOT
from .source_selection import validate_selection

MANIFEST_SCHEMA = ROOT / "config" / "download_manifest.schema.json"
DEFAULT_MAX_DOWNLOAD_MB = 500.0
CHUNK_SIZE = 64 * 1024
_ASSET_ID = re.compile(r"pexels:video:([1-9][0-9]*)\Z")
_MP4_TYPES = {"video/mp4", "application/mp4"}
_BINARY_TYPES = {"application/octet-stream", "binary/octet-stream"}


class DownloadError(RuntimeError):
    """Safe diagnostic for a selected media asset that cannot be downloaded."""


def max_download_bytes(megabytes: float) -> int:
    if not isinstance(megabytes, (int, float)) or isinstance(megabytes, bool) or not math.isfinite(megabytes):
        raise ValueError("--max-download-mb must be a positive finite number")
    scaled = megabytes * 1024 * 1024
    if not math.isfinite(scaled):
        raise ValueError("--max-download-mb is too large")
    size = int(scaled)
    if size < 1:
        raise ValueError("--max-download-mb must allow at least one byte")
    return size


def checked_https_url(url: str) -> str:
    if not isinstance(url, str) or not url or any(ord(char) < 33 for char in url):
        raise DownloadError("Media URL is malformed")
    try:
        parsed = urlsplit(url)
        if (parsed.scheme.lower() != "https" or not parsed.hostname or parsed.username is not None
                or parsed.password is not None or parsed.port == 0):
            raise DownloadError("Media URL must be HTTPS without embedded credentials")
        # Accessing port also rejects invalid port syntax.
        _ = parsed.port
    except ValueError:
        raise DownloadError("Media URL is malformed") from None
    return url


def public_url(url: str) -> str:
    """Persist URL provenance without query parameters or fragments."""
    parsed = urlsplit(url)
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "", ""))


def choose_video_variant(files: list[dict[str, Any]]) -> dict[str, Any]:
    """Prefer MP4, landscape, and a 1080p-class source over needless 4K."""
    usable = []
    target_area = 1920 * 1080
    for item in files:
        url = item.get("url")
        if not isinstance(url, str):
            continue
        mime = (item.get("file_type") or "").lower()
        if mime not in _MP4_TYPES and not (not mime and urlsplit(url).path.lower().endswith(".mp4")):
            continue
        width, height = item.get("width"), item.get("height")
        landscape = isinstance(width, int) and isinstance(height, int) and width >= height
        fhd = landscape and width >= 1920 and height >= 1080
        area = width * height if isinstance(width, int) and isinstance(height, int) else 0
        # Among FHD-or-better files, nearest to 1080p wins. Otherwise favor the largest usable file.
        rank = (landscape, fhd, -abs(area - target_area) if fhd else area, url)
        usable.append((rank, item))
    if not usable:
        raise DownloadError("Selected candidate has no usable MP4 file variant")
    return max(usable, key=lambda value: value[0])[1]


def selected_assets(plan: dict[str, Any], sources: dict[str, Any],
                    selection: dict[str, Any]) -> list[tuple[dict[str, Any], list[int]]]:
    validate_selection(selection, plan, sources)
    requests = {request["shot_id"]: request for request in sources["requests"]}
    grouped: dict[str, tuple[dict[str, Any], list[int]]] = {}
    for decision in selection["selections"]:
        if decision["status"] != "SELECTED":
            continue
        shot_id, candidate_id = decision["shot_id"], decision["candidate_id"]
        candidate = next(item for item in requests[shot_id]["candidates"]
                         if item["candidate_id"] == candidate_id)
        match = _ASSET_ID.fullmatch(candidate_id)
        if not match or candidate["provider"] != "pexels" or candidate["media_type"] != "VIDEO" or candidate["provider_asset_id"] != match.group(1):
            raise DownloadError(f"Selected asset {candidate_id} is not a supported Pexels video")
        if candidate_id in grouped:
            previous, shot_ids = grouped[candidate_id]
            if previous["provider_asset_id"] != candidate["provider_asset_id"] or previous["files"] != candidate["files"]:
                raise DownloadError(f"Conflicting source records for {candidate_id}")
            shot_ids.append(shot_id)
        else:
            grouped[candidate_id] = (candidate, [shot_id])
    return list(grouped.values())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def validate_video_file(path: Path) -> tuple[int, int, float]:
    if not path.is_file() or path.stat().st_size <= 0:
        raise DownloadError("Downloaded media file is empty or missing")
    try:
        info = probe(path)
        video = stream(info, "video")
        if video is None:
            raise ValueError("no video stream")
        width, height = int(video.get("width") or 0), int(video.get("height") or 0)
        length = probe_duration(info)
        if width <= 0 or height <= 0 or not math.isfinite(length) or length <= 0:
            raise ValueError("invalid video dimensions or duration")
        return width, height, round(length, 3)
    except (OSError, RuntimeError, ValueError, TypeError) as exc:
        raise DownloadError(f"ffprobe validation failed: {exc}") from None


def validate_manifest(document: dict[str, Any], project_name: str) -> dict[str, Any]:
    schema = json.loads(MANIFEST_SCHEMA.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    errors = list(Draft202012Validator(schema).iter_errors(document))
    if errors:
        first = errors[0]
        location = ".".join(map(str, first.path)) or "root"
        raise ValueError(f"Invalid download_manifest.json at {location}: {first.message}")
    if document["project_name"] != project_name:
        raise ValueError("Download manifest project_name does not match")
    seen: set[str] = set()
    for asset in document["assets"]:
        candidate_id = asset["candidate_id"]
        match = _ASSET_ID.fullmatch(candidate_id)
        if candidate_id in seen or not match or asset["provider_asset_id"] != match.group(1):
            raise ValueError(f"Invalid or duplicate manifest asset {candidate_id}")
        seen.add(candidate_id)
        if asset["relative_path"] != f"media/broll/pexels_video_{match.group(1)}.mp4":
            raise ValueError(f"Manifest path mismatch for {candidate_id}")
        if not math.isfinite(asset["duration"]):
            raise ValueError(f"Invalid manifest duration for {candidate_id}")
    return document


class SelectedMediaDownloader:
    def __init__(self, max_download_mb: float = DEFAULT_MAX_DOWNLOAD_MB, timeout_seconds: float = 30) -> None:
        self.max_bytes = max_download_bytes(max_download_mb)
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("Download timeout must be positive")
        self.timeout_seconds = timeout_seconds
        self._ssl_context = ssl.create_default_context(cafile=certifi.where())

    def _download(self, url: str, part: Path, candidate_id: str) -> tuple[int, str, str]:
        checked_https_url(url)
        if part.exists():
            raise DownloadError(f"Stale partial file exists for {candidate_id}: {part}")
        request = Request(url, headers={"User-Agent": "VideoFactory/1.0", "Accept": "video/mp4,application/octet-stream"})
        created = False
        completed = False
        try:
            with urlopen(request, timeout=self.timeout_seconds, context=self._ssl_context) as response:
                checked_https_url(response.geturl())
                status = response.status
                if status != 200:
                    raise DownloadError(f"Media server returned HTTP {status}")
                mime = (response.headers.get("Content-Type") or "").split(";", 1)[0].strip().lower()
                if mime not in _MP4_TYPES | _BINARY_TYPES:
                    raise DownloadError(f"Media server returned unsupported content type: {mime or 'missing'}")
                declared = response.headers.get("Content-Length")
                if declared is not None:
                    try:
                        declared_bytes = int(declared)
                    except ValueError:
                        raise DownloadError("Media server returned invalid Content-Length") from None
                    if declared_bytes < 0 or declared_bytes > self.max_bytes:
                        raise DownloadError("Media Content-Length exceeds the configured limit")
                digest = hashlib.sha256()
                size = 0
                with part.open("xb") as target:
                    created = True
                    while chunk := response.read(CHUNK_SIZE):
                        size += len(chunk)
                        if size > self.max_bytes:
                            raise DownloadError("Media stream exceeds the configured size limit")
                        target.write(chunk)
                        digest.update(chunk)
                if size == 0:
                    raise DownloadError("Media response is empty")
                with part.open("rb") as source:
                    prefix = source.read(64)
                if prefix.lstrip().lower().startswith((b"<html", b"<!doctype html", b"{", b"[")):
                    raise DownloadError("Media response looks like an HTML/JSON error document")
                if mime in _BINARY_TYPES and prefix[4:8] != b"ftyp":
                    raise DownloadError("Generic binary response does not look like MP4")
                completed = True
                return size, digest.hexdigest(), mime
        except HTTPError as exc:
            raise DownloadError(f"Media server returned HTTP {exc.code}") from None
        except (URLError, TimeoutError, OSError) as exc:
            # Never include a URL, response body, or request headers in diagnostics.
            if isinstance(exc, FileExistsError):
                raise DownloadError(f"Stale partial file exists for {candidate_id}: {part}") from None
            raise DownloadError(f"Media download failed for {candidate_id}: network or filesystem error") from None
        finally:
            if created and not completed and part.exists():
                part.unlink()

    def run(self, project_dir: Path, plan: dict[str, Any], sources: dict[str, Any],
            selection: dict[str, Any], report: Callable[[str], None] | None = None,
            before_save: Callable[[], None] | None = None) -> dict[str, Any]:
        project_name = project_dir.name
        if project_name != plan["project_name"]:
            raise ValueError("Project directory name does not match director plan")
        targets = selected_assets(plan, sources, selection)
        manifest_path = project_dir / "download_manifest.json"
        broll_dir = project_dir / "media" / "broll"
        if (project_dir.is_symlink() or (project_dir / "media").is_symlink()
                or broll_dir.is_symlink() or manifest_path.is_symlink()):
            raise DownloadError("Project media or manifest path must not be a symbolic link")
        previous = read_json(manifest_path) if manifest_path.is_file() else None
        if previous is not None:
            validate_manifest(previous, project_name)
        prior_assets = {item["candidate_id"]: item for item in previous["assets"]} if previous else {}
        staged: list[Path] = []
        created: list[Path] = []
        assets: list[dict[str, Any]] = []
        completed = False
        try:
            for candidate, shot_ids in targets:
                candidate_id = candidate["candidate_id"]
                asset_id = candidate["provider_asset_id"]
                relative_path = f"media/broll/pexels_video_{asset_id}.mp4"
                final = project_dir / relative_path
                prior = prior_assets.get(candidate_id)
                if final.exists():
                    if prior is None or prior["relative_path"] != relative_path:
                        raise DownloadError(f"Existing file for {candidate_id} has no matching manifest entry")
                    if not final.is_file() or sha256_file(final) != prior["sha256"]:
                        raise DownloadError(f"Existing file hash differs from manifest for {candidate_id}")
                    if report:
                        report(f"{candidate_id} -> reused")
                    assets.append(prior | {"status": "REUSED", "shot_ids": sorted(shot_ids)})
                    continue
                if prior is not None:
                    raise DownloadError(f"Manifest file is missing for {candidate_id}")
                try:
                    variant = choose_video_variant(candidate["files"])
                    url = checked_https_url(variant["url"])
                except DownloadError as exc:
                    raise DownloadError(f"{candidate_id}: {exc}") from None
                broll_dir.mkdir(parents=True, exist_ok=True)
                part = final.with_suffix(final.suffix + ".part")
                if report:
                    report(f"{candidate_id} -> downloading")
                if part.exists():
                    raise DownloadError(f"Stale partial file exists for {candidate_id}: {part}")
                try:
                    size, sha256, mime = self._download(url, part, candidate_id)
                    staged.append(part)
                    width, height, length = validate_video_file(part)
                except DownloadError as exc:
                    raise DownloadError(f"{candidate_id}: {exc}") from None
                assets.append({
                    "candidate_id": candidate_id, "provider": "pexels", "provider_asset_id": asset_id,
                    "media_type": "VIDEO", "status": "DOWNLOADED", "shot_ids": sorted(shot_ids),
                    "relative_path": relative_path,
                    "source_page_url": public_url(candidate["page_url"]) if candidate["page_url"] else None,
                    "download_url": public_url(url), "creator": candidate["creator"],
                    "creator_url": candidate["creator_url"], "license_name": candidate["license"],
                    "license_url": candidate["license_url"],
                    "candidate_width": candidate["width"], "candidate_height": candidate["height"],
                    "candidate_duration": candidate["duration"],
                    "chosen_file": {key: variant[key] for key in ("width", "height", "quality", "file_type")},
                    "width": width, "height": height, "duration": length,
                    "file_size_bytes": size, "sha256": sha256, "content_type": mime,
                })
            document = validate_manifest({"schema_version": SCHEMA_VERSION,
                                          "project_name": project_name, "assets": assets}, project_name)
            for part in staged:
                final = part.with_suffix("")
                if final.exists():
                    raise DownloadError(f"Final media path unexpectedly exists: {final}")
                # Atomic link avoids replacing an unrelated file created concurrently.
                os.link(part, final)
                created.append(final)
                part.unlink()
            if before_save:
                before_save()
            write_json(manifest_path, document)
            completed = True
            return document
        finally:
            if not completed:
                for path in staged + created:
                    if path.exists():
                        path.unlink()
