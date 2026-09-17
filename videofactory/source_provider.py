"""Explicit local and Pexels candidate discovery; never downloads media."""

from __future__ import annotations

import json
import os
import socket
import ssl
from pathlib import Path
from typing import Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import certifi

from .asset_manager import discover_assets
from .source_models import (CandidateFile, SourceCandidate, nonnegative_number,
                            optional_text, orientation_for, positive_int)

PEXELS_ENDPOINT = "https://api.pexels.com/v1/videos/search"
PEXELS_LICENSE = "https://www.pexels.com/license/"


class SourceSearchError(RuntimeError):
    def __init__(self, message: str, *, fatal: bool = False) -> None:
        super().__init__(message)
        self.fatal = fatal


class SourceProvider(Protocol):
    name: str

    def search(self, query: str, media_type: str, orientation: str, limit: int) -> list[SourceCandidate]: ...


class LocalSourceProvider:
    name = "local"

    def __init__(self, root: Path, asset_root: Path | None = None) -> None:
        self.root = root
        self.asset_root = asset_root
        self._assets: list[dict] | None = None

    def search(self, query: str, media_type: str, orientation: str, limit: int) -> list[SourceCandidate]:
        if self._assets is None:
            self._assets = discover_assets(self.root, "VOICEOVER", asset_root=self.asset_root)["sources"]
        wanted = " ".join(query.split()).casefold()
        results: list[SourceCandidate] = []
        for asset in self._assets:
            if (media_type == "VIDEO" and asset["kind"] != "B_ROLL") or (media_type == "IMAGE" and asset["kind"] != "IMAGE"):
                continue
            if wanted not in {" ".join(value.split()).casefold() for value in asset.get("visual_queries", [])}:
                continue
            width, height = positive_int(asset.get("width")), positive_int(asset.get("height"))
            actual_orientation = orientation_for(width, height)
            if orientation and actual_orientation != orientation:
                continue
            results.append(SourceCandidate(
                candidate_id=f"local:{asset['id']}", provider="local", provider_asset_id=asset["id"],
                media_type=media_type, query=query, width=width, height=height,
                duration=nonnegative_number(asset.get("duration")), orientation=actual_orientation,
                local_path=asset["path"],
            ))
            if len(results) >= limit:
                break
        return results


def normalize_pexels_video(video: dict, query: str) -> SourceCandidate | None:
    asset_id = video.get("id")
    if not isinstance(asset_id, (int, str)) or isinstance(asset_id, bool) or not str(asset_id).strip():
        return None
    width, height = positive_int(video.get("width")), positive_int(video.get("height"))
    user = video.get("user") if isinstance(video.get("user"), dict) else {}
    raw_files = video.get("video_files") if isinstance(video.get("video_files"), list) else []
    files = []
    for item in raw_files:
        if not isinstance(item, dict):
            continue
        url = optional_text(item.get("link"))
        if url:
            files.append(CandidateFile(url, positive_int(item.get("width")),
                                       positive_int(item.get("height")), optional_text(item.get("quality")),
                                       optional_text(item.get("file_type"))))
    return SourceCandidate(
        candidate_id=f"pexels:video:{asset_id}", provider="pexels", provider_asset_id=str(asset_id),
        media_type="VIDEO", query=query, page_url=optional_text(video.get("url")),
        preview_url=optional_text(video.get("image")), creator=optional_text(user.get("name")),
        creator_url=optional_text(user.get("url")), license="Pexels License", license_url=PEXELS_LICENSE,
        width=width, height=height, duration=nonnegative_number(video.get("duration")),
        orientation=orientation_for(width, height), files=files,
    )


class PexelsSourceProvider:
    name = "pexels"

    def __init__(self, api_key: str | None = None, timeout_seconds: float = 10.0) -> None:
        self._api_key = api_key if api_key is not None else os.environ.get("PEXELS_API_KEY")
        if not self._api_key:
            raise SourceSearchError("PEXELS_API_KEY is required for Pexels source search", fatal=True)
        if timeout_seconds <= 0:
            raise ValueError("Pexels timeout must be positive")
        self.timeout_seconds = timeout_seconds
        self._ssl_context = ssl.create_default_context(cafile=certifi.where())

    def search(self, query: str, media_type: str, orientation: str, limit: int) -> list[SourceCandidate]:
        if media_type != "VIDEO":
            raise ValueError("Pexels V3A supports video candidate search only")
        if not 1 <= limit <= 20:
            raise ValueError("Source limit must be between 1 and 20")
        url = PEXELS_ENDPOINT + "?" + urlencode({"query": query, "orientation": orientation, "per_page": limit})
        request = Request(url, headers={
            "Authorization": self._api_key,
            "Accept": "application/json",
            "User-Agent": "VideoFactory/1.0",
        })
        try:
            with urlopen(request, timeout=self.timeout_seconds, context=self._ssl_context) as response:
                raw = response.read(5_000_001)
        except HTTPError as exc:
            if exc.code in (401, 403):
                raise SourceSearchError(f"Pexels authorization failed (HTTP {exc.code}); check PEXELS_API_KEY", fatal=True) from None
            if exc.code == 429:
                raise SourceSearchError("Pexels rate limit reached (HTTP 429)", fatal=True) from None
            raise SourceSearchError(f"Pexels search failed (HTTP {exc.code})") from None
        except URLError as exc:
            reason = str(exc.reason).replace(self._api_key, "[redacted]")
            raise SourceSearchError(f"Pexels TLS/network error: {reason}") from None
        except (TimeoutError, socket.timeout):
            raise SourceSearchError("Pexels search timed out") from None
        if len(raw) > 5_000_000:
            raise SourceSearchError("Pexels response is too large")
        try:
            payload = json.loads(raw)
        except (ValueError, UnicodeDecodeError):
            raise SourceSearchError("Pexels returned malformed JSON") from None
        if not isinstance(payload, dict) or not isinstance(payload.get("videos"), list):
            raise SourceSearchError("Pexels response has no videos list")
        results = [normalize_pexels_video(item, query) for item in payload["videos"] if isinstance(item, dict)]
        return [item for item in results if item is not None][:limit]
