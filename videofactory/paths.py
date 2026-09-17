"""Project paths and names."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NAME_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}\Z")


def safe_project_name(name: str) -> str:
    if not NAME_PATTERN.fullmatch(name):
        raise ValueError("Project name must be 1–64 letters, digits, '_' or '-', starting with a letter or digit")
    return name


@dataclass(frozen=True)
class ProjectPaths:
    root: Path
    name: str

    def __post_init__(self) -> None:
        safe_project_name(self.name)

    @property
    def project(self) -> Path:
        return self.root / "projects" / self.name

    @property
    def temp(self) -> Path:
        return self.root / "temp" / self.name

    @property
    def output(self) -> Path:
        return self.root / "output" / self.name

    @property
    def draft(self) -> Path:
        return self.output / "draft.mp4"

    def create(self) -> None:
        for path in (self.project, self.temp, self.output):
            path.mkdir(parents=True, exist_ok=True)
