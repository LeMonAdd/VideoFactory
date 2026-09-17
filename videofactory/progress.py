"""Immediate, human-readable pipeline stage reporting."""

from __future__ import annotations

import sys
import time
from typing import TextIO


class ProgressReporter:
    def __init__(self, total: int, stream: TextIO | None = None) -> None:
        self.total = total
        self.stream = stream or sys.stdout
        self.started: dict[int, float] = {}

    def start(self, number: int, label: str) -> None:
        self.started[number] = time.monotonic()
        print(f"[{number}/{self.total}] {label}...", file=self.stream, flush=True)

    def complete(self, number: int, label: str) -> None:
        elapsed = time.monotonic() - self.started[number]
        print(f"[{number}/{self.total}] {label} complete ({elapsed:.1f}s)",
              file=self.stream, flush=True)
