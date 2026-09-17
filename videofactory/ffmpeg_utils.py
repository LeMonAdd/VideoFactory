"""Safe FFmpeg process execution and shared executable paths."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Sequence

FFMPEG = Path("/opt/homebrew/bin/ffmpeg")
FFPROBE = Path("/opt/homebrew/bin/ffprobe")


def run_command(command: Sequence[str | Path]) -> subprocess.CompletedProcess[str]:
    arguments = [str(part) for part in command]
    try:
        return subprocess.run(arguments, check=True, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise RuntimeError(f"Executable not found: {arguments[0]}") from exc
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "").strip()[-3000:]
        raise RuntimeError(f"{Path(arguments[0]).name} failed (exit {exc.returncode}): {detail}") from exc
