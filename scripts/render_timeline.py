"""Replay a saved VideoFactory edit without transcription or scene planning."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from videofactory.models import read_json  # noqa: E402
from videofactory.paths import ProjectPaths  # noqa: E402
from videofactory.renderer import render  # noqa: E402
from videofactory.validator import validate_output  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Render an existing timeline.json")
    parser.add_argument("--project", required=True)
    parser.add_argument("--overwrite-render", action="store_true")
    args = parser.parse_args()
    try:
        paths = ProjectPaths(ROOT, args.project)
        timeline = read_json(paths.project / "timeline.json")
        destination = paths.output / "replay.mp4"
        render(timeline, destination, overwrite=args.overwrite_render)
        result = validate_output(destination, timeline["output_settings"])
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"Timeline replay error: {exc}", file=sys.stderr)
        return 1
    print(f"Replayed and validated: {result['path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
