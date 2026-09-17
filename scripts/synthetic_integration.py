"""Generate local fixture media, render a draft, and validate it.

Run from the repository root with ./.venv/bin/python scripts/synthetic_integration.py.
All generated source media stays in temp/ and the rendered draft stays in output/.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from videofactory.cli import main  # noqa: E402
from videofactory.ffmpeg_utils import FFMPEG, run_command  # noqa: E402
from videofactory.models import read_json, write_json  # noqa: E402
from videofactory.validator import validate_output  # noqa: E402


def generate() -> tuple[Path, Path]:
    base = ROOT / "temp" / "synthetic_v1"
    broll = base / "assets" / "broll"
    broll.mkdir(parents=True, exist_ok=True)
    talking_head = base / "talking_head.mp4"
    run_command([
        FFMPEG, "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
        "-f", "lavfi", "-i", "color=c=red:s=320x180:r=30:d=20",
        "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:duration=20",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
        "-shortest", talking_head,
    ])
    for name, color in (("blue", "blue"), ("green", "green")):
        run_command([
            FFMPEG, "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
            "-f", "lavfi", "-i", f"color=c={color}:s=320x180:r=30:d=5",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", broll / f"{name}.mp4",
        ])
    write_json(broll / "blue.mp4.json", {
        "schema_version": 1,
        "visual_queries": ["modern Tokyo skyline and busy city streets"],
    })
    write_json(broll / "green.mp4.json", {
        "schema_version": 1,
        "visual_queries": ["commuters at a busy train station"],
    })
    return talking_head, base / "assets"


if __name__ == "__main__":
    video, assets = generate()
    code = main([
        "--video", str(video), "--project", "synthetic_v1",
        "--assets-dir", str(assets),
        "--fixture-transcript", str(ROOT / "tests" / "fixtures" / "synthetic_transcript.json"),
        "--overwrite-render",
    ])
    if code == 0:
        scenes = read_json(ROOT / "projects" / "synthetic_v1" / "scenes.json")
        actual = [scene["visual_type"] for scene in scenes["scenes"]]
        expected = ["A_ROLL", "B_ROLL", "A_ROLL", "B_ROLL", "A_ROLL"]
        if actual != expected:
            raise RuntimeError(f"Unexpected synthetic visual sequence: {actual}")
        timeline = read_json(ROOT / "projects" / "synthetic_v1" / "timeline.json")
        result = validate_output(ROOT / "output" / "synthetic_v1" / "draft.mp4",
                                 timeline["output_settings"])
        print(f"Integration ffprobe validation: {result}")
    raise SystemExit(code)
