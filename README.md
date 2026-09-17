# VideoFactory

VideoFactory V1 is a local automated video-editing pipeline for macOS Apple Silicon. It turns narration and local visuals into a 1920×1080, 30 fps H.264/AAC draft. It uses Python 3.12, FFmpeg/ffprobe, and local MLX Whisper for real transcription. Fixture transcripts let the rest of the pipeline run without Metal.

## Architecture

`factory.py` orchestrates independent stages: input validation → ffprobe → narration extraction → transcription → transcript normalization → deterministic scene planning → local asset selection → timeline creation → FFmpeg rendering → ffprobe validation. The modules live in `videofactory/`. The persisted `timeline.json` is the authority for an edit; `scripts/render_timeline.py` renders it again without transcription or scene planning.

**TALKING_HEAD** reads a video with narration. Its original audio runs continuously while the picture alternates between A-roll and local B-roll when available. With no supporting assets, the video remains A-roll. **VOICEOVER** reads narration audio and builds visuals from local B-roll, still images, or a plain graphic placeholder. V1 scene decisions are deterministic, not AI generated.

## Directories

| Path | Use |
| --- | --- |
| `inbox/` | Your input video or audio (ignored by Git) |
| `assets/broll/`, `assets/images/` | Reusable local visuals |
| `assets/music/` | Reserved for a future version |
| `projects/<name>/` | Persistent JSON editing state |
| `temp/<name>/` | Extracted audio and other intermediates |
| `output/<name>/draft.mp4` | Rendered draft |
| `tests/`, `scripts/` | Tests and synthetic integration utility |

## Setup and commands

Use the repository virtual environment explicitly. Install the project's Python requirements into it only if missing; the V1 pipeline itself uses the standard library, while real transcription requires the existing `mlx_whisper` installation and tests require `pytest`. FFmpeg and ffprobe must be available at `/opt/homebrew/bin/ffmpeg` and `/opt/homebrew/bin/ffprobe`. A real MLX run needs host Metal access. The default real transcription model is `mlx-community/whisper-large-v3-turbo`, configured in `config/transcription.json`. MLX Whisper uses a cached model when present; otherwise Hugging Face may download it on the first run. VideoFactory inherits Hugging Face environment settings from the host shell without changing them.

First real talking-head test:

```sh
./.venv/bin/python factory.py --video inbox/talking_head.mp4 --project demo
```

Standalone voiceover:

```sh
./.venv/bin/python factory.py --voice inbox/voice.wav --project demo_voice
```

Put optional supporting videos in `assets/broll/` and stills in `assets/images/`. Use `--whisper-model /path/to/local/model` or another Hugging Face model ID to override the configured default; the older `--mlx-model` spelling remains an alias. Add `--language ru` for Russian narration, or omit it for MLX Whisper language detection. The selected backend, model, and language setting are recorded in `project.json`. Replacing an existing generated draft requires `--overwrite-render`; source media is never modified. `--encoder h264_videotoolbox` opts into hardware encoding; software `libx264` is the default.

To bypass MLX for a deterministic test:

```sh
./.venv/bin/python factory.py --video inbox/talking_head.mp4 --project demo_fixture --fixture-transcript tests/fixtures/synthetic_transcript.json
```

Fixture segment times must fit the source duration. An optional `--assets-dir PATH` points to another local directory containing `broll/` and `images/`.

## Tests and synthetic integration

```sh
./.venv/bin/python -m pytest
./.venv/bin/python scripts/synthetic_integration.py
/opt/homebrew/bin/ffprobe -v error -show_format -show_streams -of json output/synthetic_v1/draft.mp4
```

The integration utility generates a red talking-head placeholder with continuous tone audio, plus blue and green B-roll clips. Its five timed fixture segments produce A-roll → B-roll → A-roll → B-roll → A-roll. Generated source media stays under `temp/`; the draft stays under `output/`. Neither is tracked by Git. The script overwrites only its generated draft when rerun.

Replay a saved edit with `./.venv/bin/python scripts/render_timeline.py --project synthetic_v1`. This writes `output/synthetic_v1/replay.mp4` and refuses to overwrite an existing replay unless `--overwrite-render` is supplied.

## Project state

Every `projects/<name>/` directory contains `project.json` (mode, input, settings), `transcript.json` (normalized timed segments and optional words), `scenes.json` (visual decisions), `sources.json` (source metadata and future provenance fields), and `timeline.json` (exact visual events and continuous narration source). Each document has `schema_version`. Paths in project state are absolute local paths, so reproducing an edit requires the referenced media to remain available.

V1 does not download media, search the web, select music, style subtitles, publish videos, or use external AI APIs. An AI Director and richer graphics are future work.
