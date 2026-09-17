# VideoFactory

VideoFactory is a local automated video-editing pipeline for macOS Apple Silicon. It turns narration and local visuals into a 1920×1080, 30 fps H.264/AAC draft. Python 3.12, FFmpeg/ffprobe, and local MLX Whisper handle media and transcription. V2A adds an editorial DirectorPlan; V3A searches external media candidates; V3B1 selects or rejects them without downloading or rendering. Fixture transcripts keep automated tests independent of Metal.

## Architecture

`factory.py` orchestrates independent stages: input validation → ffprobe → narration extraction → transcription → transcript normalization → DirectorPlan → local asset resolution → scenes → timeline → FFmpeg rendering → ffprobe validation. The modules live in `videofactory/`. `director_plan.json` records what should appear; `timeline.json` records the actual sources and timing used to render. The separate V3A Source Finder writes candidate requests to `sources.json`; V3B1 writes decisions to `source_selection.json`. Neither changes the timeline. `scripts/render_timeline.py` can replay a saved edit.

**TALKING_HEAD** reads a video with narration. Its original audio runs continuously while the picture may switch between presenter footage and explicitly matched local visuals. Uncovered intervals and unresolved visual requests remain A-roll. **VOICEOVER** reads narration audio and uses local B-roll, still images, or a plain graphic placeholder. Unresolved VOICEOVER requests become graphic placeholders.

The default `--director rule` uses deterministic offline rules. `--director codex` is opt-in and uses the installed Codex CLI and the user's existing authentication. It sends only project mode, language, duration, transcript timestamps and words, editorial rules, and the expected response shape. It does not send source video or repository code. The Codex process runs once per short project with `--ephemeral --sandbox read-only` in an isolated temporary directory. Codex receives the structure-only `config/codex_director_output.schema.json`; VideoFactory separately validates timing and editorial rules against `config/director_plan.schema.json`, snaps nearby cuts to word boundaries, and adds provider metadata before saving.

## Directories

| Path | Use |
| --- | --- |
| `inbox/` | Your input video or audio (ignored by Git) |
| `assets/broll/`, `assets/images/` | Reusable local visuals |
| `assets/music/` | Reserved for a future version |
| `projects/<name>/` | Persistent JSON editing state, including DirectorPlan |
| `temp/<name>/` | Extracted audio and other intermediates |
| `output/<name>/draft.mp4` | Rendered draft |
| `tests/`, `scripts/` | Tests and synthetic integration utility |

## Setup and commands

Use the repository virtual environment explicitly. Install Python dependencies with `./.venv/bin/python -m pip install -r requirements-dev.txt` if missing. V2A uses `jsonschema` for strict plan validation; real transcription also requires the existing `mlx_whisper` installation. FFmpeg and ffprobe must be available at `/opt/homebrew/bin/ffmpeg` and `/opt/homebrew/bin/ffprobe`. A real MLX run needs host Metal access. The default real transcription model is `mlx-community/whisper-large-v3-turbo`, configured in `config/transcription.json`. MLX Whisper uses a cached model when present; otherwise Hugging Face may download it on the first run. VideoFactory inherits Hugging Face environment settings from the host shell without changing them.

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

### Director and plan-only commands

```sh
./.venv/bin/python factory.py --video inbox/talking_head.mp4 --project demo --director rule
./.venv/bin/python factory.py --project demo --director rule --plan-only
./.venv/bin/python factory.py --project real_test_large_001 --director codex --plan-only
```

Normal new projects still require `--video` or `--voice`. `--plan-only` requires an existing `project.json` and `transcript.json`; it updates only `director_plan.json` and does not transcribe or render. Codex use requires the explicit `--director codex` flag. Optional `--director-model MODEL` overrides the user's Codex default, and `--director-timeout SECONDS` changes the 300-second subprocess timeout. Stage progress prints and flushes immediately, with elapsed time on completion.

B-roll and image requests are editorial intent until a local asset is explicitly tagged with the same query. For example, place `tokyo.mp4` in `assets/broll/` and add `tokyo.mp4.json` beside it:

```json
{"schema_version": 1, "visual_queries": ["modern Tokyo skyline and busy city streets"]}
```

Without that exact local description, VideoFactory keeps the query as `unresolved_local_asset` in `scenes.json` and renders A-roll or a graphic placeholder. It does not guess a match from a filename or fetch media.

### V3A source candidate search

For an existing project with `director_plan.json`, set `PEXELS_API_KEY` in your shell environment and run:

```sh
./.venv/bin/python factory.py --project real_test_large_001 --find-sources --source-provider pexels --source-limit 5
```

The Pexels provider searches B-roll video queries through the [Pexels video search API](https://www.pexels.com/api/documentation/), requests landscape results, and reuses results for repeated queries. `--source-limit` accepts 1–20; the default is 5. A-roll and graphics are skipped; Pexels image search is not implemented. `--source-provider local` searches only assets whose sidecar `visual_queries` explicitly match. Source search updates `sources.json` only; it does not transcribe, invoke Codex, download, select, or render media. External API calls occur only when this Pexels command is explicitly requested.

Each request records `FOUND`, `NO_RESULTS`, `SKIPPED`, or `ERROR` and an unselected candidate list. Candidate metadata includes the original page, creator and profile when supplied, file variants, and the [Pexels License](https://www.pexels.com/license/) link. Keep those links and creator details with any media selected in a future version; check the license and attribution requirements before publication. `PEXELS_API_KEY` is read only from the environment and is never saved in project JSON.

### V3B1 source candidate selection

Run selection against saved `director_plan.json` and `sources.json`:

```sh
./.venv/bin/python factory.py --project real_test_large_001 --select-sources --selector rule
./.venv/bin/python factory.py --project real_test_large_001 --select-sources --selector codex
```

The offline rule selector is the default. It accepts explicitly tagged local assets or Pexels page-title clues that contain every meaningful query term, provided a video is long enough for the shot. It is deliberately conservative and can reject plausible but unproven results. The opt-in Codex selector makes one read-only `codex exec` call with compact shot and candidate metadata; it sends no media, API key, repository files, or download links. `--selector-model MODEL` and `--selector-timeout SECONDS` are optional. VideoFactory validates every returned ID against that shot's saved candidates and checks duration before writing `source_selection.json`.

Selections are `SELECTED`, `NO_SUITABLE_CANDIDATE`, or `SKIPPED`. Rejection is valid when metadata does not establish a match, such as generic breakfast footage for narration specifically about Japan. Repeated queries are decided per shot; the rule selector prefers a different equally suitable clip. The file records a reason, optional confidence and refined query, plus VideoFactory-created provider metadata. Selection does not search, download, transcribe, invoke the Director, alter the timeline, or render. **Future V3B2** will download selected candidates; **future V3C** will choose source in/out points and build the final external-media render timeline.

## Tests and synthetic integration

```sh
./.venv/bin/python -m pytest
./.venv/bin/python scripts/synthetic_integration.py
/opt/homebrew/bin/ffprobe -v error -show_format -show_streams -of json output/synthetic_v1/draft.mp4
```

The integration utility generates a red talking-head placeholder with continuous tone audio, plus blue and green B-roll clips with explicit local query sidecars. Its five timed fixture segments produce A-roll → B-roll → A-roll → B-roll → A-roll through the offline rule provider. Generated source media stays under `temp/`; the draft stays under `output/`. Neither is tracked by Git. The script overwrites only its generated draft when rerun.

Replay a saved edit with `./.venv/bin/python scripts/render_timeline.py --project synthetic_v1`. This writes `output/synthetic_v1/replay.mp4` and refuses to overwrite an existing replay unless `--overwrite-render` is supplied.

## Project state

Every new `projects/<name>/` directory contains `project.json` (mode, input, settings), `transcript.json` (normalized timed segments and optional words), `director_plan.json` (editorial shots, reasons, provider/model, invocation count and elapsed time), `scenes.json` (resolved visuals or preserved unresolved queries), `sources.json` (local source metadata plus optional V3A search requests), and `timeline.json` (exact visual events and continuous narration source). Running V3B1 adds `source_selection.json`; it is not needed for the local V1 render. Each document has `schema_version`. Paths in project state are absolute local paths, so reproducing an edit requires the referenced media to remain available.

V3B1 chooses among saved candidates only. It does not download media, select music, style subtitles, publish videos, or generate images.
