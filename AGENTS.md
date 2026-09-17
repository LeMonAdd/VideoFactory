# VideoFactory Project Guide

## Purpose and Scope

VideoFactory is a local automated video-editing pipeline for macOS Apple Silicon. Python 3.12 orchestrates the workflow; FFmpeg and ffprobe process and inspect media; MLX Whisper transcribes locally. V2A adds a transcript-only AI Director. V3A adds opt-in external candidate search; all rendered visual media remains local.

Do not add YouTube downloading, automatic internet B-roll search during normal editing, web scraping, a GUI, automatic music selection, automatic subtitle styling, external transcription or Director APIs, or cloud rendering unless explicitly requested. Do not require an OpenAI API key.

## Development Environment

The repository virtual environment is `/Users/roman/VideoFactory/.venv`. For every VideoFactory Python command, explicitly use `./.venv/bin/python`, `./.venv/bin/pip`, `./.venv/bin/pytest`, or `./.venv/bin/mlx_whisper` as appropriate. Do not use system Python, pip, or pytest; the command environment may not inherit an activated virtual environment. Use `/opt/homebrew/bin/ffmpeg` and `/opt/homebrew/bin/ffprobe` for media commands.

Real transcription must use local MLX Whisper from `.venv`, never an OpenAI API or another cloud transcription service. A Metal error inside the Codex sandbox does not establish whether MLX Whisper works on the host Mac. Ask the user for approval before testing real transcription outside the sandbox.

## Input Modes and Pipeline

**TALKING_HEAD is the primary mode.** Read `inbox/<project>/talking_head.mp4`. Its original audio is the primary narration. Inspect the input with ffprobe, extract narration audio with FFmpeg, and transcribe it locally with MLX Whisper. Preserve segment timestamps and word timestamps when available; write a normalized `transcript.json`. Split narration into semantic scenes and choose a visual for each scene: `A_ROLL` (original talking head), `B_ROLL` (supporting video), `IMAGE` (still image), or `GRAPHIC` (text, chart, title, or graphic placeholder). Narration must remain continuous when B-roll replaces visible A-roll.

**VOICEOVER is the secondary mode.** Accept a standalone narration audio file, for example `./.venv/bin/python factory.py --voice inbox/voice.wav --project demo`. Build its entire visual track from B-roll, images, and graphics.

In both modes, persist editing decisions in structured JSON rather than agent memory. Generate `timeline.json` as the authoritative timeline, render drafts with FFmpeg, and validate rendered outputs with ffprobe.

## AI Director V2A

`director_plan.json` records editorial intent; `scenes.json` and `timeline.json` record resolved visuals and render instructions. The default `--director rule` is deterministic and offline. Use `--director codex` only when explicitly requested; it makes one non-interactive `codex exec` call with a transcript-only prompt, `--ephemeral`, `--sandbox read-only`, the structure-only `config/codex_director_output.schema.json`, and an isolated temporary working directory. Never grant it repository write access or pass repository code or video. Validate the result again with VideoFactory's internal `config/director_plan.schema.json` and application rules before saving or rendering; VideoFactory creates provider metadata itself. Automated tests must mock Codex and require no Codex quota, internet, or Metal.

`--plan-only` reads an existing project and transcript and updates only `director_plan.json`; it must not transcribe or render. Keep the presenter visible through uncovered TALKING_HEAD gaps. If no explicitly matched local B-roll or image exists, retain the visual query as unresolved intent and render A-roll (or a graphic for VOICEOVER). Do not claim an arbitrary local file matches an AI query.

## Source Finder V3A

`--find-sources --source-provider pexels` is the sole opt-in Pexels candidate search path. Read `PEXELS_API_KEY` from the environment; never persist or log it. Search B-roll video queries, deduplicate identical normalized queries within a run, preserve candidate provenance, and save unselected results in `sources.json`. A-roll and graphics require no search. Pexels image search, candidate selection, download, and external media rendering are outside V3A. `--source-provider local` uses only explicit local asset query tags. Tests must mock HTTP and make no real external requests.

## Project Data and Directories

Keep persistent state for each episode in `projects/<project_name>/`. Every project must contain `project.json`, `transcript.json`, `scenes.json`, `timeline.json`, and `sources.json`. Every structured file must include `schema_version`. `timeline.json` must contain enough source references, timing, visual decisions, and audio decisions to reproduce the edit without recalling earlier agent decisions.

| Directory | Purpose |
| --- | --- |
| `inbox/` | User-provided source media |
| `projects/` | Persistent metadata and editing decisions |
| `assets/broll/` | Reusable local B-roll |
| `assets/images/` | Reusable still images |
| `assets/music/` | Music assets for future versions |
| `temp/` | Intermediate and generated files |
| `output/` | Final rendered videos |
| `tests/` | Automated tests |

## Source and Git Safety

Never modify original source media, overwrite user media, or automatically delete user files. Never run `rm -rf` against project or user-media directories. Put intermediate files in `temp/` and final renders in `output/`. Do not commit large or generated media, secrets, or `.env`; keep `.env` ignored.

Never change the configured Git remote, global Git configuration, or `~/.ssh`. Inspect `git status` before commits. Do not push to GitHub unless the user explicitly requests it, and never force push.

## Implementation and Media Standards

Prefer simple, maintainable Python without unnecessary frameworks. Separate business logic from FFmpeg command construction. Use `pathlib` for paths, safe `subprocess` calls, type hints on public functions, and dataclasses or other clear typed models for project structures. Create reusable FFmpeg and ffprobe helpers. Report useful errors; never silently ignore FFmpeg, transcription, or validation failures.

Keep real transcription separate from editing and rendering so those components can be tested without a GPU.

Do not assume inputs share a resolution, aspect ratio, frame rate, codec, or audio format. The default final output target is 1920×1080 at 30 fps, H.264 video, and AAC audio.

## Testing and Development Workflow

Use pytest through `./.venv/bin/pytest`. Run tests after meaningful changes, focusing on non-video business logic where possible. Unit and integration tests must not require Metal. Integration tests should use deterministic fixture transcripts and may use synthetic media generated with FFmpeg; never commit generated test media. Claim tests passed only when they actually ran successfully.

Build incrementally. Before large changes, inspect the repository and state a brief implementation plan. Then implement, run tests, perform integration validation where appropriate, report failures honestly, and fix practical issues.
