# VideoFactory

VideoFactory is a local automated video-editing pipeline for macOS Apple Silicon. It turns narration and local visuals into a 1920×1080, 30 fps H.264/AAC draft. Python 3.12, FFmpeg/ffprobe, and local MLX Whisper handle media and transcription. V2A adds an editorial DirectorPlan; V3A discovers external candidates; V3B1 selects or rejects them; V3B2 downloads selected assets; V3C plans source ranges; V3D renders the layered edit; V4A plans conservative speech-gap cuts and a canonical time map without rendering. Fixture transcripts keep automated tests independent of Metal.

## Architecture

`factory.py` orchestrates independent stages: input validation → ffprobe → narration extraction → transcription → transcript normalization → DirectorPlan → local asset resolution → scenes → timeline → FFmpeg rendering → ffprobe validation. The modules live in `videofactory/`. `director_plan.json` records what should appear; V1 `timeline.json` records actual local sources used to render. The separate V3A Source Finder writes candidate requests to `sources.json`; V3B1 writes decisions to `source_selection.json`; V3B2 writes downloaded media and `download_manifest.json`; V3C writes `clip_plan.json` and `edit_timeline.json`; V3D renders `edit_timeline.json` to a separate edited draft. `scripts/render_timeline.py` can replay a saved V1 edit.

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

Selections are `SELECTED`, `NO_SUITABLE_CANDIDATE`, or `SKIPPED`. Rejection is valid when metadata does not establish a match, such as generic breakfast footage for narration specifically about Japan. Repeated queries are decided per shot; the rule selector prefers a different equally suitable clip. The file records a reason, optional confidence and refined query, plus VideoFactory-created provider metadata. Selection does not search, download, transcribe, invoke the Director, alter the timeline, or render.

### V3B2 selected media download

This command **downloads media from the network** for selected Pexels videos only:

```sh
./.venv/bin/python factory.py --project real_test_large_001 --download-sources --max-download-mb 500
```

It reads the saved plan, candidates, and selection; it does not search Pexels or invoke Codex. Each unique selected candidate is downloaded once even when several shots reference it. The downloader chooses one MP4 variant, preferring suitable landscape 1080p over 720p or unnecessary 4K. It enforces an HTTPS-only URL, verified TLS, a 500 MiB default per-asset limit, streaming size checks, `.part` staging, SHA-256, and ffprobe validation before making a final file available. Reruns reuse a file only when its hash matches the existing manifest; mismatches fail without overwriting it. No Pexels API key is sent to the media host.

Downloaded files live in `projects/<name>/media/broll/`. `download_manifest.json` maps each candidate to all shot IDs and records its local path, source page, creator, license, selected file metadata, measured dimensions and duration, size, SHA-256, and content type. To inspect it without a network call:

```sh
./.venv/bin/python -m json.tool projects/real_test_large_001/download_manifest.json
```

### V3C clip planning and layered edit timeline

After selected media has been downloaded, run the offline planning command:

```sh
./.venv/bin/python factory.py --project real_test_large_001 --build-edit-timeline
```

This validates saved selections and local file hashes, then writes `clip_plan.json` with a result for every director shot. Selected B-roll receives a source in/out range matching the shot duration. The planner prefers a 0.5-second head/tail safety margin, reduces it when footage is tight, and spreads repeated uses of one asset across distinct non-overlapping ranges when possible. Use `--source-margin-seconds N` to change the preference. If a selected download is too short or absent from the manifest, TALKING_HEAD falls back to A-roll; a missing file named by a valid manifest is an error. VOICEOVER fallbacks are graphic placeholders.

`edit_timeline.json` has continuous PRIMARY audio from 0 through the full probed project duration. TALKING_HEAD also has continuous PRIMARY video, including silent gaps before, between, and after director shots; selected B-roll is a visual overlay with its source audio disabled. VOICEOVER has no base video and explicitly uses graphic placeholders for uncovered intervals. Source paths are checked against the project directory, including symlink escapes. V3C chooses technically valid ranges without inspecting frames to find an ideal moment. The V1 renderer continues to use `timeline.json`.

Inspect the plans without rendering:

```sh
./.venv/bin/python -m json.tool projects/real_test_large_001/clip_plan.json
./.venv/bin/python -m json.tool projects/real_test_large_001/edit_timeline.json
```

### V3D compositing render

Render the saved TALKING_HEAD edit with:

```sh
./.venv/bin/python factory.py --project real_test_large_001 --render-edit --encoder libx264
```

V3D uses the primary video as the full-duration base, overlays the saved B-roll windows, and maps only the original narration audio. B-roll audio is muted. Repeated use of one B-roll file opens it once and splits it into independent source-range branches. The render uses 1920×1080 center-fill normalization, 30 fps, H.264/AAC, and strips source metadata. It writes `output/<name>/edited_draft.mp4` after validating a temporary MP4, plus `projects/<name>/render_manifest.json` with codec, stream, duration, size, and SHA-256 facts. The V1 `output/<name>/draft.mp4` remains separate. An existing edited draft requires `--overwrite-render`; `--encoder h264_videotoolbox` remains available.

V3D adds no transitions, music, subtitles, graphics, or editorial changes. It rejects unresolved GRAPHIC overlays. VOICEOVER V3C timelines currently have no renderable base visual, so V3D reports that mode as unsupported; V1 voiceover rendering remains available.

### V4A conservative speech-gap planning

Plan cuts and a canonical time map from an existing project without modifying media:

```sh
./.venv/bin/python factory.py --project real_test_large_001 --plan-speech-edits
```

V4A reads `project.json`, `transcript.json`, and `edit_timeline.json`. Local FFmpeg `silencedetect` analyzes only the primary audio stream for acoustic pauses; it writes no media. By default, V4A shortens detected internal silences of at least 1.0 second, retains 0.25 second of natural pause, and uses a −35 dB detection threshold. `--pause-threshold-seconds N`, `--pause-keep-seconds N`, and `--silence-noise-db N` adjust these values. At least 0.04 second remains at each detected silence edge, even when keep is set to zero. Multiple opening silence regions and ending ambience are preserved.

Whisper word timestamps are validated for transcript integrity and used to identify the first and last speech boundaries. They are not treated as sample-accurate silence boundaries: a validated acoustic silence can remain eligible even if a stretched word timestamp overlaps it. Malformed, overlapping, missing, or non-finite word timing causes an error.

The command writes `projects/<name>/speech_edit_plan.json` with cuts, detected-silence bounds, detection settings, complementary keep segments, duration totals, and B-roll shot IDs intersected by each cut. It also writes `retime_map.json`, whose segments map kept source time to edited time. The reusable `map_source_time` helper raises for timestamps inside a removed region unless the caller explicitly chooses a boundary bias. V4A does not alter B-roll choices or source ranges, apply punch-ins, change audio or video, or render. Future V4B can apply the same cuts to all layers.

### V4B speech-shortened compositing

V4B consumes the saved V4A plan and map; it does not detect silence again. It builds `projects/<name>/retimed_edit_timeline.json`, then renders again from the original primary media and selected local B-roll:

```sh
./.venv/bin/python factory.py --project real_test_large_001 --render-speech-edits --encoder libx264
```

The renderer trims every canonical keep segment from both primary video and primary audio and concatenates them in the same order. It maps B-roll through intersections with those keep segments. A B-roll overlay crossing a removed interval becomes separate pieces with advancing source ranges, so the footage is cut rather than stretched or restarted. B-roll audio remains muted. V4B writes `output/<name>/speech_edited_draft.mp4` and `projects/<name>/speech_render_manifest.json`; the V3D `edited_draft.mp4` remains separate. `--overwrite-render` applies only to the speech-edited output in this mode.

### V4C static jump-cut framing

V4C styles the existing V4B primary keep segments with a deterministic 100% / 108% / 100% / 108% framing pattern. The default `--punch-in-scale` is `1.08`; allowed values are greater than 1.0 and at most 1.20. Framing is static within each segment and uses a center crop after the proven 1920×1080 primary normalization. The same canonical segments drive unchanged primary audio. B-roll positions, source ranges, and muted audio policy are copied exactly from `retimed_edit_timeline.json`.

```sh
./.venv/bin/python factory.py --project real_test_large_001 --render-punch-ins --punch-in-scale 1.08 --encoder libx264
```

The command writes `projects/<name>/jump_cut_style_plan.json`, `styled_edit_timeline.json`, and `punch_in_render_manifest.json`, and renders `output/<name>/punch_in_draft.mp4`. Existing V3D and V4B drafts stay separate. V4C does not yet add face tracking, animated zoom, subtitles, music, or transitions.

### V5A sidecar accessibility captions

V5A exports UTF-8 SRT and WebVTT caption tracks from saved transcript word timestamps. It maps timestamps to the speech-edited timeline by collapsing removed acoustic silence to each cut boundary; words spanning a cut keep their text. The final duration comes from `retimed_edit_timeline.json`. Whisper timestamps are reused as supplied, with no retranscription. V4C punch-in framing leaves timing unchanged, so the sidecars can accompany either the speech-edited or punch-in draft for a long-form YouTube upload.

```sh
./.venv/bin/python factory.py --project real_test_large_001 --export-captions
```

The command creates `projects/<name>/caption_timeline.json`, `projects/<name>/caption_export_manifest.json`, `output/<name>/captions.srt`, and `output/<name>/captions.vtt`. Existing caption sidecars require `--overwrite-captions` to replace. Captions are not burned into video and no media is modified. Sparse visual emphasis planning is handled separately by V5B.

### V5B sparse Smart Emphasis planning

V5A sidecars provide accessibility captions. V5B makes a separate, sparse semantic plan for a few spoken phrases; future V5C will render that plan visually. V5B does not modify video, audio, or the caption sidecars.

```sh
./.venv/bin/python factory.py --project real_test_large_001 --plan-emphasis --emphasis-selector rule
./.venv/bin/python factory.py --project real_test_large_001 --plan-emphasis --emphasis-selector codex
```

The default rule selector is deterministic and offline. It favors concise phrases with natural cue or connector boundaries and gives small penalties to phrases that begin or end with common English or Russian function words; other languages use neutral word-boundary scoring. Codex is opt-in. Both selectors choose only IDs from `emphasis_candidates.json`; VideoFactory copies text and timing from validated contiguous transcript tokens into `emphasis_plan.json`. Candidates stay inside one caption cue and one visible primary keep region. B-roll-covered intervals are excluded. A readable display window is centered near each phrase: 0.2 seconds of lead plus 0.4 seconds of hold, expanded to at least 1.5 seconds and capped at 3.0 seconds, then clamped inside the visible region. Phrases that cannot fit are omitted. The default gap between selected windows is 5 seconds; `--emphasis-min-gap-seconds` changes it. `--emphasis-max-count 0` uses roughly three items per minute, capped at 12, as a maximum rather than a quota. The plan stores natural spoken text without visual styling.

Inspect the plan and map before any timing changes to media:

```sh
./.venv/bin/python -m json.tool projects/real_test_large_001/speech_edit_plan.json
./.venv/bin/python -m json.tool projects/real_test_large_001/retime_map.json
```

## Tests and synthetic integration

```sh
./.venv/bin/python -m pytest
./.venv/bin/python scripts/synthetic_integration.py
/opt/homebrew/bin/ffprobe -v error -show_format -show_streams -of json output/synthetic_v1/draft.mp4
```

The integration utility generates a red talking-head placeholder with continuous tone audio, plus blue and green B-roll clips with explicit local query sidecars. Its five timed fixture segments produce A-roll → B-roll → A-roll → B-roll → A-roll through the offline rule provider. Generated source media stays under `temp/`; the draft stays under `output/`. Neither is tracked by Git. The script overwrites only its generated draft when rerun.

Replay a saved edit with `./.venv/bin/python scripts/render_timeline.py --project synthetic_v1`. This writes `output/synthetic_v1/replay.mp4` and refuses to overwrite an existing replay unless `--overwrite-render` is supplied.

## Project state

Every new `projects/<name>/` directory contains `project.json` (mode, input, settings), `transcript.json` (normalized timed segments and optional words), `director_plan.json` (editorial shots, reasons, provider/model, invocation count and elapsed time), `scenes.json` (resolved visuals or preserved unresolved queries), `sources.json` (local source metadata plus optional V3A search requests), and `timeline.json` (exact visual events and continuous narration source). V3B1 adds `source_selection.json`; V3B2 adds `download_manifest.json` and selected media; V3C adds `clip_plan.json` and `edit_timeline.json`; V3D adds `render_manifest.json`. These later files are not inputs to the V1 renderer. Every document has `schema_version`. Referenced local media must remain available to reproduce an edit.

V3D composites the saved edit but does not select music, style subtitles, publish videos, or generate images.
