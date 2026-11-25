# MediaSift: long-term implementation plan

This document is the spec. Where it is silent, make the most conservative choice and log it in `DECISIONS.md`.

## 0. Core decisions and assumptions

**Core decisions (changing one needs the owner's approval)**

| # | Decision |
|---|---|
| C1 | Nothing is ever deleted directly: Recycle Bin / Trash or quarantine only. |
| C2 | Every action is logged before it happens and can be undone, per session. |
| C3 | The core has no GUI imports. CLI, GUI and phone page are thin layers on it. |
| C4 | Local only: no cloud, no online lookups by default. |
| C5 | MIT license. |
| C6 | All test media is generated inside the tests. No real personal media in the repo, ever. |

**Assumptions**

| # | Assumption |
|---|---|
| A1 | Name stays **MediaSift**; package and command are `mediasift`. |
| A2 | Python 3.11+, 64-bit, Windows 10+ and current Linux. macOS is untested, not deliberately broken. |
| A3 | One user, one library index per profile; one writing process at a time (lock file). |
| A4 | The index lives in the per-user data folder; `--library` or `MEDIASIFT_HOME` overrides it. |
| A5 | Symlinks, junctions and cloud placeholders are not followed or downloaded. |
| A6 | The removal default is **quarantine** (a same-volume rename, always undoable); Recycle Bin / Trash is opt-in with `--to trash`. |
| A7 | The repo stays under 5 MB. |

## 1. Architecture

### 1.1 Layers

```
cli/   gui/   web/          interfaces: no business logic, no SQL
      \  |  /
      api.py                stable facade: Library
         |
 scan  match  keeper  actions  organize  report
         |
 hashing  readers  jobs  db  core
```

- Imports only point downwards. `tests/test_architecture.py` enforces it and bans `PySide6` / `PyQt` outside `gui/`.
- Interfaces call `mediasift.api` only. Additions to it are free; breaking it needs a logged decision.

### 1.2 Package layout

| Path (`src/mediasift/`) | Content |
|---|---|
| `core/types.py` | `MediaType`, `FileRecord`, `FileSignature`, enums and frozen dataclasses |
| `core/errors.py` | `MediaSiftError` tree, `ExitCode` |
| `core/paths.py` | path keys, long paths, containment, volume and file identity |
| `core/config.py` | TOML settings with defaults |
| `core/log.py` | rotating log set-up |
| `core/plugins.py` | protocols and registries |
| `db/` | `connection.py`, `migrate.py`, `migrations/NNNN_name.sql`, one repository module per area |
| `jobs/` | cancel token, progress, bounded parallel runner, job records |
| `scan/` | rules, walker, scanner |
| `hashing/` | exact and perceptual hashers |
| `readers/` | one reader per media type or format family |
| `match/` | matchers, nearest-neighbour index, grouping |
| `keeper/` | rules and scoring engine |
| `actions/` | file-system primitives, removal backends, log, planner, executor, undo, recovery |
| `organize/`, `importer/`, `report/` | later phases (a first `report/` in phase 1) |
| `api.py` | facade |
| `cli/`, `gui/`, `web/` | interfaces |

### 1.3 Database

SQLite, WAL, `foreign_keys=ON`, `synchronous=FULL` on the operation log's connection and `NORMAL` elsewhere. One writer connection per process; readers get their own.

| Table | Key columns |
|---|---|
| `schema_migrations` | version, name, checksum, applied_at |
| `roots` | id, path (blob), display, path_key (unique), volume_id, case_sensitive, read_only, prefer_rank, enabled |
| `files` | id, root_id, rel_path (blob), rel_display, rel_key, ext, media_type, size, mtime_ns, dev, inode, nlink, status, first_scan_id, last_scan_id; unique (root_id, rel_path) |
| `features` | file_id, feature, version, value (blob), status, error, src_size, src_mtime_ns; PK (file_id, feature); index (feature, value) |
| `metadata` | file_id, reader, reader_version, width, height, orientation, duration_ms, taken_at, taken_source, camera_make, camera_model, gps_lat, gps_lon, codec, bitrate, artist, album, title, track, extra (JSON), src_size, src_mtime_ns |
| `scans` | id, started_at, finished_at, state, options (JSON), counters (JSON) |
| `scan_issues` | scan_id, root_id, path, kind, message |
| `match_runs` | id, kind, matcher, matcher_version, params (JSON), state, created_at |
| `match_groups` | id, run_id, kind, confidence, reclaimable_bytes, member_count |
| `match_members` | group_id, file_id, distance, evidence (JSON) |
| `decisions` | group_id, file_id, choice (keep / remove / undecided), source (auto / user), reason |
| `op_sessions` | id, kind, state, created_at, finished_at, summary (JSON) |
| `operations` | id, session_id, seq, action, backend, state, file_id, src (blob), dst (blob), size, mtime_ns, dev, inode, full_hash, keeper_path, detail (JSON), error, started_at, finished_at |

- **Paths** are stored as bytes (`os.fsencode`), so undecodable Linux names and lone Windows surrogates survive; `*_display` columns are for people only.
- **Features are a generic cache**: a new hasher needs no migration. A row is valid only when its `version` matches the hasher and `src_size` / `src_mtime_ns` match the file.
- **File status:** `present`, `missing`, `error`, `placeholder`.
- **Migrations:** numbered SQL files, each in one transaction, with checksums. A backup (`VACUUM INTO`) is taken before upgrading a non-empty database. A newer database, or a changed checksum, is refused with a clear error.

### 1.4 Plugin interfaces (`core/plugins.py`)

| Interface | Members |
|---|---|
| `Hasher` | `name`, `version`, `media_types`, `compute(ctx) -> Mapping[str, bytes]` |
| `Reader` | `name`, `version`, `extensions`, `available()`, `read_metadata(path)`, `load_image(path, max_side)` (images) |
| `Matcher` | `kind`, `version`, `required_features`, `find(library, params, job) -> Iterable[MatchGroup]` |
| `KeeperRule` | `name`, `media_types`, `score(candidate, group) -> RuleScore(value, reason)` |
| `RemovalBackend` | `name`, `check(path)`, `remove(path, session)`, `restore(operation)` |

Built-ins register through the same registry that entry-point plugins will use later. A missing optional library makes a reader *unavailable*, never an import error.

### 1.5 Job model

- `CancelToken` (thread-safe), `Progress(stage, done, total, bytes_done, current)`, `JobContext(cancel, on_progress)`.
- `run_bounded(items, work, workers, window, job)`: lazily pulls from an iterator, keeps at most `window` items in flight, yields results in completion order. Memory stays flat for any library size.
- Worker threads read and compute; only the caller's thread writes to the database, in batches.
- Per-item failures are results, not exceptions: one bad file never stops a job.
- **Resume** comes from the cache: each batch is committed, so a cancelled scan picks up with only the remaining work. Cancel is honoured between items and between read chunks.
- Progress callbacks are rate-limited.

### 1.6 Actions, operation log and undo

1. **Plan:** decisions become an immutable `ActionPlan` with items, space freed and warnings. Dry run prints it and stops.
2. **Check each item** immediately before acting:
   - inside an enabled root, not read-only, not a symlink, not a hard link of its keeper;
   - size, modified time and file identity unchanged; full hash re-computed and equal;
   - at least one keeper of the group exists and is unchanged (hash re-checked).
   - Any failed check skips that item and says why.
3. **Log first:** the operation row is committed as `pending` before the file moves, then set to `done` or `failed`. A JSONL manifest beside the quarantined files repeats it, so files can be restored without the database.
4. **Move:** same-volume, never overwriting.
   - Windows: `os.rename` (fails when the target exists).
   - Linux: `renameat2(RENAME_NOREPLACE)`, else refuse unless a safe no-replace fallback works.
   - Crossing a volume is refused.
   - After the move, identity is verified; a mismatch is moved back.
5. **Recovery:** on the next start, each `pending` operation is settled by looking at which of source and target exists. Ambiguous cases become `needs_attention` and are never touched automatically.

| Backend | Behaviour |
|---|---|
| Quarantine (default) | `<root>/.mediasift-quarantine/<session>/<rel path>`; the scanner always skips this folder |
| Recycle Bin (Windows) | A **guard hard link** is made in quarantine first; the file is sent to the Recycle Bin; only when the bin entry is found and is the same file is the guard released. If it is not found (full or disabled bin), the file comes back from the guard. No hard-link support: refused, with quarantine suggested |
| Trash (Linux) | freedesktop.org Trash spec: home trash on the same volume or the volume's `.Trash-$uid`; no-replace moves, `.trashinfo` written first |

- **Undo** runs in reverse order. Never overwrites: if the original path is taken, the file comes back beside it as `name (restored N).ext` and that is reported. A missing or changed stored file is reported, the rest continues. Undo is logged too.
- **Purging quarantine** is not a feature of phases 1–3; a later one may send old quarantine folders to the Recycle Bin / Trash after confirmation.

### 1.7 Interfaces on the core

| Interface | How |
|---|---|
| CLI | click + rich; `--json` gives one versioned JSON document on stdout, everything else goes to stderr |
| GUI | PySide6 models over `api.Library`; jobs on a worker thread, progress through signals |
| Phone page | stdlib HTTP server in the PC app, off by default; JSON endpoints calling `api.Library`; pairing code; can only write decisions |

**Exit codes:** 0 success · 1 failure · 2 usage · 3 finished, but some files had problems · 4 refused by a safety check · 5 library busy or incompatible · 130 cancelled.

### 1.8 Errors and logging

- `MediaSiftError` → `ConfigError`, `LibraryError` (`LibraryBusy`, `SchemaTooNew`), `ReaderError`, `SafetyError`, `ActionError`. The CLI maps them to exit codes; no traceback without `--debug`.
- Per-file problems go to `scan_issues`. Rotating log file in the library folder; logs stay local.

## 2. Libraries and tools

| Area | Choice | License | Why | Windows / Linux notes |
|---|---|---|---|---|
| Database | stdlib `sqlite3` | PSF / public domain | no dependency | check WAL on network drives; the index should be local |
| Content hash | stdlib BLAKE2b-256 | PSF | fast, releases the GIL, no dependency; the disk is the bottleneck | – |
| Arrays | NumPy ≥ 2.0 | BSD-3 | hashing and vectorised popcount | – |
| Images | Pillow | MIT-CMU | wide formats, EXIF, draft decoding | – |
| HEIC | `pi-heif` (optional extra) | BSD-3; bundles libheif / libde265 (LGPL) | decode only, so no GPL x265 encoder as in full `pillow-heif` wheels | `pillow-heif` is used by tests only, to generate HEIC |
| RAW | `rawpy` (optional extra) | MIT; LibRaw LGPL-2.1 / CDDL | previews and metadata | wheels on both |
| Perceptual hashes | own NumPy code | MIT | `imagehash` pulls in SciPy; small code, full control | – |
| Audio tags (phase 7) | `mutagen` | **GPL-2.0-or-later** | best coverage, but the licence is incompatible with shipping in an MIT-only bundle: use `tinytag` (MIT) to read and ffmpeg to write, unless the owner decides otherwise | – |
| Trash | Windows: `send2trash` (BSD) under the guard; Linux: own spec implementation | – | restoring needs the stored location | `$Recycle.Bin\<SID>` `$I` / `$R` pairs |
| Video / audio decoding | `ffmpeg` / `ffprobe` as separate programs | LGPL or GPL by build | never linked or bundled; found by setting or PATH | – |
| Audio fingerprint | `fpcalc` as a separate program | LGPL-2.1 (some builds GPL through ffmpeg) | never linked | – |
| CLI | click, rich | BSD-3, MIT | – | rich handles Windows consoles |
| Paths | platformdirs | MIT | – | – |
| GUI | PySide6 | LGPL-3 | dynamic linking only; notices in the installer | – |
| Dev | pytest, pytest-cov, hypothesis, ruff (lint and format), mypy | MIT / MPL | – | – |

## 3. Roadmap

Markings: **[H]** hard, **[S]** well-scoped. "After" = dependencies. Tasks in one milestone with no dependency between them touch different files and may run together. Every milestone also needs: `ruff check .`, `ruff format --check .`, `mypy`, `pytest` clean, CI green on Windows and Linux.

### Phase 1: foundation, scanning, exact duplicates (CLI)

| Task | Files | Behaviour | Tests | After |
|---|---|---|---|---|
| **1.1a [S]** scaffold | `pyproject.toml`, CI, README, CONTRIBUTING, LICENSE, `.gitignore` | installable package, lint, tests on both systems | import test, architecture test | – |
| **1.1b [H]** contracts | `core/types.py`, `core/errors.py`, `core/plugins.py` | shared types | unit tests | 1.1a |
| **1.2a [H]** database, paths, config | `db/*`, `core/paths.py`, `core/config.py`, `core/log.py` | §1.3 | migrations up from empty, backup, newer schema refused, checksum change refused, undecodable and Arabic names, containment | 1.1b |
| **1.2b [H]** jobs | `jobs/*` | §1.5 | window bound, cancel latency, error isolation, flat memory on 1M items | 1.1b |
| **1.2c [S]** test media kit | `tests/media/*` | generators for images, video and audio with variants | generators are deterministic; ffmpeg ones skip cleanly when it is missing | 1.1b |
| **1.3a [H]** scanner | `scan/*` | include / exclude rules, incremental rescan, moves recognised, missing only after a complete walk of a reachable root | rename, move between roots, unplugged root, files changing during the scan, permission errors, symlinks, hard links, cancel and resume | 1.2a, 1.2b |
| **1.3b [H]** exact hashing and matching | `hashing/exact.py`, `match/exact.py` | size → partial → full; hard links are not reclaimable space | same-size different content, same head and tail, empty files, changed mid-hash, cache use | 1.2a, 1.2b |
| **1.4 [S]** facade and CLI | `api.py`, `cli/*`, `report/*` | `scan`, `roots`, `dupes`, `stats`, `report` (CSV / HTML), `--json` | every command, exit codes, JSON shape, Arabic output | 1.3a, 1.3b |
| **1.5 [S]** benchmark job | `benchmarks/*`, CI job | 100k-file tree: scan, rescan, dupes; peak memory | runs in its own CI job | 1.4 |

**Done when:** `mediasift scan <dir>` then `mediasift dupes --json` on the generated tree lists exactly the planted groups, exit 0; a second `scan` reports 0 files re-hashed. **Manual gate G1:** scan a real folder.

### Phase 2: safe actions and undo

| Task | Files | Behaviour | Tests | After |
|---|---|---|---|---|
| **2.1 [H]** file-system primitives | `actions/fsops.py`, `actions/checks.py` | no-replace move, identity, re-check, containment | target exists, race after the check, cross-volume refused, read-only, symlink, long paths | 1.2a |
| **2.2 [H]** operation log and recovery | `actions/oplog.py`, `actions/manifest.py`, `actions/recovery.py` | write-ahead log, manifest, crash recovery | a crash injected at every step | 2.1 |
| **2.3 [H]** backends | `actions/backends/*` | quarantine, Recycle Bin with guard, Linux Trash | full or disabled bin simulated, restore, name collisions | 2.1 |
| **2.4 [H]** keeper engine | `keeper/*`, `match/decisions.py` | basic rules: folder preference, original-looking names, oldest date, shortest path; each with a reason; user overrides win | determinism, reason text, at least one keeper always | 1.3b |
| **2.5 [H]** planner, executor, undo | `actions/plan.py`, `actions/executor.py`, `actions/undo.py` | §1.6 | dry run changes nothing; undo restores everything; changed since the scan; read-only; undo after outside changes; cancel half-way; property test: apply then undo is the identity | 2.2, 2.3, 2.4 |
| **2.6 [S]** CLI | `cli/*`, `api.py` | `review` (show and set decisions), `apply` (dry run unless `--yes`), `undo`, `history` | every path, exit code 4 | 2.5 |

**Done when:** on a generated library `review --auto`, `apply --yes`, then `undo` leaves the tree byte-identical (names, contents, modified times). **Manual gate G2:** Recycle Bin round trip on a real PC; a USB drive; a network share.

### Phase 3: near-duplicate images

| Task | Files | Behaviour | Tests | After |
|---|---|---|---|---|
| **3.1 [S]** image reader | `readers/image.py` | metadata, orientation, reduced-size decoding, decompression-bomb limit | every format, corrupt and truncated, missing EXIF | 1.2c |
| **3.2 [S]** HEIC and RAW readers | `readers/heif.py`, `readers/raw.py` | optional; RAW preview and metadata | generated HEIC and minimal DNG; behaviour when the library is missing | 3.1 |
| **3.3 [H]** perceptual hashes | `hashing/perceptual.py` | aHash, dHash, pHash, wHash (64-bit), low-detail flag, uniform-border trimming, small colour signature | stability across resize and recompress; difference between unrelated images | 3.1 |
| **3.4 [H]** index and matcher | `match/index.py`, `match/similar.py` | multi-index hashing over 4×16-bit bands (exact results, no all-pairs comparison); candidates confirmed by several hashes and the colour signature; star-shaped groups so similarity does not chain | equals brute force on random and clustered sets; threshold edges; no chaining | 3.3 |
| **3.5 [H]** bursts | `match/bursts.py` | taken within N seconds, same camera, alike: "pick the best" groups, never auto-removed | bursts vs scene changes, missing dates | 3.4 |
| **3.6 [H]** image keeper rules | `keeper/image_rules.py` | resolution, original vs recompressed, sharpness, exposure, metadata completeness | each rule; the planted original wins | 3.3, 2.4 |
| **3.7 [S]** CLI and reports | `cli/*`, `api.py`, `report/*` | `similar`, `bursts`, HTML report | – | 3.4–3.6 |
| **3.8 [S]** quality regression | `tests/quality/*` | precision and recall per variant, floors in a checked-in file | fails when a number drops | 3.4 |

**Done when:** recall ≥ 0.95 on resize, recompress, light edits and WhatsApp-style variants, ≥ 0.80 on screenshots with borders, precision ≥ 0.98, all on generated sets. **Manual gate G3:** precision on a real photo folder.

### Phase 4: GUI review flow

| Task | Files | Behaviour | Tests | After |
|---|---|---|---|---|
| 4.1 [S] shell | `gui/app.py`, `gui/main_window.py` | overview, job panel with cancel | offscreen smoke | phase 3 |
| 4.2 [S] group list | `gui/groups/*` | paged models, biggest wins, thumbnails cache | 100k-group model test | 4.1 |
| 4.3 [H] compare view | `gui/compare/*` | synchronised zoom / pan, metadata differences | offscreen interaction | 4.1 |
| 4.4 [S] keyboard review | `gui/review.py` | keep / remove / next, undo a decision | key events | 4.2, 4.3 |
| 4.5 [S] history and undo | `gui/history.py` | sessions, undo | offscreen | 4.1 |
| 4.6 [S] Arabic, RTL | `gui/i18n/*` | English and Arabic | layout direction, missing strings | 4.1–4.5 |

**Done when:** `QT_QPA_PLATFORM=offscreen pytest tests/gui` passes; `mediasift gui` opens. **Manual gate G4:** review 50 real groups in both languages.

### Phase 5: organizing and import

| Task | Files | Behaviour | Tests | After |
|---|---|---|---|---|
| 5.1 [H] dates | `organize/dates.py` | EXIF → container → file name → modified time, source shown; time zones; bulk shift | patterns, WhatsApp names, conflicts | phase 3 |
| 5.2 [H] templates | `organize/templates.py` | safe names on both systems, collisions | property tests | – |
| 5.3 [H] organize actions | `organize/plan.py`, `actions/move.py` | move / rename with log and undo; cross-volume = verified copy, source to quarantine | same four safety tests as phase 2 | 5.1, 5.2 |
| 5.4 [H] import | `importer/*` | copy new files only, skip known content, resume | interrupted copy, full disk | 5.3 |
| 5.5 [S] CLI and GUI | `cli/*`, `gui/organize/*` | previews | – | 5.3, 5.4 |

**Done when:** `organize --dry-run`, `organize --yes`, `undo` is the identity on a generated library. **Manual gate G5:** import from a real phone and SD card.

### Phase 6: video duplicates

| Task | Files | Behaviour | Tests | After |
|---|---|---|---|---|
| 6.1 [H] ffmpeg wrapper, reader | `readers/video.py`, `core/ffmpeg.py` | probe, time-outs, rotation, variable frame rate | missing ffmpeg, corrupt | phase 3 |
| 6.2 [H] frame signatures | `hashing/video.py` | sampled hashes by time, cached | re-encode, scale, container | 6.1 |
| 6.3 [H] whole-video matcher | `match/video.py` | – | precision / recall | 6.2 |
| 6.4 [H] clips inside videos | `match/clips.py` | offset voting, time range reported | trimmed, sped-up negatives | 6.2 |
| 6.5 [H] audio track | `hashing/audio_fp.py` | fingerprint match | changed picture, same sound | 6.1 |
| 6.6 [H] phone pairs | `match/pairs.py` | Live Photos, slow-motion and edited copies kept together | pair integrity through apply and undo | 6.1 |
| 6.7 [S] CLI and GUI | `cli/*`, `gui/compare/video.py` | synchronised playback | – | 6.3–6.6 |

**Done when:** a planted clip is found with its range within ±1 s. **Manual gate G6:** real WhatsApp clips and Live Photos.

### Phase 7: audio duplicates and music organizing

| Task | Files | Behaviour | Tests | After |
|---|---|---|---|---|
| 7.1 [S] audio reader | `readers/audio.py` | tags and stream facts | every format, Arabic tags | 6.1 |
| 7.2 [H] fingerprint matcher | `match/audio.py` | formats, trims, silence | precision / recall | 6.5 |
| 7.3 [S] tag hints | `match/tag_hints.py` | separate low-confidence kind | – | 7.1 |
| 7.4 [H] voice notes | `match/voice.py` | near-exact copies | – | 7.2 |
| 7.5 [H] music organizing | `organize/music.py` | templates, tag report, optional tag fixes with undo | – | 5.3, 7.1 |
| 7.6 [S] A/B player | `gui/compare/audio.py` | waveform | offscreen | 7.2 |

**Done when:** one melody in five formats forms one group; unrelated melodies form none. **Manual gate G7:** a real music folder.

### Phase 8: keeper scoring for every media type

| Task | Files | After |
|---|---|---|
| 8.1 [H] video rules | `keeper/video_rules.py` | 6.3 |
| 8.2 [H] audio rules | `keeper/audio_rules.py` | 7.2 |
| 8.3 [H] editable rule sets | `keeper/ruleset.py` | 8.1, 8.2 |
| 8.4 [S] rule editor | `gui/rules.py` | 8.3 |

Tests: planted best version wins in ≥ 98% of generated groups. **Done when:** `pytest tests/keeper` passes and `review --explain` prints each reason. **Manual gate G8:** agreement on 50 real groups.

### Phase 9: scale and performance

| Task | Files | After |
|---|---|---|
| 9.1 [H] profiling and fixes | as measured | phase 7 |
| 9.2 [H] 500k-file run | `benchmarks/*` | 9.1 |
| 9.3 [S] I/O profiles (HDD, SSD, network) | `jobs/*`, `core/config.py` | 9.1 |

**Done when:** 100k files: rescan < 60 s on an SSD, peak memory < 500 MB, GUI list opens < 1 s. **Manual gate G9:** the real library.

### Phase 10: phone companion page

| Task | Files | After |
|---|---|---|
| 10.1 [H] server and pairing | `web/server.py`, `web/auth.py` | phase 8 |
| 10.2 [S] page | `web/static/*` | 10.1 |
| 10.3 [S] QR and GUI switch | `gui/phone.py` | 10.1 |

Tests: unpaired requests refused, pairing code expires and is rate-limited, path traversal, decisions only, Host and Origin checks. **Done when:** the API tests pass against a live local server. **Manual gate G10:** a real phone on Wi-Fi and on the PC's hotspot.

### Phase 11: optional AI matching

11.1 [H] local embedding plugin (`match/embeddings.py`, optional extra, model supplied by the user), 11.2 [H] approximate index, 11.3 [S] settings. Off by default. **Done when:** all tests pass with the extra absent and present. **Manual gate G11:** choosing a model.

### Phase 12: polish, packaging, installers

12.1 [S] Windows installer, 12.2 [S] AppImage and wheel, 12.3 [S] docs and screenshots (generated media), 12.4 [H] licence notices and release checks. **Done when:** the installers run the smoke test in CI. **Manual gate G12:** installing on a clean PC; signing needs an account and money.

## 4. Test strategy

| Topic | Approach |
|---|---|
| Images | NumPy scenes (gradients, shapes, noise textures, text-like patterns) from a seed. Variants: resize, crop, recompress, WhatsApp-style (1600 px, quality 60–75, no EXIF), brightness / contrast, screenshot with borders, rotation by EXIF, corrupt, truncated, no metadata |
| Video | ffmpeg `testsrc2`, `mandelbrot`, `life`, `sine`; variants: re-encode, scale, container, trim, clip of a longer file, no audio, rotated, variable frame rate, truncated |
| Audio | tones, noise and melodies to each format; variants: bitrate, trim, silence added, tags changed or missing |
| Matching quality | precision and recall by variant, floors checked in under `tests/quality/`; a drop needs a logged decision |
| Safety | every moving module tests dry run, undo, files changed since the scan, read-only targets; crash injection; only temporary folders are touched |
| Real bin | Recycle Bin tests are marked `realbin` and run in CI and on request only; they use temporary files and restore them |
| GUI | `QT_QPA_PLATFORM=offscreen`, models tested without widgets where possible |
| Speed | the main suite takes minutes; markers `slow`, `bench`, `realbin`, `ffmpeg` |
| Benchmarks | separate CI job on generated libraries of 100k small files |

## 5. Edge cases and risks

| Case | Handling |
|---|---|
| HEIC / RAW library missing | reader reports unavailable; exact matching still works; `stats` says what is skipped |
| ffmpeg / fpcalc missing | features disabled with an explanation, never a crash |
| Variable frame rate, rotated video | sample by timestamp; apply the display matrix |
| Video without audio, audio without tags | those matchers skip them |
| Very long recordings | chunked fingerprints, time-outs, progress inside the file |
| Files changing during a scan | stat before and after hashing; mismatch → discard, retry once, then an issue |
| Drive disconnects mid-scan | root marked unreachable; its files are never marked missing |
| Symlinks, junctions | not followed |
| Hard links | one file with several names: not reclaimable, never offered for removal against itself |
| Permissions, read-only media | issues listed; read-only roots and files are protected from actions |
| Corrupt files | issue recorded; exact hash still computed |
| Groups across drives | allowed; each file uses its own volume's quarantine; unavailable keeper → skip |
| Clock and time zones | store naive local time plus offset when known; show the source; bulk shift |
| Unicode, Arabic | bytes stored; comparison by NFC; case by volume; display isolates bidi text |
| Case-insensitive file systems | per-root `case_sensitive` probe |
| Long Windows paths | `\\?\` prefix at the file-system boundary |
| Full or disabled Recycle Bin | guard hard link; otherwise refused |
| Undo after outside changes | never overwrite; restore beside; report |
| Crash mid-action | write-ahead log, manifest, recovery |
| Two processes | lock file; exit code 5 |
| Cloud placeholders | detected by attribute, never opened |
| Decompression bombs | pixel limit |
| Similarity chaining | star-shaped groups |

## 6. Backlog

| Priority | Idea | Size |
|---|---|---|
| 1 | Google Photos Takeout import with JSON sidecars | M |
| 2 | Rotated and mirrored image matches | S |
| 3 | Scheduled scans | S |
| 4 | Purge old quarantine to the Recycle Bin / Trash | S |
| 5 | Crop-aware image matching (local features) | L |
| 6 | Third-party plugin packages (entry points) | M |
| 7 | Sidecars (XMP, AAE, SRT) travelling with their file | M |
| 8 | Online tag lookup plugin (opt-in) | M |
| 9 | Face grouping (local) | L |
| 10 | Blurry / dark photo finder | S |
| 11 | File manager integration | M |
| 12 | Map view by GPS (offline tiles) | L |
| 13 | macOS support | M |
| 14 | Index export and import between PCs | M |

## 7. Build rules

- One branch and one PR per task; tasks running together never touch the same files.
- Each PR is reviewed and its checks run by the lead before merging.
- Anything that moves or deletes files needs the four safety tests (§4) before merging.
- Non-trivial decisions go to `DECISIONS.md`, add only.
- Manual gates are marked "pending manual test" with a checklist in `docs/manual-tests.md`; work continues.
