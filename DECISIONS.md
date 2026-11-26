# Decisions

Add only. Never rewrite past entries.

| Date | Decision | Options considered | Why | Task / PR |
|---|---|---|---|---|
| 2026-11-25 | The plan was written from the planning prompt and work started without a separate approval round | wait for approval; start | the owner chose "write it, start building" | plan |
| 2026-11-25 | Public GitHub repository, commits under the GitHub noreply address | private; local only | owner's choice | plan |
| 2026-11-25 | Default removal is quarantine, Recycle Bin / Trash is opt-in | bin by default | Windows can silently delete for good when the bin is full, disabled or unsupported; a same-volume rename always undoes | plan A6 |
| 2026-11-25 | Guard hard link around every Recycle Bin move | trust the shell; own `$Recycle.Bin` writer | a lost file is recoverable from the guard before it is released | plan 1.6 |
| 2026-11-25 | BLAKE2b-256 from the standard library for content hashes | BLAKE3, xxHash, SHA-256 | no dependency, disk is the bottleneck, collision-resistant | plan 2 |
| 2026-11-25 | Multi-index hashing instead of a BK-tree | BK-tree; all pairs | exact results too, but vectorises in NumPy; a Python BK-tree is slow at 100k hashes | plan 3.4 |
| 2026-11-25 | Own perceptual hashes instead of `imagehash` | imagehash | avoids SciPy; about 100 lines | plan 2 |
| 2026-11-25 | `pi-heif` at run time, `pillow-heif` only in tests | pillow-heif everywhere | keeps the GPL x265 encoder out of the app | plan 2 |
| 2026-11-25 | Do not plan on `mutagen` (GPL) for phase 7; use `tinytag` and ffmpeg | mutagen | keeps an MIT-licensed bundle clean; owner may override | plan 2 |
| 2026-11-25 | Keeper scoring for all media moved before scale work (phases 8 and 9 reordered) | original order | performance work should measure the finished pipeline | plan 3 |
| 2026-11-25 | One writer connection with `synchronous=FULL` instead of a second connection for the operation log | separate FULL connection | a second writer in the same process can only deadlock against the first; scans commit in batches | 1.2a, PR 2 |
| 2026-11-25 | A carried-over user "remove" survives a new match run only if the new group still holds a file of the old group that was not marked remove; "keep" wins disagreements | carry every user decision | the user never agreed to lose a file in favour of files never shown to them | 1.2a, PR 2 |
| 2026-11-25 | Ids are never reused (AUTOINCREMENT); the operation log has no foreign key to files and refuses deletes (triggers) | plain rowids, cascades | an old id must never come to mean another file; history must survive index changes | 1.2a, PR 2 |
| 2026-11-25 | Paths stored as exact bytes; keys are NFC (and casefolded on case-insensitive roots) blobs, not unique | text paths | undecodable Linux names and lone Windows surrogates must survive | 1.2a, PR 2 |
| 2026-11-25 | Only name-surrogate reparse points (symlinks, junctions, mounts) count as links; OneDrive and dedup reparse points are files | all reparse points | otherwise whole OneDrive folders would be skipped | 1.2a, PR 2 |
| 2026-11-25 | A stored feature error is not retried until the file changes; transient errors (locks, permissions) are never stored | retry every scan | a corrupt file would be re-read on every scan | 1.2a, PR 2 |
| 2026-11-25 | SQLite 3.37+ required (STRICT tables) | no STRICT | type safety in the index; every supported Python ships a newer SQLite | 1.2a, PR 2 |
| 2026-11-25 | Own daemon-thread pool instead of `ThreadPoolExecutor` | ThreadPoolExecutor | about 4x less overhead per item, and executor threads are joined at exit, so a read stuck on a dead share would block even a second Ctrl+C | 1.2b, PR 3 |
| 2026-11-25 | On cancel, running items finish and their results are committed; `JobCancelled` is raised even if nothing was skipped | drop in-flight results | finished work is never lost, so a resumed job does only what is left; a cancelled job never looks complete | 1.2b, PR 3 |
| 2026-11-25 | Test image variants return encoded bytes plus extension; ffmpeg fixtures are bit-exact | PIL images | each variant carries its own format (WhatsApp JPEG, screenshot PNG) | 1.2c, PR 1 |
| 2026-11-25 | Pillow's own pixel limit is switched off at import; our 300 MP limit is checked on the header before decoding | keep Pillow's 179 MP limit | 200 MP phone photos are ordinary; all decoding goes through the readers | 3.1, PR 4 |
| 2026-11-25 | PNG EXIF is read only when it precedes the pixel data | let Pillow decode to find it | metadata must stay cheap and safe for any image size | 3.1, PR 4 |
| 2026-11-25 | HEIC: libheif's container rotation (`irot`/`imir`) wins over EXIF orientation | always apply EXIF | avoids rotating twice | 3.2, PR 4 |
| 2026-11-25 | RAW pixels from the embedded preview when it covers the requested size, else half-size demosaic | always demosaic | many times faster, and it is what the camera showed | 3.2, PR 4 |
