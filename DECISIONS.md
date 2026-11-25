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
