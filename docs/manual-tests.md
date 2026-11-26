# Manual tests

Gates that need real media, a real PC setup or a phone. Automated tests cover everything else.
Status is one of: **pending manual test**, **passed** (date, notes), **failed** (date, notes).

Before any manual test: work on a **copy** of a folder, or a folder you have a backup of. MediaSift never deletes, but a test is a test.

## G1: scan a real folder (phase 1)

Status: **pending manual test**

1. `pip install -e ".[heic,raw]"` in a fresh virtual environment.
2. `mediasift roots add "<a copy of a photo folder>"`, then `mediasift scan`.
   Expect: a summary with counts per media type, no traceback, exit code 0 or 3 (3 = some files had problems; they are listed).
3. Run `mediasift scan` again. Expect: "0 hashed", and it is much faster.
4. Rename a photo and move another to a subfolder, then scan. Expect: "moved 2", no new or missing files.
5. `mediasift dupes`. Spot-check three groups by opening the files: they must be identical.
6. `mediasift stats` and `mediasift report --html report.html`: open the report in a browser; Arabic names display correctly.
7. Unplug a USB drive that is a root (or rename the root folder), scan. Expect: "root unreachable", and its files are not reported missing.

## G2: safe actions on a real PC (phase 2)

Status: **pending manual test**

1. On a copy with known duplicates: `mediasift review --auto`, then `mediasift apply` (dry run). Expect: a plan with space freed; nothing changes on disk.
2. `mediasift apply --yes`. Expect: duplicates are in `.mediasift-quarantine` inside the root; one copy of each group stays.
3. `mediasift undo`. Expect: every file back with the same name, content and modified time.
4. Repeat 2 and 3 with `--to trash`:
   - Windows: the files appear in the Recycle Bin, and undo takes them out again.
   - Linux: the same with the desktop Trash.
5. Repeat on a USB stick (FAT32/exFAT): `--to trash` must refuse with a clear message (no hard links there); quarantine works.
6. Repeat on a network share: trash must refuse; quarantine works.
7. Mark a file read-only, apply: it is skipped with "read-only".
8. Edit a file after scanning, then apply: it is skipped with "changed since the scan".

## G3: near-duplicate photos (phase 3)

Status: **pending manual test**

1. On a copy of a real photo folder that holds WhatsApp copies, resized exports and screenshots of photos: `mediasift similar`.
2. Look through 30 groups: count the groups with a photo that does not belong (precision) and note known pairs that were missed (recall).
3. `mediasift bursts`: groups of shots taken seconds apart; none is marked for removal automatically.
4. Write the counts here, with the threshold used.
