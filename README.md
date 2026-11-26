# MediaSift

Find duplicate photos, videos and audio on Windows and Linux, pick the best version, and clean up **without ever losing a file by accident**.

> Early development: see [docs/PLAN.md](docs/PLAN.md) for the roadmap.

## Promises

- **Nothing is deleted directly.** Files move to a quarantine folder, or to the Recycle Bin / Trash.
- **Everything can be undone,** a whole session at a time.
- **Local only.** No cloud, no online lookups.

## Install (from source)

```
pip install -e .
mediasift --help
```

Python 3.11 or newer. Optional: `pip install -e ".[heic,raw]"` for HEIC and RAW photos.

## Usage

```
mediasift scan D:\Photos        # adds the folder the first time, then scans it
mediasift dupes                  # finds identical files, biggest savings first
mediasift dupes --all --min-size 1MB
mediasift stats                  # what the library holds
mediasift report --html dupes.html --csv dupes.csv
```

Folders: `mediasift roots add|list|disable|enable|accept-volume`. `scan` takes folder numbers
(from `roots list`) or paths. Nothing is changed on disk by these commands.

Global options go before the command: `--library PATH` (default: your data folder, or
`MEDIASIFT_HOME`), `--json` (one JSON document on stdout, `"schema": "mediasift/1"`; messages
and progress go to stderr), `--debug`, `-v`.

Exit codes: 0 done, 1 failure, 2 wrong usage, 3 done but some files had problems, 4 refused for
safety, 5 the library is in use by another MediaSift, 130 cancelled with Ctrl+C (what was
indexed so far is kept).

## License

MIT
