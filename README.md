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

## License

MIT
