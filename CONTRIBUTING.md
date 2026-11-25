# Contributing

```
python -m venv .venv
pip install -e ".[dev,heic,raw]"
ruff check . && ruff format --check . && mypy && pytest
```

- One branch and one pull request per task in [docs/PLAN.md](docs/PLAN.md).
- The core never imports a GUI library; interfaces use `mediasift.api` only.
- Test media is generated inside the tests. Never add real photos, videos or audio.
- Code that moves files needs tests for dry run, undo, files changed since the scan, and read-only targets.
- Log non-trivial decisions at the end of [DECISIONS.md](DECISIONS.md); never edit past entries.
