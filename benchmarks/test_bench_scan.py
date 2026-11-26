"""The benchmark at a small size, run on demand: ``pytest benchmarks -m bench``."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.bench_scan import SCHEMA, main, run_benchmark

pytestmark = pytest.mark.bench


def test_small_run_is_correct_and_reports_everything() -> None:
    result = run_benchmark(2000)
    assert result["ok"], result["checks"]
    assert result["schema"] == SCHEMA
    assert result["exact"]["groups"] == 199  # every tenth file but the first copies its neighbour
    assert result["memory"]["peak_bytes"] > 0
    for phase in ("first_scan", "rescan", "exact"):
        assert result[phase]["seconds"] >= 0


def test_command_line_writes_json(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "bench.json"
    assert main(["--files", "500", "--output", str(out)]) == 0
    assert json.loads(out.read_text(encoding="utf-8")) == json.loads(capsys.readouterr().out)
