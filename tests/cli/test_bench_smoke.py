"""The benchmark must not rot: run it tiny in the normal suite."""

from __future__ import annotations

from benchmarks.bench_scan import run_benchmark


def test_benchmark_runs_and_checks_itself() -> None:
    result = run_benchmark(100)
    assert result["ok"], result["checks"]
    assert result["exact"]["groups"] == 9
    assert result["memory"]["peak_bytes"] > 0
