"""Time a scan, a rescan and exact matching on a large generated tree.

Run from the repository root:

    python -m benchmarks.bench_scan --files 100000

One JSON document is printed on stdout (and written to ``--output``). Peak memory is the
operating system's peak figure for this process; it includes building the tree, so the
document also reports the peak before the library did any work (``baseline``) and the
peak after each phase. Where the system gives no figure, ``tracemalloc`` is used (Python
allocations only, slower).
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import tempfile
import time
import tracemalloc
from collections.abc import Callable
from pathlib import Path
from typing import Any

from tests.media.tree import cheap_tree

from mediasift.api import Library
from mediasift.core.config import JobSettings, Settings

SCHEMA = "mediasift-bench/1"


def _windows_peak() -> int | None:
    """Peak working set of this process, from the Windows API."""
    import ctypes
    from ctypes import wintypes

    class Counters(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    counters = Counters()
    counters.cb = ctypes.sizeof(counters)
    kernel = ctypes.windll.kernel32  # type: ignore[attr-defined,unused-ignore]
    psapi = ctypes.windll.psapi  # type: ignore[attr-defined,unused-ignore]
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD]
    ok = psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb)
    return int(counters.PeakWorkingSetSize) if ok else None


def _posix_peak() -> int | None:
    import resource

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports KiB, macOS bytes.
    return int(peak) if sys.platform == "darwin" else int(peak) * 1024


def _native_peak() -> int | None:
    try:
        return _windows_peak() if sys.platform == "win32" else _posix_peak()
    except (ImportError, OSError, AttributeError):
        return None


class MemoryProbe:
    """Peak memory in bytes, from the operating system or else from tracemalloc."""

    def __init__(self) -> None:
        self.method = "os-peak-rss"
        if _native_peak() is None:
            self.method = "tracemalloc"
            tracemalloc.start()

    def peak(self) -> int:
        if self.method == "tracemalloc":
            return tracemalloc.get_traced_memory()[1]
        return _native_peak() or 0


def _timed(work: Callable[[], Any]) -> tuple[float, Any]:
    started = time.perf_counter()
    result = work()
    return round(time.perf_counter() - started, 3), result


def run_benchmark(files: int, work_dir: Path | None = None) -> dict[str, Any]:
    """Build a tree of ``files`` small files, then time the library on it.

    The numbers are checked against what the tree plants, so a fast but wrong run fails.
    """
    with tempfile.TemporaryDirectory(prefix="mediasift-bench-", dir=work_dir) as base_name:
        base = Path(base_name)
        probe = MemoryProbe()
        tree_seconds, tree = _timed(lambda: cheap_tree(base / "media", files))
        baseline = probe.peak()
        settings = Settings(jobs=JobSettings(batch_size=1000))
        with Library.open(base / "library", settings=settings) as library:
            library.add_root(base / "media")
            first_seconds, first = _timed(library.scan)
            after_scan = probe.peak()
            again_seconds, again = _timed(library.scan)
            exact_seconds, exact = _timed(library.find_exact)
            after_exact = probe.peak()
            reexact_seconds, reexact = _timed(library.find_exact)
        checks = {
            "first_scan_saw_every_file": first.seen == files and first.new == files,
            "rescan_changed_nothing": again.new == again.changed == again.moved == 0
            and again.unchanged == files,
            "found_the_planted_groups": exact.groups == tree.duplicate_groups,
            "second_matching_hashed_nothing": reexact.hashed_samples == reexact.hashed_full == 0,
        }
        return {
            "schema": SCHEMA,
            "files": files,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "tree_seconds": tree_seconds,
            "first_scan": {
                "seconds": first_seconds,
                "files_per_second": _rate(files, first_seconds),
            },
            "rescan": {"seconds": again_seconds, "files_per_second": _rate(files, again_seconds)},
            "exact": {
                "seconds": exact_seconds,
                "groups": exact.groups,
                "hashed_samples": exact.hashed_samples,
                "hashed_full": exact.hashed_full,
            },
            "exact_again": {"seconds": reexact_seconds},
            "memory": {
                "method": probe.method,
                "baseline_peak_bytes": baseline,
                "peak_after_scan_bytes": after_scan,
                "peak_after_exact_bytes": after_exact,
                "peak_bytes": probe.peak(),
            },
            "checks": checks,
            "ok": all(checks.values()),
        }


def _rate(files: int, seconds: float) -> int:
    return round(files / seconds) if seconds > 0 else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark scanning and exact matching.")
    parser.add_argument("--files", type=int, default=10_000, help="files to generate")
    parser.add_argument("--output", type=Path, help="also write the JSON here")
    parser.add_argument("--dir", type=Path, help="folder for the generated tree (default: temp)")
    args = parser.parse_args(argv)
    if args.files < 10:
        parser.error("--files must be at least 10")
    result = run_benchmark(args.files, args.dir)
    text = json.dumps(result, indent=2)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
