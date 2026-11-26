"""``mediasift stats`` and ``mediasift report``."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import click

from mediasift.api import LibraryStats, ReportSummary, human_size
from mediasift.cli.common import (
    Leaf,
    emit_json,
    env_of,
    open_library,
    out,
    root_json,
    root_line,
    say,
    show,
)


def stats_json(stats: LibraryStats) -> dict[str, Any]:
    last = stats.last_scan
    return {
        "library": show(stats.directory),
        "roots": [root_json(r) for r in stats.roots],
        "files": [
            {"media_type": s.media_type, "status": s.status, "count": s.count, "bytes": s.bytes}
            for s in stats.files
        ],
        "last_scan": None
        if last is None
        else {
            "id": last.id,
            "state": last.state,
            "started_at": last.started_at,
            "finished_at": last.finished_at,
        },
        "issues": stats.issue_count,
        "duplicate_groups": stats.groups,
        "reclaimable_bytes": stats.reclaimable_bytes,
        "duplicates_searched_at": stats.exact_run_at,
    }


def _print_stats(stats: LibraryStats) -> None:
    out(f"Library: {show(stats.directory)}")
    out("Folders:" if stats.roots else "Folders: none yet (mediasift roots add PATH)")
    for root in stats.roots:
        out("  " + root_line(root))
    out(f"Files: {stats.present_files} present, {human_size(stats.present_bytes)}")
    for row in stats.files:
        out(f"  {row.media_type:<6} {row.status:<11} {row.count:>8}  {human_size(row.bytes)}")
    last = stats.last_scan
    if last is None:
        out("Last scan: never")
    else:
        out(f"Last scan: #{last.id} {last.state}, {last.started_at}, {stats.issue_count} problems")
    if stats.exact_run_at is None:
        out("Duplicates: not searched yet (mediasift dupes)")
    else:
        out(
            f"Duplicates: {stats.groups} groups, {human_size(stats.reclaimable_bytes)} can be "
            f"freed (searched {stats.exact_run_at})"
        )


@click.command(cls=Leaf)
def stats() -> None:
    """Show what the library holds."""
    with open_library(write=False) as library:
        found = library.stats()
    if env_of().json:
        emit_json("stats", stats_json(found))
    else:
        _print_stats(found)


def report_json(summary: ReportSummary) -> dict[str, Any]:
    return {
        "groups": summary.groups,
        "files": summary.files,
        "reclaimable_bytes": summary.reclaimable_bytes,
        "csv": None if summary.csv is None else show(summary.csv),
        "html": None if summary.html is None else show(summary.html),
    }


@click.command("report", cls=Leaf)
@click.option("--csv", "csv_file", type=click.Path(dir_okay=False, path_type=Path))
@click.option("--html", "html_file", type=click.Path(dir_okay=False, path_type=Path))
def report(csv_file: Path | None, html_file: Path | None) -> None:
    """Write the duplicate groups found by the last `dupes` to a CSV file or an HTML page.

    The HTML page is one self-contained file: no scripts, nothing loaded from the web.
    """
    if csv_file is None and html_file is None:
        raise click.UsageError("give --csv FILE, --html FILE or both")
    with open_library(write=False) as library:
        summary = library.export_report(csv=csv_file, html=html_file)
    if env_of().json:
        emit_json("report", report_json(summary))
    else:
        for written in (summary.csv, summary.html):
            if written is not None:
                say(f"Wrote {show(written)}")
        out(f"{summary.groups} groups, {human_size(summary.reclaimable_bytes)} can be freed.")
