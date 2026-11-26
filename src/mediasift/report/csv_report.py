"""CSV report: one row per file, so a spreadsheet can sort and filter it."""

from __future__ import annotations

import csv
from collections.abc import Iterable
from pathlib import Path

from mediasift.report.model import ReportGroup, atomic_text

COLUMNS = (
    "group",
    "kind",
    "file_size_bytes",
    "group_reclaimable_bytes",
    "path",
    "media_type",
    "modified",
    "hardlink_of",
)

# A cell that starts like this is read as a formula by spreadsheets.
_FORMULA_STARTS = ("=", "+", "-", "@", "\t", "\r")


def safe_cell(text: str) -> str:
    """Stop a file name such as ``=HYPERLINK(...)`` from running as a formula."""
    return "'" + text if text.startswith(_FORMULA_STARTS) else text


def write_csv(path: Path, groups: Iterable[ReportGroup]) -> int:
    """Write the groups and return the number of files written.

    UTF-8 with a byte order mark: Excel on Windows needs it to show Arabic names correctly.
    """
    files = 0
    with atomic_text(path, encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(COLUMNS)
        for group in groups:
            for member in group.members:
                writer.writerow(
                    [
                        group.number,
                        group.kind,
                        member.size,
                        group.reclaimable_bytes,
                        safe_cell(member.path),
                        member.media_type,
                        member.modified,
                        safe_cell(member.hardlink_of),
                    ]
                )
                files += 1
    return files
