"""HTML report: one self-contained page, no scripts, no external resources."""

from __future__ import annotations

from collections.abc import Iterable
from html import escape
from pathlib import Path

from mediasift.report.model import ReportGroup, ReportMember, atomic_text, human_size

_STYLE = """
body{font-family:system-ui,sans-serif;margin:2rem auto;max-width:60rem;padding:0 1rem;
color:#1b1b1b;background:#fff}
h1{font-size:1.5rem}
section{border:1px solid #ccc;border-radius:6px;margin:1rem 0;padding:.5rem 1rem}
h2{font-size:1rem;margin:.25rem 0}
ul{list-style:none;margin:.25rem 0;padding:0}
li{padding:.15rem 0;font-family:ui-monospace,Consolas,monospace;font-size:.9rem;
overflow-wrap:anywhere}
.meta{color:#555;font-size:.8rem;font-family:system-ui,sans-serif}
.note{color:#8a4b00}
@media (prefers-color-scheme:dark){body{color:#eee;background:#161616}
section{border-color:#444}.meta{color:#aaa}.note{color:#e0a458}}
"""


def _member(member: ReportMember) -> str:
    # dir="auto": an Arabic path reads right to left without reordering the page around it.
    path = f'<span dir="auto">{escape(member.path)}</span>'
    notes = [human_size(member.size), escape(member.modified)]
    if member.hardlink_of:
        notes.append(
            f'<span class="note">hard link of <span dir="auto">{escape(member.hardlink_of)}'
            "</span>: frees no space</span>"
        )
    return f'<li>{path} <span class="meta">{" · ".join(notes)}</span></li>'


def _group(group: ReportGroup) -> str:
    title = (
        f"Group {group.number}: {len(group.members)} files of {human_size(group.size)}, "
        f"{human_size(group.reclaimable_bytes)} can be freed"
    )
    items = "\n".join(_member(member) for member in group.members)
    return f"<section>\n<h2>{escape(title)}</h2>\n<ul>\n{items}\n</ul>\n</section>"


def write_html(
    path: Path, groups: Iterable[ReportGroup], *, title: str = "MediaSift report"
) -> int:
    """Write the groups as a page and return how many groups it holds.

    Every text is escaped: file names are data, and a file called ``<script>.jpg`` must show
    as text. The page has no script and loads nothing, so it is safe to open and to send.
    """
    count = 0
    with atomic_text(path, encoding="utf-8") as stream:
        stream.write(
            '<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
            f"<title>{escape(title)}</title>\n<style>{_STYLE}</style>\n</head>\n<body>\n"
            f"<h1>{escape(title)}</h1>\n"
        )
        for group in groups:
            stream.write(_group(group) + "\n")
            count += 1
        if count == 0:
            stream.write("<p>No duplicate files were found.</p>\n")
        stream.write("</body>\n</html>\n")
    return count
