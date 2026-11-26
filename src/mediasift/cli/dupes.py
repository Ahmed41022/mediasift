"""``mediasift dupes``."""

from __future__ import annotations

from typing import Any

import click

from mediasift.api import ExactSummary, ExitCode, GroupInfo, JobCancelled, Library, human_size
from mediasift.cli.common import (
    Leaf,
    emit_json,
    env_of,
    finish,
    iso,
    open_library,
    out,
    parse_size,
    say,
    show,
)
from mediasift.cli.progress import job_context

DEFAULT_LIMIT = 20
PAGE = 200


def pick_groups(library: Library, limit: int | None, min_size: int) -> tuple[list[GroupInfo], bool]:
    """The first ``limit`` groups whose files are at least ``min_size``, and whether more follow."""
    picked: list[GroupInfo] = []
    after = None
    while True:
        page = library.list_groups(limit=PAGE, after=after)
        for group in page.groups:
            if group.size < min_size:
                continue
            if limit is not None and len(picked) == limit:
                return picked, True
            picked.append(group)
        if page.next is None:
            return picked, False
        after = page.next


def group_json(number: int, group: GroupInfo) -> dict[str, Any]:
    return {
        "number": number,
        "id": group.id,
        "size": group.size,
        "reclaimable_bytes": group.reclaimable_bytes,
        "files": [
            {
                "id": m.file_id,
                "path": show(m.path),
                "size": m.size,
                "modified": iso(m.mtime),
                "media_type": m.media_type,
                "hardlink_of": m.hardlink_of,
            }
            for m in group.members
        ],
    }


def print_group(number: int, group: GroupInfo) -> None:
    out(
        f"Group {number}: {len(group.members)} files of {human_size(group.size)}, "
        f"{human_size(group.reclaimable_bytes)} can be freed"
    )
    by_id = {m.file_id: m for m in group.members}
    for member in group.members:
        owner = by_id.get(member.hardlink_of) if member.hardlink_of is not None else None
        mark = f"  [hard link of {show(owner.path)}: frees no space]" if owner else ""
        out(f"  {show(member.path)}{mark}")


def exact_json(exact: ExactSummary) -> dict[str, Any]:
    return {
        "groups": exact.groups,
        "reclaimable_bytes": exact.reclaimable_bytes,
        "hardlink_sets": exact.hardlink_sets,
        "empty_files": exact.empty_files,
        "hashed_samples": exact.hashed_samples,
        "hashed_full": exact.hashed_full,
        "issue_count": exact.issue_count,
        "changed_files": exact.changed_files,
        "issues": [
            {"path": show(i.path), "kind": i.kind, "message": i.message} for i in exact.issues
        ],
    }


def _cancelled() -> None:
    if env_of().json:
        emit_json("dupes", {"cancelled": True})
    say("Cancelled. The hashes made so far are kept; run again to continue.")
    finish(int(ExitCode.CANCELLED))


@click.command(cls=Leaf)
@click.option("--limit", type=click.IntRange(min=1), default=DEFAULT_LIMIT, show_default=True)
@click.option("--all", "show_all", is_flag=True, help="List every group, not only the first ones.")
@click.option(
    "--min-size", default="0", callback=parse_size, help="Skip groups of files smaller than this."
)
def dupes(limit: int, show_all: bool, min_size: int) -> None:
    """Find identical files and list the groups that free the most space first.

    Files are compared by content, whatever their names. Hard links to one file are marked:
    they are not copies and free nothing. Nothing is changed on disk.
    """
    with open_library() as library:
        try:
            with job_context() as job:
                exact = library.find_exact(job)
        except JobCancelled:
            _cancelled()
            return
        groups, more = pick_groups(library, None if show_all else limit, min_size)
    if env_of().json:
        emit_json(
            "dupes",
            {
                "exact": exact_json(exact),
                "shown": len(groups),
                "more": more,
                "groups": [group_json(n, g) for n, g in enumerate(groups, start=1)],
            },
        )
    else:
        out(
            f"{exact.groups} groups of identical files; {human_size(exact.reclaimable_bytes)} can "
            f"be freed. Hashed {exact.hashed_samples} samples and {exact.hashed_full} whole files."
        )
        for number, group in enumerate(groups, start=1):
            print_group(number, group)
        if more:
            say(f"More groups exist; use --all to list them (showing {len(groups)}).")
        if exact.has_issues:
            say(f"{exact.issue_count} file(s) could not be hashed, {exact.changed_files} changed.")
    finish(int(ExitCode.PARTIAL) if exact.has_issues else 0)
