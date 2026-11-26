"""``mediasift scan``."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import click

from mediasift.api import ExitCode, IssueInfo, Library, ScanSummary
from mediasift.cli.common import Leaf, emit_json, env_of, finish, open_library, out, say, show
from mediasift.cli.progress import job_context

ISSUES_SHOWN = 10
ISSUES_IN_JSON = 1000


def resolve_targets(library: Library, targets: tuple[str, ...]) -> list[int] | None:
    """Ids for the arguments. A number is a folder id; anything else is a folder to add.

    A folder that is not in the library yet is added, so ``mediasift scan PATH`` works as
    the first command. To name a folder called ``5``, write ``./5``.
    """
    if not targets:
        return None
    ids: list[int] = []
    for target in targets:
        if target.isdigit():
            ids.append(int(target))
            continue
        root = library.find_root(Path(target))
        if root is None:
            root = library.add_root(Path(target))
            say(f"Added folder {root.id}: {show(root.path)}")
        ids.append(root.id)
    return ids


def scan_json(summary: ScanSummary, issues: tuple[IssueInfo, ...]) -> dict[str, Any]:
    return {
        "scan_id": summary.scan_id,
        "state": summary.state,
        "files_seen": summary.seen,
        "new": summary.new,
        "changed": summary.changed,
        "unchanged": summary.unchanged,
        "moved": summary.moved,
        "missing": summary.missing,
        "restored": summary.restored,
        "placeholders": summary.placeholders,
        "skipped_links": summary.skipped_links,
        "vanished": summary.vanished,
        "errors": summary.errors,
        "by_type": dict(summary.by_type),
        "unreachable_roots": list(summary.unreachable_roots),
        "issues": [{"path": show(i.path), "kind": i.kind, "message": i.message} for i in issues],
    }


def _print_issues(summary: ScanSummary, issues: tuple[IssueInfo, ...]) -> None:
    say(f"{summary.errors} problem(s) while scanning:")
    for issue in issues[:ISSUES_SHOWN]:
        say(f"  {issue.kind}: {show(issue.path)}  {issue.message}".rstrip())
    if summary.errors > ISSUES_SHOWN:
        say(f"  ... and {summary.errors - ISSUES_SHOWN} more (see the log in the library folder)")


def exit_code(summary: ScanSummary) -> ExitCode:
    if summary.cancelled:
        return ExitCode.CANCELLED
    return ExitCode.PARTIAL if summary.has_issues else ExitCode.OK


@click.command(cls=Leaf)
@click.argument("targets", nargs=-1)
def scan(targets: tuple[str, ...]) -> None:
    """Scan folders: ROOT_ID numbers, or folders (added when new). Without arguments, all.

    Only folder listings are read, so a rescan of an unchanged library is quick.
    """
    with open_library() as library:
        ids = resolve_targets(library, targets)
        if ids is None and not any(r.enabled for r in library.list_roots()):
            raise click.UsageError("there are no folders to scan; add one with: roots add PATH")
        with job_context() as job:
            summary = library.scan(ids, job)
        issues = library.issues(summary.scan_id, ISSUES_IN_JSON) if summary.errors else ()
    if env_of().json:
        emit_json("scan", scan_json(summary, issues))
    else:
        if summary.cancelled:
            say("Cancelled. What was indexed so far is kept; scan again to continue.")
        out(
            f"{summary.seen} files seen: {summary.new} new, {summary.changed} changed, "
            f"{summary.unchanged} unchanged, {summary.moved} moved, {summary.missing} missing."
        )
        if summary.errors or summary.unreachable_roots:
            _print_issues(summary, issues)
    finish(int(exit_code(summary)))
