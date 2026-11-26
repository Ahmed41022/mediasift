"""The checks a file must pass right before it is moved.

Call ``check_item`` as late as possible: after the operation is planned, immediately before it
is logged and carried out. It trusts nothing that was recorded earlier; content is hashed
again from disk.

What can still happen between the last check and the rename, and what becomes of it:

* Another file takes the name: ``fsops.move_no_replace`` compares identity, size and modified
  time before the move and again after it, and moves a stranger back.
* The file is written to: size or modified time differ after the move, same handling. A
  program that writes in place and restores the modified time within that moment goes
  unnoticed, but its data is in the stored file, and restoring reports that it changed.
* The keeper disappears or changes: backends call ``expect_still_there`` after the move, and
  move the file back when the keeper is no longer what was checked. That also catches one file
  reached through two mounts whose identities look different.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from mediasift.actions import fsops
from mediasift.core import paths
from mediasift.core.errors import ActionError, FileChanged, SafetyError
from mediasift.core.paths import QUARANTINE_FOLDER, display_path, fs_path
from mediasift.core.types import FileRecord, FileSignature, MatchKind, Root

Cancel = Callable[[], bool]

_READ_ONLY_ATTRIBUTE = 0x1


@dataclass(frozen=True, slots=True)
class KnownFile:
    """A file as the library recorded it."""

    record: FileRecord
    full_hash: bytes


@dataclass(frozen=True, slots=True)
class VerifiedFile:
    """A file that was just found on disk, unchanged."""

    path: Path
    signature: FileSignature
    full_hash: bytes


@dataclass(frozen=True, slots=True)
class CheckedItem:
    """A file that may be removed now, and the keeper that stays instead."""

    file: VerifiedFile
    root: Root
    keeper: VerifiedFile


def check_item(
    item: KnownFile,
    keepers: Sequence[KnownFile],
    roots: Iterable[Root],
    *,
    kind: MatchKind = MatchKind.EXACT,
    cancel: Cancel | None = None,
) -> CheckedItem:
    """Verify that ``item`` may be removed. Raises ``SafetyError``, changes nothing.

    Codes: ``outside_root``, ``root_disabled``, ``root_read_only``, ``file_read_only``,
    ``is_link``, ``is_placeholder``, ``missing``, ``changed``, ``content_changed``,
    ``keeper_missing``, ``keeper_changed``, ``keeper_same_file``, ``not_regular``,
    ``in_quarantine``, ``unsupported``.

    One intact keeper is enough, but a keeper that is the same file as the item refuses it,
    whatever the others are.
    """
    by_id = {root.id: root for root in roots}
    root, current = check_location(item.record, by_id)
    _expect_recorded(item.record, current)
    _refuse_same_file(item.record.path, current, keepers)
    keeper = _find_keeper(item, keepers, by_id, kind, cancel)
    verified = _verify_content(item, current, cancel)
    # Hashing may have taken minutes; these two take microseconds.
    expect_still_there(keeper)
    if not fsops.unchanged(check_location(item.record, by_id)[1], verified.signature):
        raise SafetyError(f"{display_path(item.record.path)} has changed", code="changed")
    return CheckedItem(verified, root, keeper)


def check_location(record: FileRecord, roots: dict[int, Root]) -> tuple[Root, FileSignature]:
    """Everything that can be told without reading the content."""
    path = record.path
    root = _root_of(record, roots)
    if not root.enabled:
        raise SafetyError(f"{display_path(root.path)} is disabled", code="root_disabled")
    if root.read_only:
        raise SafetyError(f"{display_path(root.path)} is read-only", code="root_read_only")
    expect_inside(path, root)
    if in_quarantine(path):
        raise SafetyError(f"{display_path(path)} is already removed", code="in_quarantine")
    current = fsops.regular_signature(path)
    refuse_read_only(path)
    return root, current


def expect_inside(path: Path, root: Root) -> None:
    """The path must lie in the root, as written and in reality, with no link on the way."""
    fsops.parts_below(path, root.path)
    if not paths.is_within(path, root.path) and not _is_link(path):
        raise SafetyError(
            f"{display_path(path)} is not inside {display_path(root.path)}", code="outside_root"
        )
    fsops.refuse_links(path, root.path)


def in_quarantine(path: Path) -> bool:
    """True below any quarantine folder, whatever the spelling of its name."""
    return any(part.casefold() == QUARANTINE_FOLDER for part in path.parts)


def refuse_read_only(path: Path) -> None:
    """Refuse protected files, and files whose folder may not be written to."""
    st = fsops.stat_or_none(path)
    if st is None:
        raise SafetyError(f"{display_path(path)} does not exist", code="missing")
    if is_read_only(st):
        raise SafetyError(f"{display_path(path)} is read-only", code="file_read_only")
    # On Windows the attribute of a folder means something else; its rights show on renaming.
    if sys.platform != "win32" and not os.access(path.parent, os.W_OK | os.X_OK):
        raise SafetyError(
            f"the folder of {display_path(path)} may not be written to", code="file_read_only"
        )


def is_read_only(st: os.stat_result) -> bool:
    """Read-only attribute on Windows, no write permission for anyone elsewhere."""
    attributes: int = getattr(st, "st_file_attributes", 0)
    return bool(attributes & _READ_ONLY_ATTRIBUTE) or not st.st_mode & 0o222


def expect_still_there(file: VerifiedFile) -> None:
    """Refuse when a verified keeper is gone or changed.

    Opens it, because network file systems answer ``stat`` from a cache.
    """
    shown = display_path(file.path)
    try:
        current = fsops.regular_signature(file.path)
        with os.fdopen(fsops.open_for_reading(file.path), "rb", buffering=0) as handle:
            handle.read(1)
            opened = paths.signature(os.fstat(handle.fileno()))
    except SafetyError as exc:
        code = "keeper_missing" if exc.code == "missing" else "keeper_changed"
        raise SafetyError(f"the file to keep, {shown}: {exc}", code=code) from exc
    except FileNotFoundError as exc:
        raise SafetyError(f"the file to keep, {shown}, is gone", code="keeper_missing") from exc
    except (OSError, ActionError) as exc:
        raise SafetyError(
            f"the file to keep, {shown}, cannot be read: {exc}", code="keeper_changed"
        ) from exc
    if not (fsops.unchanged(current, file.signature) and fsops.unchanged(opened, file.signature)):
        raise SafetyError(f"the file to keep, {shown}, has changed", code="keeper_changed")


# --- the item -----------------------------------------------------------------------


def _root_of(record: FileRecord, roots: dict[int, Root]) -> Root:
    root = roots.get(record.root_id)
    if root is None:
        raise SafetyError(
            f"{display_path(record.path)} is not in a folder of the library", code="outside_root"
        )
    return root


def _is_link(path: Path) -> bool:
    """A link is reported as a link, wherever it leads."""
    st = fsops.stat_or_none(path)
    return st is not None and paths.is_link_like(st)


def _expect_recorded(record: FileRecord, current: FileSignature) -> None:
    recorded = record.signature
    same_identity = not recorded.inode or current.same_file(recorded)
    if not (same_identity and current.same_content_state(recorded)):
        raise SafetyError(
            f"{display_path(record.path)} has changed since it was scanned", code="changed"
        )


def _verify_content(item: KnownFile, before: FileSignature, cancel: Cancel | None) -> VerifiedFile:
    path = item.record.path
    digest, current = _hash(path, cancel)
    if not fsops.unchanged(current, before):
        raise SafetyError(f"{display_path(path)} has changed", code="changed")
    if digest != item.full_hash:
        raise SafetyError(
            f"the content of {display_path(path)} has changed since it was scanned",
            code="content_changed",
        )
    return VerifiedFile(path, current, digest)


def _hash(path: Path, cancel: Cancel | None) -> tuple[bytes, FileSignature]:
    try:
        return fsops.hash_with_signature(path, cancel)
    except FileChanged as exc:
        raise SafetyError(str(exc), code="changed") from exc
    except ActionError as exc:
        raise SafetyError(str(exc), code="unsupported") from exc


# --- keepers ------------------------------------------------------------------------


def _refuse_same_file(path: Path, current: FileSignature, keepers: Sequence[KnownFile]) -> None:
    for keeper in keepers:
        other = keeper.record.path
        st = _follow(other)
        same = st is not None and paths.signature(st).same_file(current)
        if same or paths.same_location(path, other):
            raise SafetyError(
                f"{display_path(path)} and the file to keep, {display_path(other)}, are the "
                "same file",
                code="keeper_same_file",
            )


def _follow(path: Path) -> os.stat_result | None:
    """What a keeper's path really leads to, so that a link to the item is recognised."""
    try:
        return os.stat(fs_path(path))
    except (OSError, ValueError):
        return None


def _find_keeper(
    item: KnownFile,
    keepers: Sequence[KnownFile],
    roots: dict[int, Root],
    kind: MatchKind,
    cancel: Cancel | None,
) -> VerifiedFile:
    """The first intact keeper. When there is none, say what is wrong with the first."""
    problem: SafetyError | None = None
    for keeper in keepers:
        try:
            return _verify_keeper(item, keeper, roots, kind, cancel)
        except SafetyError as exc:
            problem = problem or exc
    if problem is None:
        raise SafetyError(
            f"no file is kept instead of {display_path(item.record.path)}", code="keeper_missing"
        )
    raise problem


def _verify_keeper(
    item: KnownFile,
    keeper: KnownFile,
    roots: dict[int, Root],
    kind: MatchKind,
    cancel: Cancel | None,
) -> VerifiedFile:
    path = keeper.record.path
    shown = display_path(path)
    _expect_keeper_place(keeper.record, roots)
    if kind is MatchKind.EXACT and keeper.full_hash != item.full_hash:
        raise SafetyError(
            f"the file to keep, {shown}, is not a copy of {display_path(item.record.path)}",
            code="keeper_changed",
        )
    try:
        current = fsops.regular_signature(path)
        _expect_recorded(keeper.record, current)
        verified = _verify_content(keeper, current, cancel)
    except SafetyError as exc:
        code = "keeper_missing" if exc.code == "missing" else "keeper_changed"
        raise SafetyError(f"the file to keep: {exc}", code=code) from exc
    except ActionError as exc:
        raise SafetyError(f"the file to keep: {exc}", code="keeper_changed") from exc
    return verified


def _expect_keeper_place(record: FileRecord, roots: dict[int, Root]) -> None:
    """A keeper counts only where MediaSift can vouch for it: in an enabled folder."""
    root = roots.get(record.root_id)
    path = record.path
    if root is None or not root.enabled or in_quarantine(path):
        raise SafetyError(
            f"the file to keep, {display_path(path)}, is not in an enabled folder",
            code="keeper_missing",
        )
    try:
        fsops.parts_below(path, root.path)
        fsops.refuse_links(path, root.path)
    except SafetyError as exc:
        code = "keeper_missing" if exc.code == "missing" else "keeper_changed"
        raise SafetyError(f"the file to keep: {exc}", code=code) from exc
    except ActionError as exc:
        raise SafetyError(f"the file to keep: {exc}", code="keeper_missing") from exc
