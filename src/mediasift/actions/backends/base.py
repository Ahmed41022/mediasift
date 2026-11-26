"""What every removal backend offers, and the parts they share.

The order of events, for the code that keeps the operation log:

1. ``checks.check_item``
2. ``plan_destination``: where the file will go, without touching anything
3. write the planned operation to the log
4. ``remove``; on success write the stored place to the log
5. after a crash between 3 and 4: ``locate``, and ``tidy`` when it finds two names

A destination is planned for one file at a time, right before it moves. If it is taken in the
meantime, ``remove`` refuses with ``target_exists`` and nothing has moved: plan again.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Protocol

from mediasift.actions import checks, fsops
from mediasift.actions.checks import Cancel, VerifiedFile
from mediasift.core import paths
from mediasift.core.errors import FileChanged, MediaSiftError, SafetyError
from mediasift.core.paths import QUARANTINE_FOLDER, display_path
from mediasift.core.types import FileSignature, Root

RESTORED = "restored"

# Starts with a letter or digit, so it cannot clash with folders of MediaSift itself.
_SESSION = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}")


class Location(StrEnum):
    """Where the file of an operation is, whatever the log says."""

    SOURCE = "source"
    DESTINATION = "destination"
    BOTH = "both"
    NEITHER = "neither"


@dataclass(frozen=True, slots=True)
class PlannedRemoval:
    """Where a file will go. ``detail`` holds plain text only, so it can be logged as JSON."""

    src: Path
    dst: Path
    root: Root
    session: str
    detail: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RemovalRequest:
    """A planned removal with what was verified right before it."""

    planned: PlannedRemoval
    expected: FileSignature
    keeper: VerifiedFile


@dataclass(frozen=True, slots=True)
class StoredFile:
    """Where a removed file is now."""

    dst: Path
    detail: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RestoreRequest:
    """A stored file to bring back. ``expected`` is what was recorded when it was removed."""

    stored: StoredFile
    original: Path
    root: Root
    expected: FileSignature
    full_hash: bytes | None = None
    cancel: Cancel | None = None


@dataclass(frozen=True, slots=True)
class RestoredFile:
    """Where a file was restored. ``renamed`` when its own name was taken."""

    path: Path
    renamed: bool


@dataclass(frozen=True, slots=True)
class Located:
    """Answer of ``locate``. ``stored`` is set when the destination has the file."""

    location: Location
    stored: StoredFile | None = None


class RemovalBackend(Protocol):
    """A way to remove files that can be undone."""

    @property
    def name(self) -> str: ...

    def check(self, path: Path, root: Root) -> None:
        """Raise ``SafetyError`` when this backend cannot take the file. Changes nothing."""
        ...

    def plan_destination(self, path: Path, root: Root, session: str) -> PlannedRemoval:
        """Choose where the file will go. Changes nothing."""
        ...

    def remove(self, request: RemovalRequest) -> StoredFile:
        """Move the file away.

        ``SafetyError`` and ``ActionError`` mean that the file is where it was; only
        ``fsops.MisplacedFile`` means that it is elsewhere, and says where.
        """
        ...

    def restore(self, request: RestoreRequest) -> RestoredFile:
        """Bring a file back without overwriting anything.

        ``SafetyError`` codes ``stored_missing`` and ``stored_changed`` mean that what was
        stored is not there, or not the same, any more; then nothing was moved.
        """
        ...

    def locate(self, planned: PlannedRemoval, expected: FileSignature) -> Located:
        """Find the file of an operation that may have been interrupted. Changes nothing."""
        ...

    def tidy(self, planned: PlannedRemoval, expected: FileSignature) -> bool:
        """Drop leftovers of an interrupted removal that are provably redundant.

        True when the file then has exactly one place. False needs a person's attention.
        """
        ...


# --- removing -----------------------------------------------------------------------


def session_folder(root: Root, session: str) -> Path:
    """Where the quarantine keeps the files of one session."""
    if not _SESSION.fullmatch(session):
        raise SafetyError(f"{session!r} is not a valid session name", code="unsupported")
    return root.path / QUARANTINE_FOLDER / session


def check_removable(path: Path, root: Root) -> None:
    """Checks that hold for every backend."""
    if not root.enabled:
        raise SafetyError(f"{display_path(root.path)} is disabled", code="root_disabled")
    if root.read_only:
        raise SafetyError(f"{display_path(root.path)} is read-only", code="root_read_only")
    checks.expect_inside(path, root)
    if checks.in_quarantine(path):
        raise SafetyError(f"{display_path(path)} is already removed", code="in_quarantine")
    fsops.regular_signature(path)
    checks.refuse_read_only(path)


def check_same_volume(path: Path, root: Root) -> None:
    """Refuse files of another volume mounted inside the root."""
    file, folder = fsops.stat_or_none(path), fsops.stat_or_none(root.path)
    if file is None or folder is None or file.st_dev != folder.st_dev:
        raise SafetyError(
            f"{display_path(path)} is on another volume than {display_path(root.path)}",
            code="cross_volume",
        )


def confirm_keeper(request: RemovalRequest, undo: Callable[[], None]) -> None:
    """After a move: if the keeper is not what was checked any more, move back."""
    try:
        checks.expect_still_there(request.keeper)
    except SafetyError:
        undo()
        raise


# --- restoring ----------------------------------------------------------------------


def restore_file(source: Path, request: RestoreRequest) -> RestoredFile:
    """Verify a stored file and move it to its original place, or beside it."""
    original, root = request.original, request.root
    check_restorable(original, root)
    current = verify_stored(source, request)
    if original.parent != root.path:
        fsops.make_dirs_private(original.parent, root.path, mode=0o777)
    target = _move_to_free(source, original, current)
    _verify_restored(target, request)
    return RestoredFile(target, renamed=target != original)


def check_restorable(original: Path, root: Root) -> None:
    """Files come back into their root only, and only while it may be changed."""
    if not root.enabled:
        raise SafetyError(f"{display_path(root.path)} is disabled", code="root_disabled")
    if root.read_only:
        raise SafetyError(f"{display_path(root.path)} is read-only", code="root_read_only")
    fsops.parts_below(original, root.path)
    parent = original.parent
    if parent != root.path and not paths.is_within(parent, root.path):
        raise SafetyError(
            f"{display_path(original)} is not inside {display_path(root.path)}",
            code="outside_root",
        )
    if checks.in_quarantine(original):
        raise SafetyError(f"{display_path(original)} is in the quarantine", code="in_quarantine")


def verify_stored(source: Path, request: RestoreRequest) -> FileSignature:
    """The stored file's current signature, if it still is what was removed.

    With a recorded hash the content decides; otherwise it has to be the very same file.
    """
    shown = display_path(source)
    try:
        current = fsops.regular_signature(source)
        same = current.same_content_state(request.expected)
        if request.full_hash is None:
            same = same and recognised(current, request.expected)
        else:
            digest, current = fsops.hash_with_signature(source, request.cancel)
            same = same and digest == request.full_hash
    except FileChanged as exc:
        raise SafetyError(str(exc), code="stored_changed") from exc
    except SafetyError as exc:
        code = "stored_missing" if exc.code == "missing" else "stored_changed"
        raise SafetyError(str(exc), code=code) from exc
    if not same:
        raise SafetyError(f"{shown} has changed since it was removed", code="stored_changed")
    return current


def _move_to_free(source: Path, original: Path, current: FileSignature) -> Path:
    names = [original.name, *_restored_names(original.name)]
    for name in names:
        target = original.with_name(name)
        try:
            fsops.move_no_replace(source, target, current)
        except SafetyError as exc:
            if exc.code != "target_exists":
                raise
        else:
            return target
    raise SafetyError(f"there is no free name for {display_path(original)}", code="target_exists")


def _restored_names(name: str) -> list[str]:
    return list(fsops.numbered(name, RESTORED))


def _verify_restored(target: Path, request: RestoreRequest) -> None:
    if request.full_hash is None:
        return
    try:
        same = fsops.hash_file(target) == request.full_hash
    except MediaSiftError as exc:
        raise fsops.MisplacedFile(
            f"{display_path(target)} was restored, but could not be verified: {exc}", target
        ) from exc
    if not same:
        raise fsops.MisplacedFile(
            f"{display_path(target)} was restored, but its content changed meanwhile", target
        )


# --- recovery -----------------------------------------------------------------------


def recognised(current: FileSignature, expected: FileSignature) -> bool:
    """True for the file of an earlier session: same file number, size and modified time.

    The device number is left out because it may change when Linux restarts. Size and time
    are not, because file systems reuse the number of a deleted file at once.
    """
    return (
        bool(current.inode)
        and current.inode == expected.inode
        and (current.same_content_state(expected))
    )


def is_at(path: Path, expected: FileSignature) -> bool:
    """True when the expected file has this name."""
    st = fsops.stat_or_none(path)
    return st is not None and recognised(paths.signature(st), expected)


def locate_by_name(planned: PlannedRemoval, expected: FileSignature) -> Located:
    """``locate`` for backends that know the destination beforehand."""
    at_source = is_at(planned.src, expected)
    if not is_at(planned.dst, expected):
        return Located(Location.SOURCE if at_source else Location.NEITHER)
    stored = StoredFile(planned.dst, planned.detail)
    return Located(Location.BOTH if at_source else Location.DESTINATION, stored)
