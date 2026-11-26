r"""Windows Recycle Bin, behind a guard.

The shell may delete a file for good without a word: the bin is full, switched off, or the
file is too large for it. So before the shell sees the file it gets a second name, a hard link
in the quarantine. That guard is released only when the bin demonstrably holds the very same
file; otherwise the file comes back from the guard.

A bin entry is a pair in ``<volume>\$Recycle.Bin\<user>``: ``$Rxxxxxx.ext`` is the file,
``$Ixxxxxx.ext`` a small record with its size, deletion time and original path.
"""

from __future__ import annotations

import ctypes
import ntpath
import os
import stat as stat_module
import struct
import sys
import time
from collections.abc import Callable, Iterator
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from mediasift.actions import checks, fsops
from mediasift.actions.backends import base
from mediasift.actions.backends.base import (
    Located,
    Location,
    PlannedRemoval,
    RemovalRequest,
    RestoredFile,
    RestoreRequest,
    StoredFile,
)
from mediasift.actions.backends.quarantine import (
    make_quarantine_folder,
    quarantine_place,
    restore_from_quarantine,
)
from mediasift.core import paths
from mediasift.core.errors import ActionError, MediaSiftError, SafetyError
from mediasift.core.paths import QUARANTINE_FOLDER, display_path, fs_path
from mediasift.core.types import FileSignature, Root

BIN_FOLDER = "$Recycle.Bin"
RECORDS_FOLDER = ".bin-records"

_ADVICE = "use the quarantine instead"
_MAX_SHELL_PATH = 259
_MAX_RECORD = 70_000
_V1_PATH_BYTES = 520
_EPOCH = datetime(1601, 1, 1, tzinfo=UTC)
_RESERVED = (
    {"con", "prn", "aux", "nul", "conin$", "conout$"}
    | {f"com{n}" for n in "0123456789¹²³"}
    | {f"lpt{n}" for n in "0123456789¹²³"}
)

_DRIVE_FIXED = 3
_FILE_SUPPORTS_HARD_LINKS = 0x00400000


# --- the $I record ------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class BinRecord:
    """Content of a ``$I`` file."""

    version: int
    size: int
    deleted_at: datetime
    original: str


def parse_record(data: bytes) -> BinRecord:
    """Parse a ``$I`` file: version 1 up to Windows 8.1, version 2 since Windows 10."""
    if len(data) < 24:
        raise ValueError("the record is too short")
    version, size, ticks = struct.unpack_from("<qqQ", data)
    if version == 1:
        raw = data[24 : 24 + _V1_PATH_BYTES]
        if len(raw) < _V1_PATH_BYTES:
            raise ValueError("the record is too short")
    elif version == 2:
        raw = _v2_path(data)
    else:
        raise ValueError(f"unknown record version {version}")
    original = raw.decode("utf-16-le", "surrogatepass").split("\0", 1)[0]
    if not original:
        raise ValueError("the record has no path")
    try:
        deleted_at = _EPOCH + timedelta(microseconds=ticks // 10)
    except OverflowError as exc:
        raise ValueError("the record has an impossible date") from exc
    return BinRecord(version, size, deleted_at, original)


def _v2_path(data: bytes) -> bytes:
    if len(data) < 28:
        raise ValueError("the record is too short")
    (length,) = struct.unpack_from("<I", data, 24)
    raw = data[28 : 28 + 2 * length]
    if not length or len(raw) < 2 * length:
        raise ValueError("the record is too short")
    return raw


@dataclass(frozen=True, slots=True)
class BinEntry:
    """A file in the bin. ``content`` is the ``$R`` file, ``index`` the ``$I`` one."""

    content: Path
    index: Path
    record: BinRecord


# --- the system ---------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Volume:
    """What matters about the volume of a file."""

    root: Path
    fixed: bool
    hard_links: bool


def windows_volume(path: Path) -> Volume:
    """Ask Windows about the volume that holds an existing path."""
    if sys.platform != "win32":
        raise SafetyError("there is no Recycle Bin on this system", code="unsupported")
    kernel32 = ctypes.windll.kernel32
    buffer = ctypes.create_unicode_buffer(32768)
    if not kernel32.GetVolumePathNameW(fs_path(path), buffer, len(buffer)):
        raise SafetyError(f"cannot find the volume of {display_path(path)}", code="unsupported")
    mount = buffer.value
    flags = ctypes.c_ulong()
    known = kernel32.GetVolumeInformationW(mount, None, 0, None, None, ctypes.byref(flags), None, 0)
    return Volume(
        root=Path(paths.plain_path(mount)),
        fixed=kernel32.GetDriveTypeW(mount) == _DRIVE_FIXED,
        hard_links=bool(known and flags.value & _FILE_SUPPORTS_HARD_LINKS),
    )


def shell_send(path: str) -> None:
    """Ask the shell to move a file to the Recycle Bin."""
    if sys.platform != "win32":
        raise SafetyError("there is no Recycle Bin on this system", code="unsupported")
    from send2trash import send2trash

    send2trash(path)


def short_name(path: Path) -> str:
    """The 8.3 name of a file, which the shell may record; its own name when it has none."""
    if sys.platform == "win32":
        buffer = ctypes.create_unicode_buffer(32768)
        if ctypes.windll.kernel32.GetShortPathNameW(fs_path(path), buffer, len(buffer)):
            return ntpath.basename(buffer.value)
    return path.name


def is_shell_safe(path: str) -> bool:
    """True when the shell will understand the path literally.

    It rewrites some names that the file system allows (``photo.``, ``nul.jpg``), which would
    make it act on another file, and cannot handle long paths.
    """
    if len(path.encode("utf-16-le", "surrogatepass")) // 2 > _MAX_SHELL_PATH:
        return False
    parts = ntpath.splitdrive(path)[1].replace("/", "\\").split("\\")
    return all(_is_plain_name(part) for part in parts if part)


def _is_plain_name(name: str) -> bool:
    if name.endswith((".", " ")) or set(name) & set('*?"<>|'):
        return False
    return name.split(".", 1)[0].rstrip(" ").casefold() not in _RESERVED


# --- the backend --------------------------------------------------------------------


class RecycleBinBackend:
    """Sends files to the Recycle Bin and proves that they arrived."""

    name = "recycle_bin"

    def __init__(
        self,
        *,
        send: Callable[[str], None] = shell_send,
        volume: Callable[[Path], Volume] = windows_volume,
        short: Callable[[Path], str] = short_name,
        attempts: int = 5,
        delay: float = 0.1,
    ) -> None:
        self._send = send
        self._volume = volume
        self._short = short
        self._attempts = attempts
        self._delay = delay

    def check(self, path: Path, root: Root) -> None:
        base.check_removable(path, root)
        base.check_same_volume(path, root)
        volume = self._volume(path)
        if not volume.fixed:
            raise SafetyError(
                f"{display_path(path)} is on a drive without a Recycle Bin; {_ADVICE}",
                code="unsupported",
            )
        if not volume.hard_links:
            raise SafetyError(
                f"the drive of {display_path(path)} has no hard links, so the Recycle Bin "
                f"cannot be used safely; {_ADVICE}",
                code="unsupported",
            )
        if not is_shell_safe(os.fspath(path)):
            raise SafetyError(
                f"the Recycle Bin cannot handle the name {display_path(path)}; {_ADVICE}",
                code="unsupported",
            )

    def plan_destination(self, path: Path, root: Root, session: str) -> PlannedRemoval:
        """The guard: where the file goes within the bin is the shell's choice."""
        detail = {"short_name": self._short(path)}
        return PlannedRemoval(path, quarantine_place(path, root, session), root, session, detail)

    def remove(self, request: RemovalRequest) -> StoredFile:
        planned = request.planned
        src, guard = planned.src, planned.dst
        self.check(src, planned.root)
        make_quarantine_folder(planned)
        _make_guard(src, guard, request.expected)
        before = set(self._content_names(planned.root))
        cause = self._send_quietly(src)
        entry = self._await_entry(planned, request.expected, before)
        if entry is None:
            raise _take_back(src, guard, cause) from cause
        base.confirm_keeper(request, lambda: self._undo(planned, entry))
        detail = {**planned.detail, "guard": os.fspath(guard)}
        try:
            fsops.release_extra_name(guard, entry.content, request.expected)
        except SafetyError as exc:
            # The entry vanished at the last moment: the guard is all there is.
            raise _take_back(src, guard, exc) from exc
        except ActionError:
            detail["guard_left"] = "yes"
        return StoredFile(entry.content, detail)

    def restore(self, request: RestoreRequest) -> RestoredFile:
        content, guard = request.stored.dst, _guard_of(request.stored)
        if checks.in_quarantine(content):
            return restore_from_quarantine(content, request)
        bin_folder = self._bin(request.root)
        if len(fsops.parts_below(content, bin_folder)) != 2 or not content.name.startswith("$R"):
            raise SafetyError(
                f"{display_path(content)} is not an entry of the Recycle Bin", code="outside_root"
            )
        if fsops.stat_or_none(content) is None and guard is not None:
            return restore_from_quarantine(guard, request)
        restored = base.restore_file(content, request)
        self._put_record_away(content, request.root)
        if guard is not None:
            with suppress(MediaSiftError):
                fsops.release_extra_name(guard, restored.path)
        return restored

    def locate(self, planned: PlannedRemoval, expected: FileSignature) -> Located:
        at_source = base.is_at(planned.src, expected)
        guarded = base.is_at(planned.dst, expected)
        entry = self._newest_entry(planned, expected, set())
        if entry is not None:
            detail = {**planned.detail, "guard": os.fspath(planned.dst)}
            stored = StoredFile(entry.content, detail)
        elif guarded:
            stored = StoredFile(planned.dst, planned.detail)
        else:
            return Located(Location.SOURCE if at_source else Location.NEITHER)
        return Located(Location.BOTH if at_source else Location.DESTINATION, stored)

    def tidy(self, planned: PlannedRemoval, expected: FileSignature) -> bool:
        if base.is_at(planned.dst, expected):
            entry = self._newest_entry(planned, expected, set())
            keep = planned.src if base.is_at(planned.src, expected) else None
            if keep is None and entry is not None:
                keep = entry.content
            if keep is not None:
                fsops.release_extra_name(planned.dst, keep)
        return self.locate(planned, expected).location in (Location.SOURCE, Location.DESTINATION)

    # --- sending ---

    def _send_quietly(self, src: Path) -> Exception | None:
        """Send the file. Errors are returned: what counts is where the file is afterwards."""
        try:
            self._send(os.fspath(src))
        except Exception as exc:  # noqa: BLE001 - the shell's errors have no common base
            return exc
        return None

    def _await_entry(
        self, planned: PlannedRemoval, expected: FileSignature, before: set[Path]
    ) -> BinEntry | None:
        """The new entry. A virus scanner may keep it busy for a moment, so look again."""
        for attempt in range(self._attempts):
            if attempt:
                time.sleep(self._delay)
            if base.is_at(planned.src, expected):
                return None
            entry = self._newest_entry(planned, expected, before)
            if entry is not None:
                return entry
        return None

    def _undo(self, planned: PlannedRemoval, entry: BinEntry) -> None:
        fsops.put_back(entry.content, planned.src)
        self._put_record_away(entry.content, planned.root)
        with suppress(MediaSiftError):
            fsops.release_extra_name(planned.dst, planned.src)

    # --- the bin ---

    def _bin(self, root: Root) -> Path:
        return self._volume(root.path).root / BIN_FOLDER

    def _content_names(self, root: Root) -> Iterator[Path]:
        bin_folder = self._bin(root)
        for user in _list(bin_folder):
            if user.upper().startswith("S-1-") and _is_folder(bin_folder / user):
                for name in _list(bin_folder / user):
                    if name.upper().startswith("$R"):
                        yield bin_folder / user / name

    def _newest_entry(
        self, planned: PlannedRemoval, expected: FileSignature, before: set[Path]
    ) -> BinEntry | None:
        """The entry that holds the very file and says it came from the planned source."""
        found: list[BinEntry] = []
        for content in self._content_names(planned.root):
            if content in before or not _holds(content, expected):
                continue
            entry = _read_entry(content)
            if entry is not None and _came_from(entry.record.original, planned):
                found.append(entry)
        return max(found, key=lambda entry: entry.record.deleted_at, default=None)

    def _put_record_away(self, content: Path, root: Root) -> None:
        """After a restore the ``$I`` record describes nothing.

        It is bookkeeping of the bin, not the user's data, and Explorer would delete it. Here
        it is moved into the quarantine instead, so that no code of MediaSift ever deletes
        anything in the bin. Explorer ignores a record without a file, so failing is harmless.
        """
        folder = root.path / QUARANTINE_FOLDER / RECORDS_FOLDER
        with suppress(MediaSiftError):
            index = _index_of(content)
            current = fsops.regular_signature(index)
            fsops.make_dirs_private(folder, root.path)
            for name in (index.name, *fsops.numbered(index.name)):
                if fsops.stat_or_none(folder / name) is None:
                    fsops.move_no_replace(index, folder / name, current)
                    return


def _make_guard(src: Path, guard: Path, expected: FileSignature) -> None:
    try:
        os.link(fs_path(src), fs_path(guard))
    except FileExistsError as exc:
        raise SafetyError(f"{display_path(guard)} already exists", code="target_exists") from exc
    except FileNotFoundError as exc:
        raise SafetyError(f"{display_path(src)} does not exist", code="missing") from exc
    except OSError as exc:
        raise SafetyError(
            f"cannot protect {display_path(src)} with a hard link ({exc.strerror or exc}), so "
            f"the Recycle Bin cannot be used safely; {_ADVICE}",
            code="unsupported",
        ) from exc
    try:
        fsops.expect_unchanged(guard, expected)
        fsops.expect_unchanged(src, expected)
    except SafetyError:
        _drop_guard(src, guard)
        raise


def _drop_guard(src: Path, guard: Path) -> None:
    try:
        fsops.release_extra_name(guard, src)
    except MediaSiftError as exc:
        raise fsops.MisplacedFile(
            f"{display_path(src)} was replaced; an earlier file of that name is at "
            f"{display_path(guard)}",
            guard,
        ) from exc


def _take_back(src: Path, guard: Path, cause: Exception | None) -> ActionError:
    """The bin does not hold the file. Make sure it is at its place; return what to raise."""
    reason = f" ({cause})" if cause else ""
    if _one_file(src, guard):
        with suppress(ActionError):
            fsops.release_extra_name(guard, src)
        return ActionError(
            f"the Recycle Bin did not take {display_path(src)}{reason}; it is untouched"
        )
    fsops.put_back(guard, src)
    return ActionError(
        f"{display_path(src)} did not arrive in the Recycle Bin{reason}; it was put back"
    )


def _one_file(first: Path, second: Path) -> bool:
    a, b = fsops.stat_or_none(first), fsops.stat_or_none(second)
    return a is not None and b is not None and paths.signature(a).same_file(paths.signature(b))


def _guard_of(stored: StoredFile) -> Path | None:
    guard = stored.detail.get("guard")
    if guard is None or fsops.stat_or_none(Path(guard)) is None:
        return None
    return Path(guard)


def _list(folder: Path) -> list[str]:
    try:
        return os.listdir(fs_path(folder))
    except OSError:
        return []  # another user's folder


def _is_folder(path: Path) -> bool:
    try:
        st = os.lstat(fs_path(path))
    except OSError:
        return False
    return stat_module.S_ISDIR(st.st_mode) and not paths.is_link_like(st)


def _holds(content: Path, expected: FileSignature) -> bool:
    try:
        st = os.lstat(fs_path(content))
    except OSError:
        return False
    regular = stat_module.S_ISREG(st.st_mode) and not paths.is_link_like(st)
    return regular and base.recognised(paths.signature(st), expected)


def _index_of(content: Path) -> Path:
    return content.with_name("$I" + content.name[2:])


def _read_entry(content: Path) -> BinEntry | None:
    index = _index_of(content)
    try:
        with open(fs_path(index), "rb") as file:
            record = parse_record(file.read(_MAX_RECORD))
    except (OSError, ValueError):
        return None
    return BinEntry(content, index, record)


def _came_from(original: str, planned: PlannedRemoval) -> bool:
    """Compare as Windows does. The record may hold short (8.3) names.

    Folders still exist, so they are compared by identity. The file itself is gone from there,
    which is why its short name is looked up before it is sent.
    """
    original = original.removeprefix("\\\\?\\")
    folder, name = os.path.split(original)
    names = {planned.src.name, planned.detail.get("short_name", planned.src.name)}
    if name.casefold() not in {known.casefold() for known in names}:
        return False
    return paths.same_location(folder, planned.src.parent)
