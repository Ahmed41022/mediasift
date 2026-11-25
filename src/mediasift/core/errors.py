"""Exception tree and process exit codes."""

from __future__ import annotations

from enum import IntEnum


class ExitCode(IntEnum):
    """Exit codes of the command line (docs/PLAN.md, 1.7)."""

    OK = 0
    FAILURE = 1
    USAGE = 2
    PARTIAL = 3
    REFUSED = 4
    BUSY = 5
    CANCELLED = 130


class MediaSiftError(Exception):
    """Base of every expected error. Interfaces show the message, not a traceback."""

    exit_code: ExitCode = ExitCode.FAILURE


class ConfigError(MediaSiftError):
    """A setting or the config file is invalid."""

    exit_code = ExitCode.USAGE


class LibraryError(MediaSiftError):
    """The library index cannot be used."""


class LibraryBusy(LibraryError):
    """Another process is writing to the library."""

    exit_code = ExitCode.BUSY


class SchemaTooNew(LibraryError):
    """The library was written by a newer version."""

    exit_code = ExitCode.BUSY


class MigrationError(LibraryError):
    """A migration failed, or an applied migration was altered."""

    exit_code = ExitCode.BUSY


class ReaderError(MediaSiftError):
    """A media file cannot be read. ``kind`` is a short stable code."""

    def __init__(self, message: str, kind: str = "unreadable") -> None:
        super().__init__(message)
        self.kind = kind


class FileChanged(MediaSiftError):
    """A file changed while it was being read, or since it was scanned."""


class SafetyError(MediaSiftError):
    """A safety check refused an action. ``code`` is a short stable code."""

    exit_code = ExitCode.REFUSED

    def __init__(self, message: str, code: str = "refused") -> None:
        super().__init__(message)
        self.code = code


class ActionError(MediaSiftError):
    """An action on files failed."""


class JobCancelled(MediaSiftError):
    """The user cancelled a job."""

    exit_code = ExitCode.CANCELLED
