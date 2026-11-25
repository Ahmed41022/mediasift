"""Types shared by every layer. No I/O here."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any


class MediaType(StrEnum):
    """Kind of media, decided by extension."""

    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"


class FileStatus(StrEnum):
    """State of an indexed file."""

    PRESENT = "present"
    MISSING = "missing"
    ERROR = "error"
    PLACEHOLDER = "placeholder"


class MatchKind(StrEnum):
    """What kind of similarity a group expresses."""

    EXACT = "exact"
    SIMILAR_IMAGE = "similar_image"
    BURST = "burst"


class Choice(StrEnum):
    """Decision about one file of a group."""

    KEEP = "keep"
    REMOVE = "remove"
    UNDECIDED = "undecided"


class DecisionSource(StrEnum):
    """Who made a decision. A user decision always wins."""

    AUTO = "auto"
    USER = "user"


IMAGE_EXTENSIONS = frozenset(
    {".jpg", ".jpeg", ".jpe", ".jfif", ".png", ".webp", ".gif", ".bmp", ".tif", ".tiff"}
    | {".heic", ".heif", ".hif"}
    | {".dng", ".cr2", ".cr3", ".nef", ".arw", ".orf", ".rw2", ".raf", ".pef", ".srw"}
)
VIDEO_EXTENSIONS = frozenset({".mp4", ".m4v", ".mov", ".mkv", ".avi", ".webm", ".3gp", ".3g2"})
AUDIO_EXTENSIONS = frozenset(
    {".mp3", ".flac", ".m4a", ".aac", ".ogg", ".oga", ".opus", ".wav", ".wma", ".amr"}
)


def media_type_for(name: str) -> MediaType | None:
    """Return the media type for a file name, or None when it is not media."""
    dot = name.rfind(".")
    ext = name[dot:].lower() if dot > 0 else ""
    if ext in IMAGE_EXTENSIONS:
        return MediaType.IMAGE
    if ext in VIDEO_EXTENSIONS:
        return MediaType.VIDEO
    if ext in AUDIO_EXTENSIONS:
        return MediaType.AUDIO
    return None


@dataclass(frozen=True, slots=True)
class FileSignature:
    """Cheap fingerprint of a file's state on disk.

    ``dev`` and ``inode`` are 0 when the file system has no stable identity.
    """

    size: int
    mtime_ns: int
    dev: int = 0
    inode: int = 0
    nlink: int = 1

    def same_content_state(self, other: FileSignature) -> bool:
        """True when size and modified time agree (identity is not compared)."""
        return self.size == other.size and self.mtime_ns == other.mtime_ns

    def same_file(self, other: FileSignature) -> bool:
        """True when both name the same file on disk. Unknown identity is never the same."""
        return self.inode != 0 and (self.dev, self.inode) == (other.dev, other.inode)


@dataclass(frozen=True, slots=True)
class Root:
    """A folder the user chose. Nothing outside enabled roots is ever touched."""

    id: int
    path: Path
    case_sensitive: bool
    read_only: bool = False
    prefer_rank: int = 0
    enabled: bool = True
    volume_id: str = ""


@dataclass(frozen=True, slots=True)
class FileRecord:
    """One indexed file."""

    id: int
    root_id: int
    path: Path
    rel_path: Path
    media_type: MediaType
    signature: FileSignature
    status: FileStatus = FileStatus.PRESENT


@dataclass(frozen=True, slots=True)
class MediaMetadata:
    """Facts read from a file. Unknown values are None."""

    width: int | None = None
    height: int | None = None
    orientation: int | None = None
    taken_at: datetime | None = None
    taken_source: str | None = None
    camera_make: str | None = None
    camera_model: str | None = None
    gps_lat: float | None = None
    gps_lon: float | None = None
    codec: str | None = None
    extra: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class MatchMember:
    """A file in a group, its distance from the group's centre, and the evidence."""

    file_id: int
    distance: float = 0.0
    evidence: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class MatchGroup:
    """Files that match. ``id`` is 0 until it is stored."""

    kind: MatchKind
    members: tuple[MatchMember, ...]
    confidence: float = 1.0
    reclaimable_bytes: int = 0
    id: int = 0


@dataclass(frozen=True, slots=True)
class RuleScore:
    """A keeper rule's opinion: higher is better. ``reason`` is shown to the user."""

    value: float
    reason: str = ""
