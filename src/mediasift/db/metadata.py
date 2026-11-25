"""Metadata read from files, valid under the same rule as features."""

from __future__ import annotations

from collections.abc import Collection, Iterable, Iterator
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from mediasift.core.types import FileRecord, MediaMetadata, MediaType
from mediasift.db._util import (
    PAGE_SIZE,
    chunked,
    clean_optional,
    dump_json,
    load_json,
    marks,
)
from mediasift.db.connection import Database
from mediasift.db.files import ENABLED, FilesRepo, media_filter

_FIELDS = (
    "width, height, orientation, taken_at, taken_source, camera_make, camera_model,"
    " gps_lat, gps_lon, codec, extra"
)
_NAMES = ("file_id", "reader", "reader_version", *_FIELDS.split(", "), "src_size", "src_mtime_ns")

_UPSERT = (
    f"INSERT OR REPLACE INTO metadata ({', '.join(_NAMES)}) VALUES ({', '.join('?' * len(_NAMES))})"
)

_CURRENT = "m.src_size = f.size AND m.src_mtime_ns = f.mtime_ns"


@dataclass(frozen=True, slots=True)
class MetadataValue:
    """Metadata of a file, who read it, and the state of the file when it was read."""

    file_id: int
    reader: str
    reader_version: int
    metadata: MediaMetadata
    src_size: int
    src_mtime_ns: int


class MetadataRepo:
    """Rows of the metadata table."""

    def __init__(self, db: Database) -> None:
        self._db = db
        self._files = FilesRepo(db)

    def put_many(self, values: Iterable[MetadataValue]) -> int:
        """Store metadata, replacing older rows."""
        with self._db.transaction():
            return self._db.executemany(_UPSERT, map(_row, values))

    def get(self, file_id: int) -> MediaMetadata | None:
        """Valid metadata of one file, or None."""
        return self.get_many([file_id]).get(file_id)

    def get_many(self, file_ids: Iterable[int]) -> dict[int, MediaMetadata]:
        """Valid metadata by file id, whichever reader it came from."""
        found = {}
        for ids in chunked(file_ids):
            rows = self._db.execute(
                f"SELECT m.file_id, {_FIELDS} FROM metadata m JOIN files f ON f.id = m.file_id"
                f" WHERE m.file_id IN ({marks(len(ids))}) AND {_CURRENT}",
                ids,
            )
            for row in rows.fetchall():
                found[row[0]] = _metadata(row[1:])
        return found

    def iter_lacking(
        self,
        reader: str,
        version: int,
        media_types: Collection[MediaType] | None = None,
        *,
        extensions: Collection[str] | None = None,
        page_size: int = PAGE_SIZE,
    ) -> Iterator[FileRecord]:
        """Present files of enabled roots this reader has not read in their present state."""
        where, params = media_filter(media_types)
        if extensions is not None:
            names = {f"ext{n}": ext for n, ext in enumerate(sorted(extensions))}
            where += f" AND f.ext IN ({', '.join(f':{name}' for name in names)})"
            params.update(names)
        where = (
            f"f.status = 'present' AND {ENABLED}{where} AND NOT EXISTS ("
            "SELECT 1 FROM metadata m WHERE m.file_id = f.id AND m.reader = :reader"
            f" AND m.reader_version = :version AND {_CURRENT})"
        )
        params.update(reader=reader, version=version)
        yield from self._files.iter_where(where, params, page_size=page_size)


def _row(value: MetadataValue) -> tuple[Any, ...]:
    data = value.metadata
    return (
        value.file_id,
        value.reader,
        value.reader_version,
        data.width,
        data.height,
        data.orientation,
        # Local time, with the offset only when it is known.
        None if data.taken_at is None else data.taken_at.isoformat(),
        clean_optional(data.taken_source),
        clean_optional(data.camera_make),
        clean_optional(data.camera_model),
        data.gps_lat,
        data.gps_lon,
        clean_optional(data.codec),
        dump_json(data.extra),
        value.src_size,
        value.src_mtime_ns,
    )


def _metadata(row: tuple[Any, ...]) -> MediaMetadata:
    width, height, orientation, taken_at, source, make, model, lat, lon, codec, extra = row
    return MediaMetadata(
        width=width,
        height=height,
        orientation=orientation,
        taken_at=None if taken_at is None else datetime.fromisoformat(taken_at),
        taken_source=source,
        camera_make=make,
        camera_model=model,
        gps_lat=lat,
        gps_lon=lon,
        codec=codec,
        extra=load_json(extra),
    )
