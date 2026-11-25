"""Cached features: hashes and anything else computed from a file's content.

A row counts only while its version is the wanted one and the file still has the size and
modified time the value was computed from. Stale rows are simply ignored and overwritten.
"""

from __future__ import annotations

from collections.abc import Collection, Iterable, Iterator
from dataclasses import dataclass
from itertools import groupby
from typing import Any

from mediasift.core.types import FileRecord, MediaType
from mediasift.db._util import PAGE_SIZE, chunked, clean_optional, marks
from mediasift.db.connection import Database
from mediasift.db.files import ENABLED, FilesRepo, media_filter

# Feature "x" belongs to the present content of file "f".
_CURRENT = "x.src_size = f.size AND x.src_mtime_ns = f.mtime_ns"

# Looked up per file in the size index: counting sizes up front would repeat for each page.
_SIZE_REPEATS = (
    "EXISTS (SELECT 1 FROM files o WHERE o.size = f.size AND o.id <> f.id"
    " AND o.status = 'present'"
    " AND EXISTS (SELECT 1 FROM roots q WHERE q.id = o.root_id AND q.enabled = 1))"
)

_UPSERT = """
INSERT INTO features (file_id, feature, version, value, status, error, src_size, src_mtime_ns)
VALUES (?, ?, ?, ?, ?, ?, ?, ?)
ON CONFLICT (file_id, feature) DO UPDATE SET
  version = excluded.version, value = excluded.value, status = excluded.status,
  error = excluded.error, src_size = excluded.src_size, src_mtime_ns = excluded.src_mtime_ns
"""


@dataclass(frozen=True, slots=True)
class FeatureValue:
    """A computed feature, or why it cannot be computed (``value`` None, ``error`` set).

    ``src_size`` and ``src_mtime_ns`` are the state of the file when it was read.
    Store an error only when it depends on the content, such as a corrupt file: it is not
    tried again until the file changes. Leave out errors that may pass, such as a locked file.
    """

    file_id: int
    feature: str
    version: int
    value: bytes | None
    src_size: int
    src_mtime_ns: int
    error: str | None = None


@dataclass(frozen=True, slots=True)
class FeatureGroup:
    """Files sharing a feature value. ``size`` is None unless sizes were compared too."""

    value: bytes
    file_ids: tuple[int, ...]
    size: int | None = None


class FeaturesRepo:
    """Rows of the features table."""

    def __init__(self, db: Database) -> None:
        self._db = db
        self._files = FilesRepo(db)

    def put_many(self, features: Iterable[FeatureValue]) -> int:
        """Store features, replacing older values."""
        with self._db.transaction():
            return self._db.executemany(_UPSERT, map(_row, features))

    def get(self, file_id: int, feature: str, version: int) -> bytes | None:
        """The valid value of one file, or None."""
        return self.get_many([file_id], feature, version).get(file_id)

    def get_many(self, file_ids: Iterable[int], feature: str, version: int) -> dict[int, bytes]:
        """Valid values by file id. Files without one, or with an error, are left out."""
        found: dict[int, bytes] = {}
        for ids in chunked(file_ids):
            rows = self._db.execute(
                "SELECT x.file_id, x.value FROM features x JOIN files f ON f.id = x.file_id"
                f" WHERE x.file_id IN ({marks(len(ids))}) AND x.feature = ? AND x.version = ?"
                f" AND x.status = 'ok' AND {_CURRENT}",
                [*ids, feature, version],
            )
            found.update(rows.fetchall())
        return found

    def errors(self, feature: str, version: int, limit: int = 100) -> list[tuple[int, str]]:
        """File id and message of files whose feature is known to be impossible."""
        return self._db.execute(
            "SELECT x.file_id, x.error FROM features x JOIN files f ON f.id = x.file_id"
            f" WHERE x.feature = ? AND x.version = ? AND x.status = 'error' AND {_CURRENT}"
            " ORDER BY x.file_id LIMIT ?",
            (feature, version, limit),
        ).fetchall()

    def iter_lacking(
        self,
        feature: str,
        version: int,
        media_types: Collection[MediaType] | None = None,
        *,
        repeated_sizes_only: bool = False,
        retry_errors: bool = False,
        page_size: int = PAGE_SIZE,
    ) -> Iterator[FileRecord]:
        """Present files of enabled roots without a valid value.

        ``repeated_sizes_only`` leaves out files no other present file can be equal to.
        Storing features while iterating is fine.
        """
        where, params = media_filter(media_types)
        usable = " AND x.status = 'ok'" if retry_errors else ""
        where = (
            f"f.status = 'present' AND {ENABLED}{where} AND NOT EXISTS ("
            "SELECT 1 FROM features x WHERE x.file_id = f.id AND x.feature = :feature"
            f" AND x.version = :version AND {_CURRENT}{usable})"
        )
        if repeated_sizes_only:
            where += f" AND {_SIZE_REPEATS}"
        params.update(feature=feature, version=version)
        yield from self._files.iter_where(where, params, page_size=page_size)

    def iter_values(
        self,
        feature: str,
        version: int,
        media_types: Collection[MediaType] | None = None,
        *,
        page_size: int = PAGE_SIZE,
    ) -> Iterator[tuple[int, bytes]]:
        """File id and valid value of present files of enabled roots, ordered by value."""
        for value, file_id, _ in self._iter_by_value(feature, version, media_types, page_size):
            yield file_id, value

    def iter_groups(
        self,
        feature: str,
        version: int,
        media_types: Collection[MediaType] | None = None,
        *,
        same_size: bool = False,
        page_size: int = PAGE_SIZE,
    ) -> Iterator[FeatureGroup]:
        """Values that two or more present files of enabled roots share.

        ``same_size`` also requires equal file sizes, for partial hashes.
        """
        rows = self._iter_by_value(feature, version, media_types, page_size)
        for value, same_value in groupby(rows, key=lambda row: row[0]):
            by_size: dict[int | None, list[int]] = {}
            for _, file_id, size in same_value:
                by_size.setdefault(size if same_size else None, []).append(file_id)
            for shared_size, ids in by_size.items():
                if len(ids) > 1:
                    yield FeatureGroup(value, tuple(ids), shared_size)

    def _iter_by_value(
        self,
        feature: str,
        version: int,
        media_types: Collection[MediaType] | None,
        page_size: int,
    ) -> Iterator[tuple[bytes, int, int]]:
        where, params = media_filter(media_types)
        # CROSS JOIN fixes the order: walk the value index, so that no page needs a sort.
        sql = (
            "SELECT x.value, x.file_id, f.size"
            " FROM features x INDEXED BY features_value CROSS JOIN files f"
            " WHERE x.feature = :feature AND (x.value, x.file_id) > (:value, :file)"
            " AND x.version = :version AND x.status = 'ok' AND f.id = x.file_id"
            f" AND {_CURRENT} AND f.status = 'present' AND {ENABLED}{where}"
            " ORDER BY x.value, x.file_id LIMIT :limit"
        )
        params.update(feature=feature, version=version, limit=page_size, value=b"", file=0)
        while True:
            rows: list[tuple[bytes, int, int]] = self._db.execute(sql, params).fetchall()
            yield from rows
            if len(rows) < page_size:
                return
            params.update(value=rows[-1][0], file=rows[-1][1])


def _row(feature: FeatureValue) -> tuple[Any, ...]:
    status = "ok" if feature.value is not None else "error"
    error = None if feature.value is not None else (feature.error or "unknown error")
    return (
        feature.file_id,
        feature.feature,
        feature.version,
        feature.value,
        status,
        clean_optional(error),
        feature.src_size,
        feature.src_mtime_ns,
    )
