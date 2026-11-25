from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from mediasift.core.types import FileRecord, FileStatus, MediaMetadata, MediaType, Root
from mediasift.db.connection import Database
from mediasift.db.features import FeatureGroup, FeaturesRepo, FeatureValue
from mediasift.db.files import FilesRepo, FoundFile
from mediasift.db.metadata import MetadataRepo, MetadataValue
from mediasift.db.roots import RootsRepo
from tests.db.conftest import AddFiles

IMAGE = [MediaType.IMAGE]


@pytest.fixture
def features(db: Database) -> FeaturesRepo:
    return FeaturesRepo(db)


def value(record: FileRecord, data: bytes | None, version: int = 1, **more: str) -> FeatureValue:
    sig = record.signature
    return FeatureValue(record.id, "full", version, data, sig.size, sig.mtime_ns, **more)


def ids(records: Iterable[FileRecord]) -> list[int]:
    return [record.id for record in records]


def change(db: Database, record: FileRecord, scan_id: int, **what: int) -> FileRecord:
    """The file changes on disk and a scan notices."""
    sig = replace(record.signature, **what)
    files = FilesRepo(db)
    files.upsert_many([FoundFile(record.root_id, record.rel_path, record.media_type, sig)], scan_id)
    return files.get(record.id)


def test_put_and_get(features: FeaturesRepo, add_files: AddFiles) -> None:
    a, b, c = add_files("a.jpg", "b.jpg", "c.jpg")
    assert features.put_many([value(a, b"1"), value(b, b"")]) == 2
    assert features.get_many([a.id, b.id, c.id], "full", 1) == {a.id: b"1", b.id: b""}
    features.put_many([value(a, b"2")])
    assert features.get(a.id, "full", 1) == b"2"
    assert features.get(a.id, "other", 1) is None
    assert features.get(a.id, "full", 2) is None


@pytest.mark.parametrize("what", [{"size": 101}, {"mtime_ns": 1001}])
def test_invalid_once_the_file_changes(
    features: FeaturesRepo, db: Database, add_files: AddFiles, scan_id: int, what: dict[str, int]
) -> None:
    a, b = add_files("a.jpg", "b.jpg")
    features.put_many([value(a, b"same"), value(b, b"same")])
    assert ids(features.iter_lacking("full", 1)) == []
    assert len(list(features.iter_groups("full", 1))) == 1

    changed = change(db, a, scan_id, **what)
    assert features.get(a.id, "full", 1) is None
    assert ids(features.iter_lacking("full", 1)) == [a.id]
    assert list(features.iter_groups("full", 1)) == []
    features.put_many([value(changed, b"new")])
    assert features.get(a.id, "full", 1) == b"new"


def test_same_content_state_stays_valid(
    features: FeaturesRepo, db: Database, add_files: AddFiles, scan_id: int
) -> None:
    (a,) = add_files("a.jpg")
    features.put_many([value(a, b"1")])
    change(db, a, scan_id, inode=12345, nlink=2)  # renamed or linked, not edited
    assert features.get(a.id, "full", 1) == b"1"


def test_value_computed_from_an_older_state_is_never_valid(
    features: FeaturesRepo, add_files: AddFiles
) -> None:
    (a,) = add_files("a.jpg")
    features.put_many([replace(value(a, b"1"), src_mtime_ns=5)])
    assert features.get(a.id, "full", 1) is None


def test_new_version_is_lacking(features: FeaturesRepo, add_files: AddFiles) -> None:
    a, b = add_files("a.jpg", "b.mp4")
    features.put_many([value(a, b"1"), value(b, b"1")])
    assert ids(features.iter_lacking("full", 2)) == [a.id, b.id]
    assert ids(features.iter_lacking("full", 2, IMAGE)) == [a.id]
    assert list(features.iter_groups("full", 2)) == []


def test_errors_are_not_retried_until_the_file_changes(
    features: FeaturesRepo, db: Database, add_files: AddFiles, scan_id: int
) -> None:
    a, b = add_files("a.jpg", "b.jpg")
    features.put_many([value(a, None, error="bad \udcff"), value(b, b"1")])
    assert features.get(a.id, "full", 1) is None
    assert features.errors("full", 1) == [(a.id, "bad \\udcff")]
    assert ids(features.iter_lacking("full", 1)) == []
    assert ids(features.iter_lacking("full", 1, retry_errors=True)) == [a.id]
    change(db, a, scan_id, size=1)
    assert ids(features.iter_lacking("full", 1)) == [a.id]
    assert features.errors("full", 1) == []


def test_lacking_with_repeated_sizes(
    features: FeaturesRepo, db: Database, add_files: AddFiles, make_root: Callable[[str], Root]
) -> None:
    a, b, _alone, gone, c = add_files(
        ("a.jpg", 5, 1), ("b.mp4", 5, 2), ("c.jpg", 6, 1), ("gone.jpg", 7, 1), ("d.jpg", 7, 1)
    )
    files = FilesRepo(db)
    other = make_root("other")
    (far,) = add_files(("far.jpg", 6, 1), to=other)
    RootsRepo(db).set_enabled(other.id, False)
    files.set_status([gone.id], FileStatus.MISSING)

    def lacking(*types: MediaType) -> list[int]:
        return ids(features.iter_lacking("full", 1, types or None, repeated_sizes_only=True))

    assert lacking() == [a.id, b.id]
    assert lacking(MediaType.IMAGE) == [a.id]  # its twin may be of another type
    files.set_status([gone.id], FileStatus.PRESENT)
    RootsRepo(db).set_enabled(other.id, True)
    assert lacking() == ids([a, b, _alone, gone, c, far])


def test_storing_while_iterating(features: FeaturesRepo, add_files: AddFiles) -> None:
    records = add_files(*(f"{n}.jpg" for n in range(30)))
    seen = []
    for record in features.iter_lacking("full", 1, page_size=4):
        seen.append(record.id)
        features.put_many([value(record, b"x")])
    assert seen == ids(records)
    assert ids(features.iter_lacking("full", 1)) == []


@pytest.mark.parametrize("page_size", [1, 2, 3, 1000])
def test_groups(features: FeaturesRepo, add_files: AddFiles, page_size: int) -> None:
    a, b, c, d, e = add_files(
        ("a.jpg", 1, 1), ("b.jpg", 1, 1), ("c.mp4", 1, 1), ("d.jpg", 2, 1), ("e.jpg", 1, 1)
    )
    features.put_many([value(a, b"x"), value(b, b"y"), value(c, b"x"), value(d, b"x")])
    features.put_many([value(e, b"z")])
    groups = list(features.iter_groups("full", 1, page_size=page_size))
    assert groups == [FeatureGroup(b"x", (a.id, c.id, d.id))]
    sized = list(features.iter_groups("full", 1, same_size=True, page_size=page_size))
    assert sized == [FeatureGroup(b"x", (a.id, c.id), 1)]
    images = list(features.iter_groups("full", 1, IMAGE, page_size=page_size))
    assert images == [FeatureGroup(b"x", (a.id, d.id))]
    pairs = list(features.iter_values("full", 1, IMAGE, page_size=page_size))
    assert pairs == [(a.id, b"x"), (d.id, b"x"), (b.id, b"y"), (e.id, b"z")]


def test_groups_hold_present_files_of_enabled_roots(
    features: FeaturesRepo, db: Database, add_files: AddFiles, make_root: Callable[[str], Root]
) -> None:
    a, b = add_files("a.jpg", "b.jpg")
    other = make_root("other")
    (c,) = add_files("c.jpg", to=other)
    features.put_many([value(r, b"x") for r in (a, b, c)])
    RootsRepo(db).set_enabled(other.id, False)
    assert list(features.iter_groups("full", 1)) == [FeatureGroup(b"x", (a.id, b.id))]
    FilesRepo(db).set_status([b.id], FileStatus.MISSING)
    assert list(features.iter_groups("full", 1)) == []


# --- metadata ---


def meta(record: FileRecord, data: MediaMetadata, version: int = 1) -> MetadataValue:
    sig = record.signature
    return MetadataValue(record.id, "image", version, data, sig.size, sig.mtime_ns)


def test_metadata_round_trip(db: Database, add_files: AddFiles) -> None:
    a, b, c = add_files("a.jpg", "b.jpg", "c.jpg")
    repo = MetadataRepo(db)
    full = MediaMetadata(
        width=4000,
        height=3000,
        orientation=6,
        taken_at=datetime(2024, 2, 29, 13, 5, 59, tzinfo=timezone(timedelta(hours=2))),
        taken_source="exif",
        camera_make="كاميرا",
        camera_model="X-1",
        gps_lat=30.0444,
        gps_lon=-31.2357,
        codec="jpeg",
        extra={"iso": 100, "tags": ["a", "ب"], "nested": {"x": None}},
    )
    naive = MediaMetadata(taken_at=datetime(2024, 1, 2, 3, 4, 5, 6))
    repo.put_many([meta(a, full), meta(b, naive)])
    assert repo.get(a.id) == full
    assert repo.get(b.id) == naive
    assert repo.get(c.id) is None
    repo.put_many([meta(a, naive)])
    assert repo.get(a.id) == naive


def test_metadata_tolerates_what_files_contain(db: Database, add_files: AddFiles) -> None:
    (a,) = add_files("a.jpg")
    repo = MetadataRepo(db)
    odd = MediaMetadata(
        camera_make="Maker\0\0",
        camera_model="bad \udcff",
        extra={"nan": float("nan"), "raw": b"\x01\xff", "when": datetime(2024, 1, 1), 5: {1}},  # type: ignore[dict-item]
    )
    repo.put_many([meta(a, odd)])
    read = repo.get(a.id)
    assert read is not None
    assert (read.camera_make, read.camera_model) == ("Maker", "bad \\udcff")
    assert read.extra == {"nan": None, "raw": "01ff", "when": "2024-01-01 00:00:00", "5": [1]}


def test_metadata_validity(db: Database, add_files: AddFiles, scan_id: int) -> None:
    a, b, c = add_files("a.jpg", "b.jpg", "c.mp4")
    repo = MetadataRepo(db)
    repo.put_many([meta(a, MediaMetadata(width=1))])
    assert ids(repo.iter_lacking("image", 1)) == [b.id, c.id]
    assert ids(repo.iter_lacking("image", 1, IMAGE)) == [b.id]
    assert ids(repo.iter_lacking("image", 1, extensions={".mp4", ".mov"})) == [c.id]
    assert ids(repo.iter_lacking("image", 2, IMAGE)) == [a.id, b.id]
    assert ids(repo.iter_lacking("other", 1, IMAGE)) == [a.id, b.id]
    change(db, a, scan_id, mtime_ns=5)
    assert repo.get(a.id) is None
    assert ids(repo.iter_lacking("image", 1, IMAGE)) == [a.id, b.id]
