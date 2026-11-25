from __future__ import annotations

import os
import time
import unicodedata
from collections.abc import Callable
from pathlib import Path

import pytest

from mediasift.core.errors import LibraryError
from mediasift.core.types import FileSignature, FileStatus, MediaType, Root
from mediasift.db.connection import Database
from mediasift.db.features import FeaturesRepo, FeatureValue
from mediasift.db.files import FileMove, FilesRepo, FoundFile
from mediasift.db.roots import RootsRepo
from mediasift.db.scans import ScansRepo
from tests.core.test_paths import UNDECODABLE
from tests.db.conftest import AddFiles

BIG = 2**64 - 3


@pytest.fixture
def files(db: Database) -> FilesRepo:
    return FilesRepo(db)


def found(root: Root, name: str | Path, sig: FileSignature, **more: FileStatus) -> FoundFile:
    return FoundFile(root.id, name, MediaType.IMAGE, sig, **more)


def test_round_trip(files: FilesRepo, root: Root, scan_id: int) -> None:
    sig = FileSignature(2**40, -5, BIG, BIG - 1, 3)
    name = Path("sub") / "صورة.JPG"
    assert files.upsert_many([found(root, name, sig)], scan_id) == 1
    (record,) = files.iter_root(root.id)
    assert record.signature == sig
    assert record.rel_path == name
    assert record.path == root.path / name
    assert record.media_type is MediaType.IMAGE
    assert record.status is FileStatus.PRESENT
    assert files.get(record.id) == record
    with pytest.raises(LibraryError, match="number 999"):
        files.get(999)


def test_odd_names(files: FilesRepo, db: Database, root: Root, scan_id: int) -> None:
    names = [
        UNDECODABLE,
        unicodedata.normalize("NFC", "é.jpg"),
        unicodedata.normalize("NFD", "é.jpg"),
        "x" * 200 + os.sep + "y" * 200 + ".jpg",
        "a%b_c'\".jpg" if os.name != "nt" else "a%b_c'.jpg",
    ]
    sig = FileSignature(1, 1)
    files.upsert_many([found(root, name, sig) for name in names], scan_id)
    assert {str(record.rel_path) for record in files.iter_root(root.id)} == set(names)
    assert set(files.find_by_paths(root.id, names)) == set(names)
    shown = [row[0] for row in db.execute("SELECT rel_display FROM files")]
    assert all(text.encode("utf-8") for text in shown)
    keys = db.execute("SELECT count(DISTINCT rel_key) FROM files").fetchone()[0]
    assert keys == len(names) - 1  # NFC and NFD share one


@pytest.mark.parametrize("name", ["../a.jpg", os.path.abspath("a.jpg"), "", "a/../../b.jpg"])
def test_paths_outside_the_root_are_refused(
    files: FilesRepo, root: Root, scan_id: int, name: str
) -> None:
    with pytest.raises(LibraryError, match="not a path inside"):
        files.upsert_many([found(root, name, FileSignature(1, 1))], scan_id)
    assert files.stats() == []


def test_unknown_root_is_refused(files: FilesRepo, root: Root, scan_id: int) -> None:
    item = FoundFile(root.id + 1, "a.jpg", MediaType.IMAGE, FileSignature(1, 1))
    with pytest.raises(LibraryError, match="no folder"):
        files.upsert_many([item], scan_id)


def test_upsert_updates_in_place(files: FilesRepo, db: Database, root: Root, scan_id: int) -> None:
    files.upsert_many([found(root, "a.jpg", FileSignature(1, 1, 2, 3))], scan_id)
    (before,) = files.iter_root(root.id)
    second = ScansRepo(db).start()
    changed = FileSignature(9, 9, 2, 4)
    files.upsert_many([found(root, "a.jpg", changed, status=FileStatus.ERROR)], second)
    (after,) = files.iter_root(root.id)
    assert after.id == before.id
    assert (after.signature, after.status) == (changed, FileStatus.ERROR)
    scans = db.execute("SELECT first_scan_id, last_scan_id FROM files").fetchone()
    assert scans == (scan_id, second)


def test_upsert_is_all_or_nothing(files: FilesRepo, root: Root, scan_id: int) -> None:
    good = found(root, "a.jpg", FileSignature(1, 1))
    bad = found(root, "../b.jpg", FileSignature(1, 1))
    with pytest.raises(LibraryError):
        files.upsert_many([good, bad], scan_id)
    assert files.stats() == []


def test_find_by_identity_and_size(files: FilesRepo, root: Root, scan_id: int) -> None:
    files.upsert_many(
        [
            found(root, "a.jpg", FileSignature(10, 50, BIG, BIG, 2)),
            found(root, "b.jpg", FileSignature(10, 50, BIG, BIG, 2)),
            found(root, "c.jpg", FileSignature(10, 51, 1, 0)),
        ],
        scan_id,
    )
    names = [r.rel_path.name for r in files.find_by_identity(BIG, BIG)]
    assert names == ["a.jpg", "b.jpg"]
    assert files.find_by_identity(1, 0) == []  # unknown identity matches nothing
    assert [r.rel_path.name for r in files.find_by_size_mtime(10, 51)] == ["c.jpg"]
    assert files.find_by_size_mtime(10, 52) == []


def test_seen_and_missing(
    files: FilesRepo, db: Database, add_files: AddFiles, make_root: Callable[[str], Root]
) -> None:
    a, b, c = add_files("a.jpg", "b.jpg", "c.jpg")
    other_root = make_root("other")
    (elsewhere,) = add_files("d.jpg", to=other_root)
    scan = ScansRepo(db).start()
    assert files.mark_seen([a.id], scan) == 1
    assert files.mark_missing(a.root_id, scan) == 2
    assert files.get(a.id).status is FileStatus.PRESENT
    assert files.get(b.id).status is FileStatus.MISSING
    assert files.get(elsewhere.id).status is FileStatus.PRESENT
    assert files.mark_missing(a.root_id, scan) == 0
    again = ScansRepo(db).start()
    files.mark_seen([c.id], again)  # comes back
    assert files.get(c.id).status is FileStatus.PRESENT


def test_missing_leaves_unread_folders_alone(
    files: FilesRepo, db: Database, add_files: AddFiles
) -> None:
    names = ["locked/a.jpg", "locked/deep/b.jpg", "locked2/c.jpg", "open/locked/d.jpg", "e.jpg"]
    records = add_files(*names)
    scan = ScansRepo(db).start()
    count = files.mark_missing(records[0].root_id, scan, unread_folders=["locked"])
    assert count == 3
    statuses = [files.get(r.id).status for r in records]
    assert statuses == [FileStatus.PRESENT] * 2 + [FileStatus.MISSING] * 3
    assert files.mark_missing(records[0].root_id, scan, unread_folders=[Path(".")]) == 0


def test_missing_streams_past_one_page(
    files: FilesRepo, db: Database, root: Root, scan_id: int
) -> None:
    sig = FileSignature(1, 1)
    files.upsert_many((found(root, f"{n}.jpg", sig) for n in range(2500)), scan_id)
    scan = ScansRepo(db).start()
    assert files.mark_missing(root.id, scan) == 2500


def test_move_keeps_id_and_features(
    files: FilesRepo, db: Database, add_files: AddFiles, make_root: Callable[[str], Root]
) -> None:
    (record,) = add_files("a.jpg")
    other = make_root("other")
    features = FeaturesRepo(db)
    sig = record.signature
    features.put_many([FeatureValue(record.id, "full", 1, b"hash", sig.size, sig.mtime_ns)])
    files.set_status([record.id], FileStatus.MISSING)
    scan = ScansRepo(db).start()
    files.move(FileMove(record.id, other.id, "new/صورة.jpg", sig), scan)
    moved = files.get(record.id)
    assert moved.path == other.path / "new" / "صورة.jpg"
    assert moved.status is FileStatus.PRESENT
    assert features.get(record.id, "full", 1) == b"hash"
    assert files.mark_missing(other.id, scan) == 0
    assert files.find_by_paths(record.root_id, ["a.jpg"]) == {}


def test_swap_and_chain(files: FilesRepo, add_files: AddFiles, scan_id: int) -> None:
    a, b, c = add_files("a.jpg", "b.jpg", "c.jpg")
    sig, root_id = a.signature, a.root_id
    files.move_many(
        [
            FileMove(a.id, root_id, "b.jpg", sig),
            FileMove(b.id, root_id, "a.jpg", sig),
            FileMove(c.id, root_id, "d.jpg", sig),
        ],
        scan_id,
    )
    places = {str(r.rel_path): r.id for r in files.iter_root(root_id)}
    assert places == {"b.jpg": a.id, "a.jpg": b.id, "d.jpg": c.id}


def test_move_onto_a_missing_file_forgets_it(
    files: FilesRepo, add_files: AddFiles, scan_id: int
) -> None:
    a, b = add_files("a.jpg", "b.jpg")
    files.set_status([b.id], FileStatus.MISSING)
    files.move(FileMove(a.id, a.root_id, "b.jpg", a.signature), scan_id)
    assert [(r.id, str(r.rel_path)) for r in files.iter_root(a.root_id)] == [(a.id, "b.jpg")]


def test_move_onto_a_present_file_changes_nothing(
    files: FilesRepo, add_files: AddFiles, scan_id: int
) -> None:
    a, b, c = add_files("a.jpg", "b.jpg", "c.jpg")
    moves = [
        FileMove(c.id, c.root_id, "d.jpg", c.signature),
        FileMove(a.id, a.root_id, "b.jpg", a.signature),
    ]
    with pytest.raises(LibraryError, match="already a known file"):
        files.move_many(moves, scan_id)
    assert [str(r.rel_path) for r in files.iter_root(a.root_id)] == ["a.jpg", "b.jpg", "c.jpg"]
    with pytest.raises(LibraryError, match="number 999"):
        files.move(FileMove(999, b.root_id, "x.jpg", b.signature), scan_id)


def test_present_files_of_enabled_roots(
    files: FilesRepo, db: Database, add_files: AddFiles, make_root: Callable[[str], Root]
) -> None:
    a, b, gone = add_files("a.jpg", "b.mp4", "gone.jpg")
    other = make_root("other")
    (c,) = add_files("c.jpg", to=other)
    files.set_status([gone.id], FileStatus.MISSING)

    def ids(*types: MediaType) -> list[int]:
        return [r.id for r in files.iter_present(types or None)]

    assert ids() == [a.id, b.id, c.id]
    assert ids(MediaType.IMAGE) == [a.id, c.id]
    assert ids(MediaType.VIDEO, MediaType.AUDIO) == [b.id]
    assert [r.id for r in files.iter_present([])] == []
    RootsRepo(db).set_enabled(other.id, False)
    assert ids() == [a.id, b.id]


def test_stats(
    files: FilesRepo, db: Database, add_files: AddFiles, make_root: Callable[[str], Root]
) -> None:
    a, *_ = add_files(("a.jpg", 10, 1), ("b.jpg", 5, 1), ("c.mp4", 100, 1))
    files.set_status([a.id], FileStatus.MISSING)
    other = make_root("other")
    add_files(("d.jpg", 1, 1), to=other)
    RootsRepo(db).set_enabled(other.id, False)
    rows = [(s.media_type, s.status, s.count, s.bytes) for s in files.stats()]
    assert rows == [
        (MediaType.IMAGE, FileStatus.MISSING, 1, 10),
        (MediaType.IMAGE, FileStatus.PRESENT, 1, 5),
        (MediaType.VIDEO, FileStatus.PRESENT, 1, 100),
    ]
    everything = {(s.media_type, s.status): s.count for s in files.stats(enabled_only=False)}
    assert everything[MediaType.IMAGE, FileStatus.PRESENT] == 2


def test_writing_while_iterating(files: FilesRepo, add_files: AddFiles, scan_id: int) -> None:
    records = add_files(*(f"{n}.jpg" for n in range(25)))
    seen = []
    for record in files.iter_present(page_size=4):
        seen.append(record.id)
        files.mark_seen([record.id], scan_id, FileStatus.ERROR)
    assert seen == [r.id for r in records]


def test_get_many_past_the_parameter_limit(files: FilesRepo, add_files: AddFiles) -> None:
    records = add_files(*(f"{n}.jpg" for n in range(1300)))
    assert set(files.get_many([*(r.id for r in records), 10**9])) == {r.id for r in records}


def test_fifty_thousand_rows_stream_quickly(
    files: FilesRepo, db: Database, root: Root, scan_id: int
) -> None:
    count = 50_000
    start = time.perf_counter()
    files.upsert_many(
        (found(root, f"d{n % 100}/{n}.jpg", FileSignature(n, n, 1, n + 1)) for n in range(count)),
        scan_id,
    )
    features = FeaturesRepo(db)
    lacking = features.iter_lacking("full", 1)
    values = (
        FeatureValue(r.id, "full", 1, b"v%d" % (r.id // 2), r.signature.size, r.signature.mtime_ns)
        for r in lacking
    )
    assert features.put_many(values) == count
    assert sum(1 for _ in files.iter_present()) == count
    assert sum(1 for _ in features.iter_groups("full", 1)) == count // 2 - 1 + count % 2
    # Generous for slow test machines; test_pages_never_sort guards the paging itself.
    assert time.perf_counter() - start < 60


def test_pages_never_sort(files: FilesRepo, db: Database, root: Root) -> None:
    """Every page of a long read must come straight from an ordered index."""
    plans: list[str] = []
    statements: list[str] = []
    db._conn.set_trace_callback(statements.append)
    features = FeaturesRepo(db)
    list(files.iter_root(root.id))
    list(files.iter_present([MediaType.IMAGE]))
    list(features.iter_lacking("full", 1, [MediaType.IMAGE], repeated_sizes_only=True))
    list(features.iter_groups("full", 1, [MediaType.IMAGE]))
    db._conn.set_trace_callback(None)
    paged = [sql for sql in statements if "LIMIT" in sql]
    assert len(paged) == 4
    for sql in paged:
        plans += [row[3] for row in db.execute(f"EXPLAIN QUERY PLAN {sql}")]
    assert not [line for line in plans if "TEMP B-TREE" in line]
    assert not [line for line in plans if line.startswith("SCAN")], plans
