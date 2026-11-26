from __future__ import annotations

import os
import shutil
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

from mediasift.core import paths
from mediasift.core.errors import LibraryError
from mediasift.core.types import FileStatus, MediaType, Root
from mediasift.db._util import to_signed
from mediasift.db.connection import Database
from mediasift.db.features import FeaturesRepo, FeatureValue
from mediasift.db.files import FilesRepo
from mediasift.db.roots import RootsRepo
from mediasift.db.scans import ScansRepo, ScanState
from mediasift.jobs.context import JobContext, Progress
from mediasift.scan import walker as walker_module
from mediasift.scan.walker import Entry, Walker
from tests.media.tree import build_tree, cheap_tree
from tests.scan.conftest import MakeRoot, present, records, run, settings, write

T0 = 1_600_000_000 * 10**9


def make_hardlink(source: Path, target: Path) -> None:
    try:
        os.link(source, target)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"no hard links here: {exc}")


def put_feature(db: Database, path: Path) -> int:
    record = records(db)[path]
    sig = record.signature
    FeaturesRepo(db).put_many([FeatureValue(record.id, "h", 1, b"v", sig.size, sig.mtime_ns)])
    return record.id


def feature(db: Database, file_id: int) -> bytes | None:
    return FeaturesRepo(db).get(file_id, "h", 1)


# --- first scan and rescans ---


def test_first_scan_finds_exactly_the_media(db: Database, root: Root) -> None:
    expected = build_tree(root.path)
    result = run(db)
    assert result.state is ScanState.DONE
    wanted = {*expected.media_paths, *expected.empty_files}
    assert set(present(db)) == wanted
    assert result.new == result.seen == len(wanted)
    assert result.missing == result.moved == result.errors == 0
    assert sum(result.by_type.values()) == len(wanted)
    assert result.by_type[MediaType.VIDEO.value] == 5
    stored = ScansRepo(db).get(result.scan_id)
    assert stored.state is ScanState.DONE
    assert stored.counters["new"] == len(wanted)


def test_second_scan_changes_nothing(db: Database, root: Root) -> None:
    build_tree(root.path)
    run(db)
    before = records(db)
    result = run(db)
    assert result.unchanged == result.seen == len(before)
    assert result.new == result.changed == result.moved == result.missing == 0
    assert records(db) == before


def test_changed_file_keeps_its_id(db: Database, root: Root) -> None:
    path = write(root.path / "a.jpg", b"one", T0)
    run(db)
    file_id = put_feature(db, path)
    write(path, b"longer", T0 + 10**9)
    result = run(db)
    assert result.changed == 1
    assert present(db) == {path: file_id}
    assert feature(db, file_id) is None  # stale by signature


# --- moves ---


def test_rename_and_move_keep_ids_and_features(db: Database, root: Root) -> None:
    a = write(root.path / "x" / "a.jpg", b"aaa", T0)
    b = write(root.path / "x" / "b.jpg", b"bbbb", T0)
    run(db)
    ids = {a: put_feature(db, a), b: put_feature(db, b)}
    renamed = root.path / "x" / "renamed.jpg"
    moved = root.path / "y" / "deeper" / "b.jpg"
    moved.parent.mkdir(parents=True)
    a.rename(renamed)
    b.rename(moved)
    result = run(db)
    assert result.moved == 2
    assert result.new == result.missing == 0
    assert present(db) == {renamed: ids[a], moved: ids[b]}
    assert feature(db, ids[a]) == b"v"
    assert feature(db, ids[b]) == b"v"


@pytest.mark.parametrize("order", ["source_first", "target_first"])
def test_move_between_roots_in_any_order(db: Database, make_root: MakeRoot, order: str) -> None:
    one, two = make_root("one"), make_root("two")
    path = write(one.path / "a.jpg", b"data", T0)
    run(db)
    file_id = put_feature(db, path)
    target = two.path / "sub" / "a.jpg"
    target.parent.mkdir()
    path.rename(target)
    result = run(db, [one, two] if order == "source_first" else [two, one])
    assert result.moved == 1
    assert result.missing == 0
    assert present(db) == {target: file_id}
    assert feature(db, file_id) == b"v"


def test_rename_with_edit_is_new_and_missing(db: Database, root: Root) -> None:
    path = write(root.path / "a.jpg", b"data", T0)
    run(db)
    old_id = present(db)[path]
    target = root.path / "b.jpg"
    path.rename(target)
    write(target, b"edited!", T0 + 5 * 10**9)
    result = run(db)
    assert (result.new, result.missing, result.moved) == (1, 1, 0)
    assert records(db)[path].status is FileStatus.MISSING
    assert present(db)[target] != old_id


def test_swap_of_two_names_follows_the_content(db: Database, root: Root) -> None:
    a = write(root.path / "a.jpg", b"aaaa", T0)
    b = write(root.path / "b.jpg", b"bbbbbbbb", T0 + 10**9)
    run(db)
    ids = {a: put_feature(db, a), b: put_feature(db, b)}
    temp = root.path / "temp"
    a.rename(temp)
    b.rename(a)
    temp.rename(b)
    result = run(db)
    assert result.moved == 2
    assert present(db) == {a: ids[b], b: ids[a]}
    assert feature(db, ids[a]) == feature(db, ids[b]) == b"v"


def test_rename_chain(db: Database, root: Root) -> None:
    names = [root.path / f"{n}.jpg" for n in "abc"]
    write(names[0], b"1", T0)
    write(names[1], b"22", T0)
    run(db)
    ids = present(db)
    names[1].rename(names[2])
    names[0].rename(names[1])
    result = run(db)
    assert result.moved == 2
    assert present(db) == {names[1]: ids[names[0]], names[2]: ids[names[1]]}


def test_copy_is_new(db: Database, root: Root) -> None:
    path = write(root.path / "a.jpg", b"data", T0)
    run(db)
    file_id = present(db)[path]
    copy = root.path / "copies" / "a.jpg"
    copy.parent.mkdir()
    shutil.copy2(path, copy)  # same name, size and time: not a move, the original is there
    result = run(db)
    assert result.new == 1
    assert result.moved == 0
    assert present(db)[path] == file_id
    assert copy in present(db)


def test_hard_link_is_new_and_both_stay_fresh(db: Database, root: Root) -> None:
    path = write(root.path / "a.jpg", b"data", T0)
    run(db)
    file_id = present(db)[path]
    link = root.path / "link.jpg"
    make_hardlink(path, link)
    result = run(db)
    assert (result.new, result.moved) == (1, 0)
    found = records(db)
    assert found[path].id == file_id
    assert found[path].signature.same_file(found[link].signature)
    assert found[path].signature.nlink == found[link].signature.nlink == 2
    # Written through one name, the listing of the other may be stale on Windows.
    with open(link, "ab") as handle:
        handle.write(b" and more")
    os.utime(link, ns=(T0 + 10**9, T0 + 10**9))
    run(db)
    found = records(db)
    assert found[path].signature.size == found[link].signature.size == len(b"data and more")


def test_hard_link_added_while_original_unchanged(db: Database, root: Root) -> None:
    path = write(root.path / "a.jpg", b"data", T0)
    run(db)
    other = root.path / "sub" / "a.jpg"
    other.parent.mkdir()
    make_hardlink(path, other)
    result = run(db)
    assert result.moved == 0
    assert set(present(db)) == {path, other}


def test_inode_reuse_is_not_a_move(db: Database, root: Root) -> None:
    old = write(root.path / "old.jpg", b"old content", T0)
    run(db)
    old_id = present(db)[old]
    old.unlink()
    new = write(root.path / "new.jpg", b"new", T0 + 10**9)
    sig = paths.signature(new)
    # Simulate a file system that gave the deleted file's inode to the new one.
    db.execute(
        "UPDATE files SET dev = ?, inode = ? WHERE id = ?",
        (to_signed(sig.dev), to_signed(sig.inode), old_id),
    )
    result = run(db)
    assert (result.new, result.missing, result.moved) == (1, 1, 0)
    assert present(db)[new] != old_id


def test_moved_file_from_another_volume(db: Database, make_root: MakeRoot) -> None:
    one, two = make_root("one"), make_root("two")
    path = write(one.path / "a.jpg", b"data", T0)
    run(db)
    file_id = put_feature(db, path)
    # Pretend the root lived on another device: the identity cannot match.
    db.execute("UPDATE files SET dev = dev + 1 WHERE id = ?", (file_id,))
    target = two.path / "a.jpg"
    shutil.copy2(path, target)
    path.unlink()
    result = run(db)
    assert result.moved == 1
    assert present(db) == {target: file_id}


def test_cross_volume_guess_needs_one_candidate(db: Database, make_root: MakeRoot) -> None:
    one, two = make_root("one"), make_root("two")
    first = write(one.path / "x" / "a.jpg", b"data", T0)
    second = write(one.path / "y" / "a.jpg", b"data", T0)
    run(db)
    ids = present(db)
    db.execute("UPDATE files SET dev = dev + 1")
    target = two.path / "a.jpg"
    shutil.copy2(first, target)
    first.unlink()
    second.unlink()
    result = run(db)
    assert result.moved == 0
    assert result.new == 1
    assert present(db)[target] not in ids.values()


def test_cross_volume_guess_with_two_arrivals(db: Database, make_root: MakeRoot) -> None:
    one, two = make_root("one"), make_root("two")
    path = write(one.path / "a.jpg", b"data", T0)
    run(db)
    db.execute("UPDATE files SET dev = dev + 1")
    for folder in ("p", "q"):
        (two.path / folder).mkdir()
        shutil.copy2(path, two.path / folder / "a.jpg")
    path.unlink()
    result = run(db)
    assert (result.moved, result.new, result.missing) == (0, 2, 1)


@pytest.mark.skipif(sys.platform != "win32", reason="case-insensitive volume needed")
def test_case_only_rename(db: Database, root: Root) -> None:
    assert not root.case_sensitive
    path = write(root.path / "Folder" / "IMG.jpg", b"data", T0)
    run(db)
    file_id = present(db)[path]
    path.rename(root.path / "Folder" / "img.JPG")
    (root.path / "Folder").rename(root.path / "folder")
    result = run(db)
    assert result.moved == 1
    assert present(db) == {root.path / "folder" / "img.JPG": file_id}


def test_deleted_and_restored(db: Database, root: Root) -> None:
    path = write(root.path / "a.jpg", b"data", T0)
    run(db)
    file_id = put_feature(db, path)
    path.unlink()
    assert run(db).missing == 1
    assert records(db)[path].status is FileStatus.MISSING
    write(path, b"data", T0)
    result = run(db)
    assert result.restored == 1
    assert present(db) == {path: file_id}
    assert feature(db, file_id) == b"v"


def test_moved_while_missing_comes_back(db: Database, root: Root) -> None:
    path = write(root.path / "a.jpg", b"data", T0)
    run(db)
    file_id = present(db)[path]
    hidden = root.path.parent / "aside.jpg"
    path.rename(hidden)
    run(db)
    target = root.path / "b.jpg"
    hidden.rename(target)
    assert run(db).moved == 1
    assert present(db) == {target: file_id}


# --- roots that cannot be trusted ---


def test_unplugged_root_keeps_its_files(db: Database, make_root: MakeRoot) -> None:
    one, two = make_root("one"), make_root("two")
    a = write(one.path / "a.jpg", b"a", T0)
    run(db)
    one.path.rename(one.path.parent / "away")
    b = write(two.path / "b.jpg", b"b", T0)
    result = run(db)
    assert result.unreachable_roots == [one.id]
    assert result.missing == 0
    assert set(present(db)) == {a, b}
    issues = ScansRepo(db).issues(result.scan_id)
    assert [issue.kind for issue in issues] == ["root_unreachable"]


def test_file_from_an_unreachable_root_is_not_a_move(db: Database, make_root: MakeRoot) -> None:
    one, two = make_root("one"), make_root("two")
    a = write(one.path / "a.jpg", b"a", T0)
    run(db)
    old_id = present(db)[a]
    a.rename(two.path / "a.jpg")
    one.path.rename(one.path.parent / "away")  # its old place cannot be checked now
    result = run(db, [two])
    assert (result.moved, result.new) == (0, 1)
    assert present(db)[a] == old_id


def test_empty_mount_point_keeps_files(db: Database, root: Root) -> None:
    a = write(root.path / "a.jpg", b"a", T0)
    run(db)
    shutil.rmtree(root.path)
    root.path.mkdir()  # what an unplugged drive leaves behind is on another volume
    RootsRepo(db).set_volume_id(root.id, "dev:another")
    result = run(db)
    assert result.unreachable_roots == [root.id]
    assert set(present(db)) == {a}


def test_root_going_away_during_the_walk(
    db: Database, root: Root, monkeypatch: pytest.MonkeyPatch
) -> None:
    a = write(root.path / "a.jpg", b"a", T0)
    write(root.path / "b.jpg", b"b", T0)
    run(db)
    real = walker_module.root_problem
    calls = iter([None, "gone"])
    monkeypatch.setattr(
        "mediasift.scan.scanner.root_problem",
        lambda r: next(calls, None) if r.id == root.id else real(r),
    )
    a.unlink()
    result = run(db)
    assert result.missing == 0
    assert result.unreachable_roots == [root.id]


def test_unreadable_folder_keeps_its_files(
    db: Database, root: Root, monkeypatch: pytest.MonkeyPatch
) -> None:
    inside = write(root.path / "locked" / "deep" / "a.jpg", b"a", T0)
    outside = write(root.path / "b.jpg", b"b", T0)
    run(db)
    real = os.scandir

    def scandir(path: str) -> object:
        if paths.plain_path(path).endswith("locked"):
            raise PermissionError(13, "Permission denied")
        return real(path)

    monkeypatch.setattr(walker_module.os, "scandir", scandir)
    outside.unlink()
    result = run(db)
    assert result.missing == 1
    assert inside in present(db)
    kinds = [issue.kind for issue in ScansRepo(db).issues(result.scan_id)]
    assert kinds == ["unreadable_folder"]


@pytest.mark.skipif(os.name == "nt" or os.geteuid() == 0, reason="POSIX permissions")
def test_unreadable_folder_by_permissions(db: Database, root: Root) -> None:
    inside = write(root.path / "locked" / "a.jpg", b"a", T0)
    run(db)
    (root.path / "locked").chmod(0)
    try:
        result = run(db)
    finally:
        (root.path / "locked").chmod(0o755)
    assert result.errors == 1
    assert inside in present(db)


def test_file_that_cannot_be_asked_is_kept(
    db: Database, root: Root, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = write(root.path / "a.jpg", b"a", T0)
    run(db)
    file_id = present(db)[path]
    original = Walker.files

    def files(self: Walker) -> Iterator[Entry]:
        for entry in original(self):
            yield Entry(entry.rel, entry.media_type, None)

    monkeypatch.setattr(Walker, "files", files)
    result = run(db)
    assert result.missing == 0
    assert records(db)[path].status is FileStatus.ERROR
    assert records(db)[path].id == file_id
    monkeypatch.undo()
    run(db)
    assert present(db) == {path: file_id}


# --- races ---


def inject(monkeypatch: pytest.MonkeyPatch, name: str, action: object) -> None:
    """Run ``action`` right after the walker lists the file called ``name``."""
    original = Walker.files

    def files(self: Walker) -> Iterator[Entry]:
        for entry in original(self):
            yield entry
            if os.path.basename(entry.rel) == name:
                action(self.root.path / entry.rel)  # type: ignore[operator]

    monkeypatch.setattr(Walker, "files", files)


def test_files_vanishing_or_growing_during_the_walk(
    db: Database, root: Root, monkeypatch: pytest.MonkeyPatch
) -> None:
    gone = write(root.path / "gone.jpg", b"gone", T0)
    grows = write(root.path / "grows.jpg", b"small", T0)
    inject(monkeypatch, "gone.jpg", Path.unlink)
    first = run(db, config=settings(batch_size=1000))
    assert first.state is ScanState.DONE
    monkeypatch.undo()
    inject(monkeypatch, "grows.jpg", lambda p: write(p, b"much larger now", T0 + 10**9))
    run(db, config=settings(batch_size=1000))
    monkeypatch.undo()
    run(db)
    found = records(db)
    assert gone not in present(db)
    assert found[grows].signature.size == len(b"much larger now")


# --- links ---


def make_folder_link(link: Path, target: Path) -> None:
    try:
        link.symlink_to(target, target_is_directory=True)
    except OSError:
        if os.name != "nt":
            raise
        import _winapi

        _winapi.CreateJunction(str(target), str(link))  # type: ignore[attr-defined]


def test_links_and_loops_are_not_followed(db: Database, make_root: MakeRoot) -> None:
    root, other = make_root("photos"), make_root("other")
    write(root.path / "a" / "a.jpg", b"a", T0)
    write(other.path / "o.jpg", b"o", T0)
    make_folder_link(root.path / "a" / "loop", root.path)
    make_folder_link(root.path / "to_other", other.path)
    result = run(db, [root])
    assert result.skipped_links == 2
    assert set(present(db)) == {root.path / "a" / "a.jpg"}


def test_file_symlink_is_skipped(db: Database, root: Root) -> None:
    target = write(root.path / "a.jpg", b"a", T0)
    try:
        (root.path / "link.jpg").symlink_to(target)
    except OSError as exc:
        pytest.skip(f"no symlinks here: {exc}")
    result = run(db)
    assert result.skipped_links == 1
    assert set(present(db)) == {target}


# --- rules ---


def test_excluded_and_system_folders(db: Database, root: Root, tmp_path: Path) -> None:
    keep = write(root.path / "keep.jpg", b"k", T0)
    write(root.path / "private" / "deep" / "private" / "a.jpg", b"p", T0)
    write(root.path / "x" / "private" / "b.jpg", b"p", T0)
    for folder in (".mediasift-quarantine", "$RECYCLE.BIN", "@eaDir", ".Trash-1000", ".hidden"):
        write(root.path / folder / "c.jpg", b"q", T0)
    write(root.path / ".dot.jpg", b"d", T0)
    write(root.path / "small.jpg", b"", T0)
    run(db, config=settings(exclude=("private/",), min_size=1))
    assert set(present(db)) == {keep}


def test_library_inside_a_root_is_skipped(tmp_path: Path) -> None:
    folder = tmp_path / "photos"
    keep = write(folder / "a.jpg", b"a", T0)
    with Database.open(folder / "lib") as db:
        write(folder / "lib" / "thumb.jpg", b"t", T0)
        RootsRepo(db).add(folder)
        run(db)
        assert set(present(db)) == {keep}


def test_excludes_added_later_mark_files_missing(db: Database, root: Root) -> None:
    keep = write(root.path / "a.jpg", b"a", T0)
    hidden = write(root.path / "raw" / "b.jpg", b"b", T0)
    run(db)
    file_id = present(db)[hidden]
    result = run(db, config=settings(exclude=("raw",)))
    assert result.missing == 1
    assert set(present(db)) == {keep}
    result = run(db)
    assert result.restored == 1
    assert present(db)[hidden] == file_id


# --- names ---


def test_unusual_names(db: Database, root: Root) -> None:
    names = ["صورة ١.jpg", "My Holiday ✈ 2020 😀.jpg", "trailing.jpg", "éte.jpg"]
    if os.name == "nt":
        names.append("lone \udc80 surrogate.jpg")
    else:
        names.append(os.fsdecode(b"undecodable \xff.jpg"))
    made = set()
    for name in names:
        try:
            made.add(write(root.path / name, name.encode("utf-8", "surrogateescape"), T0))
        except (OSError, UnicodeError):
            continue
    run(db)
    assert set(present(db)) == made
    first = present(db)
    assert run(db).unchanged == len(made)
    assert present(db) == first


def test_long_paths(db: Database, root: Root) -> None:
    folder = root.path
    for n in range(12):
        folder = folder / f"{n:02d}_{'x' * 28}"
    long = folder / "a.jpg"
    os.makedirs(paths.fs_path(folder))
    with open(paths.fs_path(long), "wb") as handle:
        handle.write(b"data")
    assert len(str(long)) > 300
    run(db)
    file_id = present(db)[long]
    renamed = folder / "b.jpg"
    os.rename(paths.fs_path(long), paths.fs_path(renamed))
    assert run(db).moved == 1
    assert present(db) == {renamed: file_id}


# --- placeholders ---


def test_placeholders_are_recorded(
    db: Database, root: Root, monkeypatch: pytest.MonkeyPatch
) -> None:
    cloud = write(root.path / "cloud.jpg", b"x" * 777, T0)
    local = write(root.path / "local.jpg", b"local", T0)
    monkeypatch.setattr(paths, "is_placeholder", lambda st: st.st_size == 777)
    result = run(db)
    assert result.placeholders == 1
    found = records(db)
    assert found[cloud].status is FileStatus.PLACEHOLDER
    assert found[local].status is FileStatus.PRESENT
    assert run(db).placeholders == 1
    monkeypatch.undo()
    run(db)
    assert records(db)[cloud].status is FileStatus.PRESENT
    assert records(db)[cloud].id == found[cloud].id


# --- subsets and states ---


def test_scanning_one_root_leaves_the_other(db: Database, make_root: MakeRoot) -> None:
    one, two = make_root("one"), make_root("two")
    write(one.path / "a.jpg", b"a", T0)
    b = write(two.path / "b.jpg", b"b", T0)
    run(db)
    b.unlink()
    assert run(db, [one]).missing == 0
    assert b in present(db)
    assert ScansRepo(db).latest().options == {"roots": [one.id]}  # type: ignore[union-attr]


def test_disabled_root_is_refused(db: Database, root: Root) -> None:
    RootsRepo(db).set_enabled(root.id, False)
    with pytest.raises(LibraryError, match="disabled"):
        run(db, [root])
    assert run(db).seen == 0  # not scanned by default either


def test_failure_marks_the_scan_failed(
    db: Database, root: Root, monkeypatch: pytest.MonkeyPatch
) -> None:
    write(root.path / "a.jpg", b"a", T0)

    def broken(*args: object, **kwargs: object) -> int:
        raise RuntimeError("disk on fire")

    monkeypatch.setattr(FilesRepo, "upsert_many", broken)
    with pytest.raises(RuntimeError):
        run(db)
    assert ScansRepo(db).latest().state is ScanState.FAILED  # type: ignore[union-attr]


def test_progress(db: Database, root: Root) -> None:
    for n in range(7):
        write(root.path / f"d{n % 2}" / f"{n}.jpg", b"x" * (n + 1), T0)
    seen: list[Progress] = []
    result = run(db, config=settings(batch_size=2), job=JobContext(on_progress=seen.append))
    assert {p.stage for p in seen} == {"scan"}
    assert seen[0].total is None
    assert seen[-1].done == seen[-1].total == result.seen == 7
    assert seen[-1].bytes_done == sum(range(1, 8))


# --- cancel and resume ---


def snapshot(db: Database) -> dict[str, tuple[str, int, int]]:
    return {
        str(r.rel_path): (r.status.value, r.signature.size, r.signature.mtime_ns)
        for r in records(db).values()
    }


def cancel_after(monkeypatch: pytest.MonkeyPatch, job: JobContext, count: int) -> None:
    original = Walker.files

    def files(self: Walker) -> Iterator[Entry]:
        for n, entry in enumerate(original(self)):
            if n == count:
                job.cancel.cancel()
            yield entry
        if count < 0:  # after the whole walk, before moves are settled
            job.cancel.cancel()

    monkeypatch.setattr(Walker, "files", files)


@pytest.mark.parametrize("count", [0, 1, 4, 9, -1])
def test_cancel_then_resume(
    db: Database, root: Root, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, count: int
) -> None:
    for n in range(12):
        write(root.path / f"d{n % 3}" / f"{n}.jpg", b"x" * (n + 1), T0 + n)
    run(db)
    ids = present(db)
    # Changes the rescan has to find: a rename, a deletion, a new file.
    (root.path / "d0" / "0.jpg").rename(root.path / "d1" / "renamed.jpg")
    (root.path / "d2" / "2.jpg").unlink()
    write(root.path / "d2" / "new.jpg", b"new", T0)

    job = JobContext()
    cancel_after(monkeypatch, job, count)
    cancelled = run(db, config=settings(batch_size=2), job=job)
    assert cancelled.state is ScanState.CANCELLED
    assert cancelled.missing == 0
    assert ScansRepo(db).get(cancelled.scan_id).state is ScanState.CANCELLED
    monkeypatch.undo()
    resumed = run(db)
    assert resumed.state is ScanState.DONE

    with Database.open(tmp_path / "fresh") as fresh:
        RootsRepo(fresh).add(root.path)
        run(fresh)
        live = {k: v for k, v in snapshot(db).items() if v[0] == "present"}
        assert live == snapshot(fresh)
    assert present(db)[root.path / "d1" / "renamed.jpg"] == ids[root.path / "d0" / "0.jpg"]
    assert records(db)[root.path / "d2" / "2.jpg"].status is FileStatus.MISSING


@pytest.mark.slow
def test_rescan_of_a_large_tree(db: Database, root: Root) -> None:
    tree = cheap_tree(root.path, 20_000)
    first = run(db)
    assert first.new == tree.count
    second = run(db)
    assert second.unchanged == tree.count
    assert second.new == second.changed == second.moved == 0
