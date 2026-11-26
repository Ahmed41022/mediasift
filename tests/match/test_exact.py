from __future__ import annotations

import os
from pathlib import Path

import pytest

from mediasift.core import paths
from mediasift.core.errors import JobCancelled
from mediasift.core.types import (
    FileRecord,
    FileSignature,
    FileStatus,
    MatchGroup,
    MatchKind,
    MediaType,
    Root,
)
from mediasift.db.features import FeaturesRepo
from mediasift.db.files import FoundFile
from mediasift.db.matches import MatchesRepo
from mediasift.db.roots import RootsRepo
from mediasift.hashing.exact import FEATURE_FULL, FULL_VERSION, WHOLE_LIMIT, full_hash
from mediasift.jobs.context import JobContext
from mediasift.match import exact
from mediasift.match.exact import ExactMatcher, ensure_full_hashes
from tests.match.conftest import HashCalls
from tests.match.library import Group, Library, rel_of
from tests.media.tree import build_tree

KIB = 1024
BIG = WHOLE_LIMIT + 50 * KIB


def write(path: Path, data: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def match(lib: Library, workers: int = 2, job: JobContext | None = None) -> exact.ExactResult:
    return ExactMatcher(lib.db, workers=workers).run(job or JobContext())


def group(size: int, *names: str, copies: int | None = None) -> Group:
    return frozenset(names), size * ((copies or len(names)) - 1)


@pytest.fixture
def root(tmp_path: Path, lib: Library) -> Root:
    return lib.add_root(tmp_path / "photos")


# --- the generated library ------------------------------------------------------------


def test_the_planted_groups_are_found(tmp_path: Path, lib: Library) -> None:
    expected = build_tree(tmp_path / "tree")
    root = lib.add_root(tmp_path / "tree")
    lib.index(root)
    result = match(lib)
    top = expected.root
    want = {
        (frozenset(rel_of(p, top) for p in g.paths), g.size * (len(g.paths) - 1))
        for g in expected.duplicate_groups
    }
    assert lib.groups() == want
    assert result.groups == len(want)
    assert result.reclaimable_bytes == expected.reclaimable_bytes
    assert result.hardlink_sets == len(expected.hardlink_groups)
    assert result.empty_files == len(expected.empty_files)
    assert result.issue_count == 0
    # Same size, or same head and tail, is not the same content.
    grouped = {name for names, _ in lib.groups() for name in names}
    for twins in expected.same_size_different + expected.same_head_tail_different_middle:
        assert not grouped & {rel_of(p, top) for p in twins}


def test_a_second_run_computes_nothing(tmp_path: Path, lib: Library, calls: HashCalls) -> None:
    build_tree(tmp_path / "tree")
    lib.index(lib.add_root(tmp_path / "tree"))
    first = match(lib)
    assert calls.partial and calls.full
    calls.clear()
    second = match(lib)
    assert (calls.partial, calls.full) == ([], [])
    assert (second.partial.computed, second.full.computed) == (0, 0)
    assert (second.groups, second.reclaimable_bytes) == (first.groups, first.reclaimable_bytes)
    assert second.run.id != first.run.id


def test_only_what_is_needed_is_hashed(root: Root, lib: Library, calls: HashCalls) -> None:
    same = os.urandom(BIG)
    write(root.path / "a.mp4", same)
    write(root.path / "b.mp4", same)
    write(root.path / "unique size.mp4", os.urandom(BIG + 1))
    write(root.path / "c.mp4", same[:-1] + bytes([same[-1] ^ 1]))  # same size, other tail
    lib.index(root)
    match(lib)
    assert sorted(p.name for p in calls.partial) == ["a.mp4", "b.mp4", "c.mp4"]
    assert sorted(p.name for p in calls.full) == ["a.mp4", "b.mp4"]
    assert lib.groups() == {group(BIG, "a.mp4", "b.mp4")}


def test_one_changed_file_is_the_only_one_hashed_again(
    root: Root, lib: Library, calls: HashCalls
) -> None:
    for size in (100, BIG):
        data = os.urandom(size)
        for name in ("a", "b", "c"):
            write(root.path / f"{size}/{name}.jpg", data)
    lib.index(root)
    match(lib)
    calls.clear()
    changed = root.path / f"{BIG}/b.jpg"
    changed.write_bytes(os.urandom(BIG))
    lib.refresh(root, [f"{BIG}/b.jpg"])
    match(lib)
    assert calls.partial == [changed]
    assert calls.full == []
    assert lib.groups() == {
        group(100, "100/a.jpg", "100/b.jpg", "100/c.jpg"),
        group(BIG, f"{BIG}/a.jpg", f"{BIG}/c.jpg"),
    }


def test_content_matters_not_the_type_or_extension(root: Root, lib: Library) -> None:
    data = os.urandom(5000)
    for name in ("photo.jpg", "clip.MP4", "song.mp3"):
        write(root.path / name, data)
    lib.index(root)
    match(lib)
    assert lib.groups() == {group(5000, "photo.jpg", "clip.MP4", "song.mp3")}


def test_empty_files_are_counted_not_grouped(root: Root, lib: Library) -> None:
    for name in ("a.jpg", "b.jpg", "c.png"):
        write(root.path / name, b"")
    lib.index(root)
    result = match(lib)
    assert lib.groups() == set()
    assert result.empty_files == 3
    # Even with a full hash known for them, they stay out.
    ensure_full_hashes(lib.db, [r.id for r in lib.records(root).values()], JobContext())
    assert match(lib).groups == 0


# --- hard links -----------------------------------------------------------------------


def link_or_skip(source: Path, target: Path) -> None:
    try:
        os.link(source, target)
    except OSError:
        pytest.skip("hard links are not supported here")


@pytest.mark.parametrize("size", [1000, BIG])
def test_names_of_one_file_are_not_duplicates(root: Root, lib: Library, size: int) -> None:
    write(root.path / "a.jpg", os.urandom(size))
    link_or_skip(root.path / "a.jpg", root.path / "link.jpg")
    lib.index(root)
    result = match(lib)
    assert lib.groups() == set()
    assert (result.groups, result.hardlink_sets) == (0, 1)


@pytest.mark.parametrize("size", [1000, BIG])
def test_a_hard_link_and_a_copy(root: Root, lib: Library, size: int) -> None:
    data = os.urandom(size)
    write(root.path / "a.jpg", data)
    link_or_skip(root.path / "a.jpg", root.path / "link.jpg")
    write(root.path / "copy.jpg", data)
    records = lib.index(root)
    result = match(lib)
    assert lib.groups() == {group(size, "a.jpg", "link.jpg", "copy.jpg", copies=2)}
    assert (result.reclaimable_bytes, result.hardlink_sets) == (size, 0)
    [stored] = MatchesRepo(lib.db).iter_groups(MatchKind.EXACT)
    evidence = {m.file_id: dict(m.evidence) for m in stored.members}
    first = min(records["a.jpg"].id, records["link.jpg"].id)
    second = max(records["a.jpg"].id, records["link.jpg"].id)
    assert evidence == {first: {}, second: {"hardlink_of": first}, records["copy.jpg"].id: {}}


def test_an_unknown_identity_is_a_file_of_its_own(root: Root, lib: Library) -> None:
    data = os.urandom(BIG)
    found = []
    for name in ("a.jpg", "b.jpg"):
        sig = paths.signature(write(root.path / name, data))
        unknown = FileSignature(sig.size, sig.mtime_ns)
        found.append(FoundFile(root.id, name, MediaType.IMAGE, unknown))
    lib.files.upsert_many(found, lib.scan_id)
    result = match(lib)
    assert lib.groups() == {group(BIG, "a.jpg", "b.jpg")}
    assert result.hardlink_sets == 0


# --- files that change ----------------------------------------------------------------


@pytest.mark.parametrize("size", [1000, BIG])
def test_a_file_changed_after_hashing_is_left_out(
    root: Root, lib: Library, calls: HashCalls, size: int
) -> None:
    data = os.urandom(size)
    for name in ("a.jpg", "b.jpg", "c.jpg"):
        write(root.path / name, data)
    lib.index(root)
    target = root.path / "b.jpg"

    def change_b_after_its_hashes(kind: str, path: Path) -> None:
        # Files are hashed in id order: when c's last hash starts, b's is done. Change b.
        last = "partial" if size <= WHOLE_LIMIT else "full"
        if kind == last and path.name == "c.jpg":
            before = target.stat().st_mtime_ns
            target.write_bytes(bytes(size))
            # Coarse clocks can give the rewrite the same time; a same-size rewrite within one
            # tick is undetectable here by design (actions hash again), so make it visible.
            os.utime(target, ns=(before + 2_000_000_000, before + 2_000_000_000))

    calls.before = change_b_after_its_hashes
    result = match(lib, workers=1)
    assert target.read_bytes() == bytes(size)
    assert lib.groups() == {group(size, "a.jpg", "c.jpg")}
    b = lib.records(root)["b.jpg"].id
    assert result.check.changed == result.changed == {b}
    assert [i.kind for i in result.issues] == ["changed"]
    # Its hash, valid for the index but not for the disk, keeps it out of later runs too.
    calls.before = None
    assert match(lib).changed == {b}
    assert lib.groups() == {group(size, "a.jpg", "c.jpg")}


@pytest.mark.parametrize("how", ["grow", "truncate", "replace"])
def test_a_file_changed_while_hashed(root: Root, lib: Library, calls: HashCalls, how: str) -> None:
    data = os.urandom(BIG)
    for name in ("a.mp4", "b.mp4", "c.mp4"):
        write(root.path / name, data)
    lib.index(root)
    target = root.path / "b.mp4"

    def alter(kind: str, path: Path) -> None:
        if kind != "full" or path != target:
            return
        if how == "grow":
            with target.open("ab") as f:
                f.write(b"more")
        elif how == "truncate":
            os.truncate(target, 10)
        else:
            other = write(root.path / "new.bin", bytes(BIG))
            st = os.stat(target)
            os.utime(other, ns=(st.st_mtime_ns, st.st_mtime_ns))
            os.replace(other, target)

    calls.before = alter
    result = match(lib)
    b = lib.records(root)["b.mp4"].id
    assert result.changed == {b}
    assert result.full.changed == {b}
    assert lib.groups() == {group(BIG, "a.mp4", "c.mp4")}
    assert FeaturesRepo(lib.db).get(b, FEATURE_FULL, FULL_VERSION) is None


def test_a_symlink_swapped_in_is_not_followed(root: Root, lib: Library, tmp_path: Path) -> None:
    data = os.urandom(BIG)
    for name in ("a.jpg", "b.jpg"):
        write(root.path / name, data)
    lib.index(root)
    outside = write(tmp_path / "outside.jpg", data)
    target = root.path / "b.jpg"
    st = os.stat(target)
    os.utime(outside, ns=(st.st_mtime_ns, st.st_mtime_ns))
    target.unlink()
    try:
        target.symlink_to(outside)
    except OSError:
        pytest.skip("cannot create symlinks here")
    result = match(lib)
    assert lib.groups() == set()
    assert result.changed == {lib.records(root)["b.jpg"].id}


def test_placeholders_are_never_opened(
    root: Root, lib: Library, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = os.urandom(BIG)
    for name in ("a.jpg", "b.jpg", "cloud.jpg"):
        write(root.path / name, data)
    lib.index(root)
    cloud = os.lstat(root.path / "cloud.jpg")
    real_open = os.open
    monkeypatch.setattr(paths, "is_placeholder", lambda st: st.st_ino == cloud.st_ino)

    def guarded_open(name: str, *args: int) -> int:
        assert not name.endswith("cloud.jpg"), "a placeholder was opened"
        return real_open(name, *args)

    monkeypatch.setattr(os, "open", guarded_open)
    result = match(lib)
    assert lib.groups() == {group(BIG, "a.jpg", "b.jpg")}
    assert [i.kind for i in result.issues] == ["placeholder"]
    # Not stored as an error: it is tried again once it is back.
    monkeypatch.undo()
    match(lib)
    assert lib.groups() == {group(BIG, "a.jpg", "b.jpg", "cloud.jpg")}


def test_an_unreadable_file_is_an_issue_and_tried_again(
    root: Root, lib: Library, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = os.urandom(BIG)
    for name in ("a.jpg", "b.jpg", "locked.jpg"):
        write(root.path / name, data)
    lib.index(root)
    real_open = os.open

    def deny(name: str, *args: int) -> int:
        if name.endswith("locked.jpg"):
            raise PermissionError(13, "Permission denied", name)
        return real_open(name, *args)

    monkeypatch.setattr(os, "open", deny)
    result = match(lib)
    assert lib.groups() == {group(BIG, "a.jpg", "b.jpg")}
    [issue] = result.issues
    assert (issue.kind, issue.permanent) == ("permission", False)
    assert FeaturesRepo(lib.db).errors("partial", 1) == []
    monkeypatch.undo()
    assert match(lib).issue_count == 0
    assert lib.groups() == {group(BIG, "a.jpg", "b.jpg", "locked.jpg")}


# --- scope ----------------------------------------------------------------------------


def test_disabled_roots_and_missing_files_are_left_alone(
    tmp_path: Path, lib: Library, calls: HashCalls
) -> None:
    data = os.urandom(BIG)
    first, second = lib.add_root(tmp_path / "one"), lib.add_root(tmp_path / "two")
    for name in ("a.jpg", "b.jpg", "c.jpg"):
        write(first.path / name, data)
    write(second.path / "d.jpg", data)
    lib.index(first)
    lib.index(second)
    gone = lib.records(first)["b.jpg"]
    lib.files.set_status([gone.id], FileStatus.MISSING)
    RootsRepo(lib.db).set_enabled(second.id, False)
    match(lib)
    assert sorted(p.name for p in calls.partial) == ["a.jpg", "c.jpg"]
    assert sorted(p.name for p in calls.full) == ["a.jpg", "c.jpg"]
    assert lib.groups() == {group(BIG, "a.jpg", "c.jpg")}
    RootsRepo(lib.db).set_enabled(second.id, True)
    match(lib)
    assert lib.groups() == {group(BIG, "a.jpg", "c.jpg", "d.jpg")}


def test_long_and_arabic_paths(root: Root, lib: Library) -> None:
    folder = root.path
    for n in range(5):
        folder = folder / f"مجلد {n} {'y' * 50}"
    os.makedirs(paths.fs_path(folder))
    data = os.urandom(BIG)
    for name in ("صورة.jpg", "نسخة.jpg"):
        with open(paths.fs_path(folder / name), "wb") as f:
            f.write(data)
    assert len(str(folder / "صورة.jpg")) > 300
    lib.index(root)
    match(lib)
    [(names, freed)] = lib.groups()
    assert {Path(n).name for n in names} == {"صورة.jpg", "نسخة.jpg"}
    assert freed == BIG


# --- cancel ---------------------------------------------------------------------------


def test_cancel_keeps_the_previous_groups_and_resumes(
    root: Root, lib: Library, calls: HashCalls
) -> None:
    data = os.urandom(BIG)
    write(root.path / "old/a.jpg", data)
    write(root.path / "old/b.jpg", data)
    lib.index(root)
    before = match(lib, workers=1)
    for n in range(6):
        content = os.urandom(3000 + n)
        write(root.path / f"new/{n}a.jpg", content)
        write(root.path / f"new/{n}b.jpg", content)
    lib.index(root)
    calls.clear()
    job = JobContext()

    def cancel_before_the_fourth(kind: str, path: Path) -> None:
        if len(calls.partial) == 4:
            job.cancel.cancel()

    calls.before = cancel_before_the_fourth
    with pytest.raises(JobCancelled):
        match(lib, workers=1, job=job)
    run = MatchesRepo(lib.db).latest_run(MatchKind.EXACT)
    assert run is not None and run.id == before.run.id
    assert lib.groups() == {group(BIG, "old/a.jpg", "old/b.jpg")}
    assert len(calls.partial) == 4
    done = calls.partial[:3]  # the fourth saw the cancel and stopped
    calls.clear()
    calls.before = None
    result = match(lib, workers=1)
    assert not set(done) & set(calls.partial)
    assert len(done) + len(calls.partial) == 12
    assert result.groups == 7


def test_cancel_while_grouping_keeps_the_previous_run(
    root: Root, lib: Library, monkeypatch: pytest.MonkeyPatch
) -> None:
    for n in range(3):
        content = os.urandom(100 + n)
        write(root.path / f"{n}a.jpg", content)
        write(root.path / f"{n}b.jpg", content)
    lib.index(root)
    before = match(lib)
    job = JobContext()
    real = exact._match_group
    built: list[object] = []

    def cancel_on_second(records: list[FileRecord]) -> MatchGroup | None:
        built.append(records)
        if len(built) == 2:
            job.cancel.cancel()
        return real(records)

    monkeypatch.setattr(exact, "_match_group", cancel_on_second)
    with pytest.raises(JobCancelled):
        match(lib, job=job)
    run = MatchesRepo(lib.db).latest_run(MatchKind.EXACT)
    assert run is not None and run.id == before.run.id
    assert MatchesRepo(lib.db).count_groups(MatchKind.EXACT)[0] == 3
    assert len(built) == 3  # the last group was built, yet nothing was stored


# --- full hashes for actions ----------------------------------------------------------


def test_ensure_full_hashes(tmp_path: Path, lib: Library, calls: HashCalls) -> None:
    root, other = lib.add_root(tmp_path / "one"), lib.add_root(tmp_path / "two")
    data = os.urandom(BIG)
    for folder in (root, other):
        write(folder.path / "a.jpg", data)
        write(folder.path / "b.jpg", os.urandom(10))
        write(folder.path / "gone.jpg", data)
        lib.index(folder)
    records = lib.records(root)
    lib.files.set_status([records["gone.jpg"].id], FileStatus.MISSING)
    RootsRepo(lib.db).set_enabled(other.id, False)
    disabled = lib.records(other)["a.jpg"].id
    ids = [records["a.jpg"].id, records["b.jpg"].id, records["gone.jpg"].id, disabled, 9999]
    found = ensure_full_hashes(lib.db, ids, JobContext())
    a = records["a.jpg"]
    assert found.values == {
        a.id: full_hash(a.path, a.signature),
        records["b.jpg"].id: full_hash(records["b.jpg"].path, records["b.jpg"].signature),
    }
    assert found.computed == 2
    assert {(i.file_id, i.kind) for i in found.issues} == {
        (records["gone.jpg"].id, "missing"),
        (disabled, "disabled"),
    }
    assert {p.name for p in calls.full} == {"a.jpg", "b.jpg"}
    calls.clear()
    again = ensure_full_hashes(lib.db, ids, JobContext())
    assert (again.values, again.computed, calls.full) == (found.values, 0, [])
