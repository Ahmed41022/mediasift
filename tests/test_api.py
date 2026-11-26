"""The facade: plain results, paging, errors, read-only use."""

from __future__ import annotations

import dataclasses
from collections.abc import Iterator
from pathlib import Path

import pytest

from mediasift import api
from mediasift.api import Library
from mediasift.core.config import JobSettings, ScanSettings, Settings
from mediasift.core.errors import LibraryBusy, LibraryError, SafetyError
from mediasift.db.roots import RootsRepo
from tests.media.tree import Expected, build_tree

KIB = 1024


@pytest.fixture
def tree(tmp_path: Path) -> Expected:
    return build_tree(tmp_path / "media")


@pytest.fixture
def library(tmp_path: Path) -> Iterator[Library]:
    with Library.open(tmp_path / "lib") as lib:
        yield lib


@pytest.fixture
def scanned(library: Library, tree: Expected) -> Library:
    library.add_root(tree.root)
    library.scan()
    library.find_exact()
    return library


def test_open_creates_the_library_and_closes(tmp_path: Path) -> None:
    with Library.open(tmp_path / "new") as lib:
        assert lib.writable and lib.directory == (tmp_path / "new").resolve()
        with pytest.raises(LibraryBusy):
            Library.open(tmp_path / "new")
    Library.open(tmp_path / "new").close()  # the lock was released
    lib.close()  # closing twice is harmless


def test_open_uses_the_home_variable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MEDIASIFT_HOME", str(tmp_path / "home"))
    with Library.open() as lib:
        assert lib.directory == (tmp_path / "home").resolve()


def test_settings_come_from_the_config_file_unless_given(tmp_path: Path, tree: Expected) -> None:
    folder = tmp_path / "lib"
    folder.mkdir()
    (folder / "config.toml").write_text("[scan]\nmin_size = 100000\n", encoding="utf-8")
    with Library.open(folder) as lib:
        assert lib.settings.scan.min_size == 100000
        lib.add_root(tree.root)
        assert lib.scan().seen == 4
    own = Settings(scan=ScanSettings(min_size=0), jobs=JobSettings(workers=1))
    with Library.open(folder, settings=own) as lib:
        assert lib.scan().seen == len(tree.media_paths) + len(tree.empty_files)


def test_bad_config_is_a_config_error(tmp_path: Path) -> None:
    folder = tmp_path / "lib"
    folder.mkdir()
    (folder / "config.toml").write_text("[nonsense]\n", encoding="utf-8")
    with pytest.raises(api.ConfigError):
        Library.open(folder)


def test_roots(library: Library, tmp_path: Path) -> None:
    (tmp_path / "a" / "sub").mkdir(parents=True)
    (tmp_path / "b").mkdir()
    a = library.add_root(tmp_path / "a", prefer_rank=1)
    b = library.add_root(tmp_path / "b", read_only=True)
    assert (a.enabled, a.read_only, a.prefer_rank) == (True, False, 1)
    assert b.read_only and b.id != a.id
    assert library.list_roots() == (a, b)  # the preferred one first
    assert library.find_root(tmp_path / "b") == b
    assert library.find_root(tmp_path / "a" / "sub") is None
    with pytest.raises(SafetyError, match="overlaps"):
        library.add_root(tmp_path / "a" / "sub")
    assert library.disable_root(a.id).enabled is False
    assert library.enable_root(a.id).enabled is True
    with pytest.raises(LibraryError):
        library.disable_root(999)


def test_set_root_volume(library: Library, tmp_path: Path) -> None:
    (tmp_path / "a").mkdir()
    root = library.add_root(tmp_path / "a")
    real = root.volume_id
    assert real
    # Pretend another drive was there when the folder was added.
    RootsRepo(library._db).set_volume_id(root.id, "dev:old")
    assert library.list_roots()[0].volume_id == "dev:old"
    assert library.set_root_volume(root.id).volume_id == real
    with pytest.raises(LibraryError):
        library.set_root_volume(999)


def test_scan_summary_and_rescan(library: Library, tree: Expected) -> None:
    root = library.add_root(tree.root)
    first = library.scan([root.id])
    assert first.state == "done" and not first.cancelled and not first.has_issues
    assert first.seen == first.new == len(tree.media_paths) + len(tree.empty_files)
    assert sum(first.by_type.values()) == first.seen
    again = library.scan()
    assert (again.new, again.changed, again.moved, again.missing) == (0, 0, 0, 0)
    assert again.unchanged == first.seen
    with pytest.raises(dataclasses.FrozenInstanceError):
        first.seen = 0  # type: ignore[misc]


def test_cancelled_scan_is_a_result_not_an_error(library: Library, tree: Expected) -> None:
    library.add_root(tree.root)
    job = api.JobContext()
    job.cancel.cancel()
    summary = library.scan(job=job)
    assert summary.cancelled and summary.state == "cancelled"
    assert library.scan().state == "done"


def test_find_exact_summary(scanned: Library, tree: Expected) -> None:
    again = scanned.find_exact()
    assert again.groups == len(tree.duplicate_groups)
    assert again.reclaimable_bytes == tree.reclaimable_bytes
    assert again.hardlink_sets == 1
    assert again.empty_files == len(tree.empty_files)
    assert (again.hashed_samples, again.hashed_full) == (0, 0)  # all cached
    assert not again.has_issues and again.issues == ()


def test_groups_pages_walk_everything_in_order(scanned: Library, tree: Expected) -> None:
    total = len(tree.duplicate_groups)
    seen: list[api.GroupInfo] = []
    after = None
    pages = 0
    while True:
        page = scanned.list_groups(limit=2, after=after)
        assert 1 <= len(page.groups) <= 2
        seen.extend(page.groups)
        pages += 1
        if page.next is None:
            break
        assert len(page.groups) == 2
        after = page.next
    assert len(seen) == total and pages == (total + 1) // 2
    assert [g.id for g in seen] == [g.id for g in scanned.iter_groups()]
    freed = [g.reclaimable_bytes for g in seen]
    assert freed == sorted(freed, reverse=True)
    assert scanned.list_groups(limit=total).next is None
    with pytest.raises(ValueError, match="limit"):
        scanned.list_groups(limit=0)


def test_group_members_are_plain_and_complete(scanned: Library, tree: Expected) -> None:
    by_paths = {frozenset(g.paths): g for g in tree.duplicate_groups}
    for group in scanned.iter_groups():
        expected = by_paths[frozenset(m.path.resolve() for m in group.members)]
        assert group.kind == "exact" and group.size == expected.size
        assert group.reclaimable_bytes == expected.size * (len(expected.paths) - 1)
        for member in group.members:
            assert member.size == group.size
            assert member.mtime.tzinfo is not None
            assert member.media_type in {"image", "video", "audio"}
            assert member.hardlink_of is None
            assert member.path.is_file()


def test_nothing_from_the_database_leaks(scanned: Library) -> None:
    results = [
        scanned.stats(),
        scanned.list_groups(),
        scanned.scan(),
        scanned.find_exact(),
        *scanned.list_roots(),
        *scanned.stats().files,
    ]
    for result in results:
        assert dataclasses.is_dataclass(result)
        assert type(result).__module__ == "mediasift.api"
        assert dataclasses.fields(result)
    for group in scanned.list_groups().groups:
        assert type(group).__module__ == "mediasift.api"
        assert all(type(m).__module__ == "mediasift.api" for m in group.members)


def test_hard_link_members_point_at_their_owner(library: Library, tmp_path: Path) -> None:
    import os

    folder = tmp_path / "m"
    folder.mkdir()
    (folder / "a.jpg").write_bytes(b"zzz" * 100)
    try:
        os.link(folder / "a.jpg", folder / "b.jpg")
    except OSError:
        pytest.skip("this file system has no hard links")
    (folder / "c.jpg").write_bytes(b"zzz" * 100)
    library.add_root(folder)
    library.scan()
    assert library.find_exact().groups == 1
    (group,) = library.iter_groups()
    ids = {m.file_id for m in group.members}
    linked = [m for m in group.members if m.hardlink_of is not None]
    assert (
        len(linked) == 1
        and linked[0].hardlink_of in ids
        and linked[0].hardlink_of != linked[0].file_id
    )
    assert group.reclaimable_bytes == group.size


def test_stats(scanned: Library, tree: Expected) -> None:
    stats = scanned.stats()
    assert stats.present_files == len(tree.media_paths) + len(tree.empty_files)
    assert stats.present_bytes == sum(p.stat().st_size for p in tree.media_paths + tree.empty_files)
    assert stats.groups == len(tree.duplicate_groups)
    assert stats.reclaimable_bytes == tree.reclaimable_bytes
    assert stats.exact_run_at is not None
    assert stats.last_scan is not None and stats.last_scan.state == "done"
    assert stats.issue_count == 0 and len(stats.roots) == 1
    assert {s.media_type for s in stats.files} == {"image", "video", "audio"}


def test_stats_of_an_empty_library(library: Library) -> None:
    stats = library.stats()
    assert (stats.present_files, stats.groups, stats.last_scan, stats.exact_run_at) == (
        0,
        0,
        None,
        None,
    )
    assert library.issues() == ()


def test_issues_of_the_latest_scan(scanned: Library) -> None:
    RootsRepo(scanned._db).set_volume_id(1, "dev:other")
    summary = scanned.scan()
    assert summary.has_issues and summary.unreachable_roots == (1,)
    (issue,) = scanned.issues()
    assert issue.kind == "root_unreachable" and issue.path == scanned.list_roots()[0].path
    assert scanned.issues(summary.scan_id) == (issue,)
    assert scanned.issues(summary.scan_id - 1) == ()
    assert scanned.stats().issue_count == 1


def test_export_report(scanned: Library, tree: Expected, tmp_path: Path) -> None:
    summary = scanned.export_report(csv=tmp_path / "r.csv", html=tmp_path / "r.html")
    assert summary.groups == len(tree.duplicate_groups)
    assert summary.files == sum(len(g.paths) for g in tree.duplicate_groups)
    assert (tmp_path / "r.csv").is_file() and (tmp_path / "r.html").is_file()
    only = scanned.export_report(html=tmp_path / "only.html")
    assert only.csv is None and only.files == 0
    with pytest.raises(ValueError, match="CSV"):
        scanned.export_report()
    with pytest.raises(api.ActionError):
        scanned.export_report(csv=tmp_path / "missing" / "r.csv")


def test_export_report_needs_a_duplicate_search(library: Library, tmp_path: Path) -> None:
    with pytest.raises(LibraryError, match="dupes"):
        library.export_report(csv=tmp_path / "r.csv")


def test_read_only_library(scanned: Library, tree: Expected, tmp_path: Path) -> None:
    folder = scanned.directory
    scanned.close()
    with Library.open(folder, write=False) as lib:
        assert not lib.writable
        assert lib.stats().groups == len(tree.duplicate_groups)
        assert len(list(lib.iter_groups())) == len(tree.duplicate_groups)
        assert len(lib.list_roots()) == 1
        with pytest.raises(LibraryError, match="reading only"):
            lib.scan()
        with pytest.raises(LibraryError, match="reading only"):
            lib.add_root(tmp_path)
        with pytest.raises(LibraryError, match="reading only"):
            lib.find_exact()
        # A reader does not wait for, or disturb, a writer.
        with Library.open(folder):
            assert lib.stats().groups == len(tree.duplicate_groups)


def test_read_only_needs_an_existing_library(tmp_path: Path) -> None:
    with pytest.raises(LibraryError, match="no library"):
        Library.open(tmp_path / "none", write=False)
    assert not (tmp_path / "none").exists()


def test_the_facade_exports_what_interfaces_need() -> None:
    for name in ("ExitCode", "MediaSiftError", "JobContext", "CancelToken", "Progress"):
        assert name in api.__all__ and hasattr(api, name)
    assert callable(api.cancel_on_sigint)
