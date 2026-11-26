"""Every command and flag, human output and --json, exit codes."""

from __future__ import annotations

import logging
import os
import signal
from pathlib import Path
from typing import Any

import pytest

from mediasift.api import Library
from mediasift.cli.main import cli
from mediasift.core.errors import ExitCode
from mediasift.db.connection import Database
from mediasift.db.roots import RootsRepo
from tests.cli.conftest import Cli, listed_groups, norm, planted_groups
from tests.media.tree import ARABIC_DIR, Expected

KIB = 1024


# --- roots ---


def test_roots_add_list_disable_enable(cli_run: Cli, tmp_path: Path) -> None:
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    added = cli_run.doc("roots add", str(tmp_path / "a"), "--read-only", "--rank", "5")
    assert added["root"]["read_only"] is True
    assert added["root"]["prefer_rank"] == 5
    assert added["root"]["enabled"] is True
    cli_run.run("roots", "add", str(tmp_path / "b"))

    listing = cli_run.doc("roots list")
    assert [norm(r["path"]) for r in listing["roots"]] == [
        norm(tmp_path / "a"),
        norm(tmp_path / "b"),
    ]

    assert cli_run.doc("roots disable", "1")["root"]["enabled"] is False
    human = cli_run.run("roots", "list").stdout
    assert "disabled" in human and "read-only" in human
    assert cli_run.doc("roots enable", "1")["root"]["enabled"] is True
    assert "Enabled folder" in cli_run.run("roots", "enable", "1").stdout


def test_roots_list_empty(cli_run: Cli, tmp_path: Path) -> None:
    (tmp_path / "a").mkdir()
    cli_run.run("roots", "add", str(tmp_path / "a"))
    cli_run.run("roots", "disable", "1")
    assert cli_run.doc("roots list")["roots"][0]["enabled"] is False


def test_roots_overlap_is_refused_with_exit_4(cli_run: Cli, tmp_path: Path) -> None:
    (tmp_path / "a" / "inner").mkdir(parents=True)
    cli_run.run("roots", "add", str(tmp_path / "a"))
    result = cli_run.run("roots", "add", str(tmp_path / "a" / "inner"), code=ExitCode.REFUSED)
    assert "overlaps" in result.stderr
    assert result.stdout == ""
    document = cli_run.doc("roots add", str(tmp_path / "a" / "inner"), code=ExitCode.REFUSED)
    assert document["error"]["exit_code"] == 4
    assert "overlaps" in document["error"]["message"]


def test_roots_add_missing_folder_fails(cli_run: Cli, tmp_path: Path) -> None:
    result = cli_run.run("roots", "add", str(tmp_path / "nope"), code=ExitCode.FAILURE)
    assert "not a folder" in result.stderr


def test_roots_disable_unknown_id(cli_run: Cli, tmp_path: Path) -> None:
    (tmp_path / "a").mkdir()
    cli_run.run("roots", "add", str(tmp_path / "a"))
    result = cli_run.run("roots", "disable", "99", code=ExitCode.FAILURE)
    assert "99" in result.stderr
    assert "Traceback" not in result.stderr


def test_accept_volume_after_the_drive_changed(cli_run: Cli, tree: Expected, libdir: Path) -> None:
    first = cli_run.doc("scan", str(tree.root))
    with Database.open(libdir) as db:
        RootsRepo(db).set_volume_id(1, "dev:not-this-drive")
    # A scan refuses the folder instead of treating every file as deleted.
    result = cli_run.run("scan", code=ExitCode.PARTIAL)
    assert "another volume" in result.stderr
    document = cli_run.doc("scan", code=ExitCode.PARTIAL)
    assert document["unreachable_roots"] == [1]
    assert document["missing"] == 0

    accepted = cli_run.doc("roots accept-volume", "1")
    assert accepted["root"]["volume_id"] != "dev:not-this-drive"
    clean = cli_run.doc("scan")
    assert clean["unreachable_roots"] == []
    assert clean["unchanged"] == first["files_seen"]


# --- scan ---


def test_scan_a_folder_adds_it_then_rescan_changes_nothing(cli_run: Cli, tree: Expected) -> None:
    first = cli_run.doc("scan", str(tree.root))
    assert first["state"] == "done"
    assert first["files_seen"] == len(tree.media_paths) + len(tree.empty_files)
    assert first["new"] == first["files_seen"]
    assert first["by_type"]["image"] > 0

    again = cli_run.doc("scan", "1")
    assert (again["new"], again["changed"], again["moved"], again["missing"]) == (0, 0, 0, 0)
    assert again["unchanged"] == first["files_seen"]

    human = cli_run.run("scan")
    assert "0 new" in human.stdout


def test_scan_notes_the_added_folder_on_stderr(cli_run: Cli, tree: Expected) -> None:
    result = cli_run.run("scan", str(tree.root))
    assert "Added folder 1" in result.stderr
    assert "files seen" in result.stdout


def test_scan_without_roots_is_a_usage_error(cli_run: Cli) -> None:
    result = cli_run.run("scan", code=ExitCode.USAGE)
    assert "roots add" in result.stderr


def test_scan_unknown_root_id(cli_run: Cli, scanned: Expected) -> None:
    result = cli_run.run("scan", "42", code=ExitCode.FAILURE)
    assert "42" in result.stderr


def test_scan_picks_up_new_and_missing_files(cli_run: Cli, scanned: Expected) -> None:
    (scanned.root / "new one.jpg").write_bytes(b"x" * 50)
    gone = scanned.exact_groups[0].paths[0]
    gone.unlink()
    document = cli_run.doc("scan")
    assert document["new"] == 1
    assert document["missing"] == 1


def test_scan_with_issues_exits_3_and_lists_them(
    cli_run: Cli, scanned: Expected, libdir: Path
) -> None:
    with Database.open(libdir) as db:
        RootsRepo(db).set_volume_id(1, "dev:other")
    result = cli_run.run("scan", code=ExitCode.PARTIAL)
    assert "problem" in result.stderr
    assert "root_unreachable" in result.stderr
    document = cli_run.doc("scan", code=ExitCode.PARTIAL)
    assert document["errors"] == 1
    assert document["issues"][0]["kind"] == "root_unreachable"


def test_scan_ctrl_c_cancels_cleanly_with_130(
    cli_run: Cli, tree: Expected, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_scan = Library.scan

    def interrupted(self: Library, roots: Any = None, job: Any = None) -> Any:
        signal.raise_signal(signal.SIGINT)  # what Ctrl+C does, through the installed handler
        return real_scan(self, roots, job)

    monkeypatch.setattr(Library, "scan", interrupted)
    result = cli_run.run("scan", str(tree.root), code=ExitCode.CANCELLED)
    assert "Cancelled" in result.stderr
    assert "kept" in result.stderr
    assert "files seen" in result.stdout
    # The folder was added before the interruption, and the library is intact.
    monkeypatch.undo()
    assert cli_run.doc("scan", "1")["state"] == "done"


def test_scan_ctrl_c_json_document_says_cancelled(
    cli_run: Cli, tree: Expected, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_scan = Library.scan

    def interrupted(self: Library, roots: Any = None, job: Any = None) -> Any:
        signal.raise_signal(signal.SIGINT)
        return real_scan(self, roots, job)

    monkeypatch.setattr(Library, "scan", interrupted)
    document = cli_run.doc("scan", str(tree.root), code=ExitCode.CANCELLED)
    assert document["state"] == "cancelled"


# --- dupes ---


def test_dupes_lists_exactly_the_planted_groups(cli_run: Cli, scanned: Expected) -> None:
    document = cli_run.doc("dupes", "--all")
    assert listed_groups(document) == planted_groups(scanned)
    assert document["more"] is False
    assert document["exact"]["groups"] == len(scanned.duplicate_groups)
    assert document["exact"]["reclaimable_bytes"] == scanned.reclaimable_bytes
    assert document["exact"]["hardlink_sets"] == 1
    assert document["exact"]["empty_files"] == len(scanned.empty_files)
    numbers = [g["number"] for g in document["groups"]]
    assert numbers == list(range(1, len(numbers) + 1))
    freed = [g["reclaimable_bytes"] for g in document["groups"]]
    assert freed == sorted(freed, reverse=True)


def test_second_dupes_hashes_nothing(cli_run: Cli, scanned: Expected) -> None:
    first = cli_run.doc("dupes")
    assert first["exact"]["hashed_samples"] > 0
    second = cli_run.doc("dupes")
    assert second["exact"]["hashed_samples"] == 0
    assert second["exact"]["hashed_full"] == 0
    assert second["exact"]["groups"] == first["exact"]["groups"]


def test_dupes_human_output(cli_run: Cli, scanned: Expected) -> None:
    result = cli_run.run("dupes", "--all")
    lines = result.stdout.splitlines()
    assert lines[0].startswith(f"{len(scanned.duplicate_groups)} groups of identical files")
    assert "Group 1: 2 files of 300.0 KiB, 300.0 KiB can be freed" in result.stdout
    for group in scanned.duplicate_groups:
        for path in group.paths:
            assert any(norm(line.strip()) == norm(path) for line in lines), path


def test_dupes_limit_all_and_more(cli_run: Cli, scanned: Expected) -> None:
    two = cli_run.doc("dupes", "--limit", "2")
    assert two["shown"] == 2 and two["more"] is True
    human = cli_run.run("dupes", "--limit", "2")
    assert "Group 2:" in human.stdout and "Group 3:" not in human.stdout
    assert "--all" in human.stderr
    everything = cli_run.doc("dupes", "--limit", "2", "--all")
    assert everything["shown"] == len(scanned.duplicate_groups) and everything["more"] is False
    exact = cli_run.doc("dupes", "--limit", str(len(scanned.duplicate_groups)))
    assert exact["more"] is False


def test_dupes_min_size_skips_small_files(cli_run: Cli, scanned: Expected) -> None:
    document = cli_run.doc("dupes", "--all", "--min-size", "100k")
    assert [g["size"] for g in document["groups"]] == [300 * KIB]
    assert cli_run.doc("dupes", "--all", "--min-size", "1MB")["groups"] == []
    sizes = cli_run.doc("dupes", "--all", "--min-size", "9216")["groups"]
    assert min(g["size"] for g in sizes) == 9 * KIB


def test_dupes_marks_hard_links(cli_run: Cli, tmp_path: Path) -> None:
    folder = tmp_path / "media"
    folder.mkdir()
    original = folder / "a.jpg"
    original.write_bytes(b"same content" * 100)
    try:
        os.link(original, folder / "a link.jpg")
    except OSError:
        pytest.skip("this file system has no hard links")
    (folder / "copy.jpg").write_bytes(b"same content" * 100)
    cli_run.run("scan", str(folder))

    document = cli_run.doc("dupes")
    (group,) = document["groups"]
    assert len(group["files"]) == 3
    linked = [f for f in group["files"] if f["hardlink_of"] is not None]
    assert len(linked) == 1
    assert group["reclaimable_bytes"] == group["size"]  # the extra name frees nothing
    human = cli_run.run("dupes").stdout
    assert "hard link of" in human
    assert human.count("hard link of") == 1


def test_dupes_without_scan_finds_nothing(cli_run: Cli) -> None:
    document = cli_run.doc("dupes")
    assert document["groups"] == [] and document["exact"]["groups"] == 0


def test_dupes_arabic_paths_stay_readable_in_json(cli_run: Cli, scanned: Expected) -> None:
    result = cli_run.run("--json", "dupes", "--all")
    assert ARABIC_DIR in result.stdout  # not \uXXXX escapes
    assert "\\u" not in result.stdout
    assert any(
        ARABIC_DIR in f["path"] for g in cli_run.doc("dupes", "--all")["groups"] for f in g["files"]
    )


def test_dupes_human_output_has_arabic_paths(cli_run: Cli, scanned: Expected) -> None:
    assert ARABIC_DIR in cli_run.run("dupes", "--all").stdout


def test_dupes_cancelled_exits_130(
    cli_run: Cli, scanned: Expected, monkeypatch: pytest.MonkeyPatch
) -> None:
    real = Library.find_exact

    def interrupted(self: Library, job: Any = None) -> Any:
        signal.raise_signal(signal.SIGINT)
        return real(self, job)

    monkeypatch.setattr(Library, "find_exact", interrupted)
    result = cli_run.run("dupes", code=ExitCode.CANCELLED)
    assert "Cancelled" in result.stderr
    document = cli_run.doc("dupes", code=ExitCode.CANCELLED)
    assert document["cancelled"] is True


@pytest.mark.parametrize(
    "args",
    [
        ["dupes", "--limit", "0"],
        ["dupes", "--limit", "many"],
        ["dupes", "--min-size", "huge"],
        ["roots", "add"],
        ["roots", "disable", "x"],
        ["report"],
        ["nonsense"],
    ],
)
def test_usage_errors_exit_2_without_a_document(
    cli_run: Cli, scanned: Expected, args: list[str]
) -> None:
    result = cli_run.run("--json", *args, code=ExitCode.USAGE)
    assert result.stdout == ""
    assert "Error" in result.stderr or "error" in result.stderr


def test_dupes_changed_file_exits_3(cli_run: Cli, scanned: Expected) -> None:
    victim = scanned.exact_groups[0].paths[0]
    victim.write_bytes(victim.read_bytes() + b"more")  # changed after the scan
    document = cli_run.doc("dupes", code=ExitCode.PARTIAL)
    assert document["exact"]["changed_files"] >= 1
    result = cli_run.run("dupes", code=ExitCode.PARTIAL)
    assert "changed" in result.stderr


# --- stats ---


def test_stats_json_and_human(cli_run: Cli, scanned: Expected) -> None:
    before = cli_run.doc("stats")
    assert before["duplicates_searched_at"] is None
    assert before["last_scan"]["state"] == "done"
    assert before["issues"] == 0
    cli_run.run("dupes")
    document = cli_run.doc("stats")
    assert document["duplicate_groups"] == len(scanned.duplicate_groups)
    assert document["reclaimable_bytes"] == scanned.reclaimable_bytes
    assert document["duplicates_searched_at"] is not None
    present = sum(f["count"] for f in document["files"] if f["status"] == "present")
    assert present == len(scanned.media_paths) + len(scanned.empty_files)
    assert {f["media_type"] for f in document["files"]} == {"image", "video", "audio"}
    assert [norm(r["path"]) for r in document["roots"]] == [norm(scanned.root)]
    human = cli_run.run("stats").stdout
    assert "Duplicates:" in human and "Last scan" in human and "image" in human


def test_stats_of_a_library_that_does_not_exist(cli_run: Cli) -> None:
    result = cli_run.run("stats", code=ExitCode.FAILURE)
    assert "no library" in result.stderr
    assert "Traceback" not in result.stderr


def test_stats_works_while_another_process_writes(
    cli_run: Cli, scanned: Expected, libdir: Path
) -> None:
    with Database.open(libdir):
        assert cli_run.doc("stats")["last_scan"] is not None


# --- busy library ---


def test_busy_library_exits_5(cli_run: Cli, scanned: Expected, libdir: Path) -> None:
    with Database.open(libdir):  # another writer
        result = cli_run.run("scan", code=ExitCode.BUSY)
        assert "another process" in result.stderr
        assert "Traceback" not in result.stderr
        document = cli_run.doc("dupes", code=ExitCode.BUSY)
        assert document["error"]["type"] == "LibraryBusy"
        cli_run.run("roots", "disable", "1", code=ExitCode.BUSY)
    cli_run.run("scan")  # free again


# --- report ---


def test_report_csv_and_html(cli_run: Cli, scanned: Expected, tmp_path: Path) -> None:
    cli_run.run("dupes")
    csv_file, html_file = tmp_path / "out" / "r.csv", tmp_path / "r.html"
    csv_file.parent.mkdir()
    document = cli_run.doc("report", "--csv", str(csv_file), "--html", str(html_file))
    assert document["groups"] == len(scanned.duplicate_groups)
    assert document["files"] == sum(len(g.paths) for g in scanned.duplicate_groups)
    assert norm(document["csv"]) == norm(csv_file)
    text = csv_file.read_text(encoding="utf-8-sig")
    assert text.splitlines()[0].startswith("group,kind,file_size_bytes")
    assert ARABIC_DIR in text
    page = html_file.read_text(encoding="utf-8")
    assert page.startswith("<!DOCTYPE html>") and ARABIC_DIR in page
    assert 'dir="auto"' in page
    human = cli_run.run("report", "--csv", str(csv_file))
    assert "Wrote" in human.stderr and "groups" in human.stdout


def test_report_before_dupes_is_refused(cli_run: Cli, scanned: Expected, tmp_path: Path) -> None:
    result = cli_run.run("report", "--csv", str(tmp_path / "r.csv"), code=ExitCode.FAILURE)
    assert "dupes" in result.stderr
    assert not (tmp_path / "r.csv").exists()


def test_report_to_a_missing_folder_fails_cleanly(
    cli_run: Cli, scanned: Expected, tmp_path: Path
) -> None:
    cli_run.run("dupes")
    result = cli_run.run(
        "report", "--html", str(tmp_path / "nope" / "r.html"), code=ExitCode.FAILURE
    )
    assert "cannot write" in result.stderr


def test_report_escapes_awkward_file_names(cli_run: Cli, tmp_path: Path) -> None:
    folder = tmp_path / "media"
    folder.mkdir()
    names = ["<b>&'\".jpg", "=cmd.jpg"] if os.name != "nt" else ["&'x&amp;.jpg", "=cmd.jpg"]
    for name in names:
        (folder / name).write_bytes(b"data" * 50)
    (folder / "plain.jpg").write_bytes(b"data" * 50)
    cli_run.run("scan", str(folder))
    cli_run.run("dupes")
    cli_run.run("report", "--html", str(tmp_path / "r.html"), "--csv", str(tmp_path / "r.csv"))
    page = (tmp_path / "r.html").read_text(encoding="utf-8")
    assert "<b>" not in page
    expected = "&lt;b&gt;&amp;&#x27;&quot;.jpg" if os.name != "nt" else "&amp;&#x27;x&amp;amp;.jpg"
    assert expected in page
    csv_text = (tmp_path / "r.csv").read_text(encoding="utf-8-sig")
    assert str(folder / "=cmd.jpg") in csv_text  # a path starts with the folder, not with "="


# --- options ---


def test_verbose_logs_to_stderr(cli_run: Cli, scanned: Expected) -> None:
    try:
        result = cli_run.run("-v", "dupes")
        assert "exact:" in result.stderr
        assert "exact:" not in result.stdout
    finally:
        for handler in list(logging.getLogger("mediasift").handlers):
            if getattr(handler, "_console", False):
                logging.getLogger("mediasift").removeHandler(handler)


def test_unexpected_error_hides_traceback_unless_debug(
    cli_run: Cli, scanned: Expected, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken(self: Library) -> Any:
        raise RuntimeError("boom")

    monkeypatch.setattr(Library, "stats", broken)
    quiet = cli_run.run("stats", code=ExitCode.FAILURE)
    assert "boom" in quiet.stderr and "Traceback" not in quiet.stderr
    loud = cli_run.runner.invoke(cli, ["--library", str(cli_run.library), "--debug", "stats"])
    assert isinstance(loud.exception, RuntimeError)


def test_library_from_environment(
    tmp_path: Path, tree: Expected, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MEDIASIFT_HOME", str(tmp_path / "from-env"))
    runner = Cli(tmp_path / "unused")
    runner.run("scan", str(tree.root), library=False)
    assert any((tmp_path / "from-env").glob("*.db"))


def test_config_file_applies_to_the_scan(cli_run: Cli, tree: Expected, libdir: Path) -> None:
    libdir.mkdir()
    (libdir / "config.toml").write_text("[scan]\nmin_size = 100000\n", encoding="utf-8")
    document = cli_run.doc("scan", str(tree.root))
    # Only files of 100000 bytes or more: the two raw files and the two big videos.
    assert document["files_seen"] == 4


def test_bad_config_exits_2(cli_run: Cli, tree: Expected, libdir: Path) -> None:
    libdir.mkdir()
    (libdir / "config.toml").write_text("[scan]\nmin_size = 'x'\n", encoding="utf-8")
    result = cli_run.run("scan", str(tree.root), code=ExitCode.USAGE)
    assert "min_size" in result.stderr


def test_help_and_version(cli_run: Cli) -> None:
    helped = cli_run.run("--help", library=False)
    for command in ("roots", "scan", "dupes", "stats", "report", "--json", "--library"):
        assert command in helped.stdout
    assert "mediasift" in cli_run.run("--version", library=False).stdout
