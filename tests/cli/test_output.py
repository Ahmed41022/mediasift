"""Real processes (module entry point, legacy code pages), report writers, progress bar."""

from __future__ import annotations

import csv
import io
import json
import os
import signal
import subprocess
import sys
from pathlib import Path

import pytest
from rich.console import Console

import mediasift
from mediasift.api import Progress
from mediasift.cli.progress import ProgressBar, job_context
from mediasift.report import ReportGroup, ReportMember, human_size, write_csv, write_html
from tests.media.tree import ARABIC_DIR, Expected

SRC = str(Path(mediasift.__file__).resolve().parent.parent)


def mediasift_process(
    library: Path, *args: str, encoding: str | None = None
) -> subprocess.CompletedProcess[bytes]:
    env = {**os.environ, "PYTHONPATH": SRC, "PYTHONUTF8": "0"}
    if encoding:
        env["PYTHONIOENCODING"] = encoding
    return subprocess.run(
        [sys.executable, "-m", "mediasift", "--library", str(library), *args],
        capture_output=True,
        env=env,
        timeout=120,
        check=False,
    )


# --- real processes ---


def test_python_dash_m_runs_the_whole_flow(tmp_path: Path, tree: Expected) -> None:
    library = tmp_path / "lib"
    version = subprocess.run(
        [sys.executable, "-m", "mediasift", "--version"],
        capture_output=True,
        env={**os.environ, "PYTHONPATH": SRC},
        timeout=60,
        check=False,
    )
    assert version.returncode == 0 and b"mediasift" in version.stdout

    assert mediasift_process(library, "scan", str(tree.root)).returncode == 0
    done = mediasift_process(library, "--json", "dupes", "--all")
    assert done.returncode == 0
    document = json.loads(done.stdout.decode("utf-8"))
    assert document["exact"]["groups"] == len(tree.duplicate_groups)


def test_legacy_code_page_does_not_crash(tmp_path: Path, tree: Expected) -> None:
    library = tmp_path / "lib"
    assert mediasift_process(library, "scan", str(tree.root), encoding="cp1252").returncode == 0

    human = mediasift_process(library, "dupes", "--all", encoding="cp1252")
    assert human.returncode == 0, human.stderr
    text = human.stdout.decode("cp1252")
    assert "Group 1:" in text
    assert ARABIC_DIR not in text  # cannot be shown in cp1252 ...
    assert "?" in text  # ... and is replaced, not fatal

    listing = mediasift_process(library, "roots", "list", encoding="cp1252")
    assert listing.returncode == 0 and b"?" not in listing.stderr

    # JSON loses nothing: the same document, with escapes where the console cannot show text.
    as_json = mediasift_process(library, "--json", "dupes", "--all", encoding="cp1252")
    assert as_json.returncode == 0
    text = as_json.stdout.decode("ascii")
    assert "\\u" in text
    paths = [f["path"] for g in json.loads(text)["groups"] for f in g["files"]]
    assert any(ARABIC_DIR in p for p in paths)


def test_error_message_with_arabic_on_a_legacy_console(tmp_path: Path) -> None:
    result = mediasift_process(
        tmp_path / "lib", "roots", "add", str(tmp_path / ARABIC_DIR / "x"), encoding="cp1252"
    )
    assert result.returncode == 1
    assert b"error:" in result.stderr and b"Traceback" not in result.stderr


# --- report writers ---


def awkward_group() -> ReportGroup:
    nasty = "<script>alert(1)</script>.jpg"
    members = (
        ReportMember(f"/photos/{nasty}", 2048, "2024-05-01T10:00:00+00:00", "image"),
        ReportMember(
            "/photos/صورة & \"quotes\" 'x'.jpg",
            2048,
            "2024-05-01T10:00:00+00:00",
            "image",
            hardlink_of=f"/photos/{nasty}",
        ),
    )
    return ReportGroup(1, "exact", 2048, 2048, members)


def test_html_is_escaped_self_contained_and_bidi_safe(tmp_path: Path) -> None:
    target = tmp_path / "r.html"
    assert write_html(target, [awkward_group()]) == 1
    page = target.read_text(encoding="utf-8")
    assert "<script>" not in page
    assert "&lt;script&gt;alert(1)&lt;/script&gt;.jpg" in page
    assert "&quot;quotes&quot;" in page and "&amp;" in page
    assert page.count('dir="auto"') >= 3
    assert "صورة" in page
    for external in ("http://", "https://", "src=", "<link", "<script", "@import", "url("):
        assert external not in page
    assert "hard link of" in page
    assert page.rstrip().endswith("</html>")


def test_html_without_groups_says_so(tmp_path: Path) -> None:
    write_html(tmp_path / "r.html", [])
    assert "No duplicate files" in (tmp_path / "r.html").read_text(encoding="utf-8")


def test_csv_round_trips_and_defuses_formulas(tmp_path: Path) -> None:
    target = tmp_path / "r.csv"
    group = ReportGroup(
        7,
        "exact",
        10,
        10,
        (
            ReportMember('=HYPERLINK("http://x")', 10, "t", "image"),
            ReportMember('a,b "c"\nd.jpg', 10, "t", "image", hardlink_of="@x"),
        ),
    )
    assert write_csv(target, [group]) == 2
    raw = target.read_bytes()
    assert raw.startswith(b"\xef\xbb\xbf")
    rows = list(csv.reader(io.StringIO(raw.decode("utf-8-sig"), newline="")))
    assert rows[0][0] == "group" and rows[1][0] == "7"
    assert rows[1][4] == '\'=HYPERLINK("http://x")'
    assert rows[2][4] == 'a,b "c"\nd.jpg'
    assert rows[2][7] == "'@x"


def test_a_failed_report_keeps_the_old_file(tmp_path: Path) -> None:
    target = tmp_path / "r.csv"
    target.write_text("old", encoding="utf-8")

    def broken() -> object:
        yield awkward_group()
        raise RuntimeError("disk full")

    with pytest.raises(RuntimeError):
        write_csv(target, broken())  # type: ignore[arg-type]
    assert target.read_text(encoding="utf-8") == "old"
    assert [p.name for p in tmp_path.iterdir()] == ["r.csv"]


def test_human_size() -> None:
    assert [human_size(n) for n in (0, 1023, 1024, 1536, 5 * 1024**3)] == [
        "0 B",
        "1023 B",
        "1.0 KiB",
        "1.5 KiB",
        "5.0 GiB",
    ]


# --- progress ---


def test_progress_bar_draws_stage_names_on_a_terminal() -> None:
    stream = io.StringIO()
    console = Console(file=stream, force_terminal=True, width=100, color_system=None)
    with ProgressBar(True, console) as bar:
        bar.update(Progress("scan", done=5, total=10))
        bar.update(Progress("partial hashes", done=1, total=None))
        bar.update(Progress("scan", done=10, total=10))
    assert "scan" in stream.getvalue()


def test_progress_bar_is_silent_when_disabled() -> None:
    stream = io.StringIO()
    with ProgressBar(False) as bar:
        bar.update(Progress("scan", done=1, total=2))
    assert stream.getvalue() == ""


def test_job_context_is_cancelled_by_sigint_and_restores_the_handler() -> None:
    before = signal.getsignal(signal.SIGINT)
    with job_context(show_bar=False) as job:
        assert not job.cancelled
        signal.raise_signal(signal.SIGINT)
        assert job.cancelled
    assert signal.getsignal(signal.SIGINT) is before
