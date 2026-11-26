from __future__ import annotations

import json
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner, Result

from mediasift.cli.main import cli
from tests.media.tree import Expected, build_tree


def norm(path: str | Path) -> str:
    """One spelling of a path, for comparing what the tool printed with what was planted."""
    return os.path.normcase(str(Path(path).resolve()))


def planted_groups(expected: Expected) -> set[frozenset[str]]:
    return {frozenset(norm(p) for p in g.paths) for g in expected.duplicate_groups}


def listed_groups(document: dict[str, Any]) -> set[frozenset[str]]:
    return {frozenset(norm(f["path"]) for f in g["files"]) for g in document["groups"]}


class Cli:
    """Runs commands against one library folder."""

    def __init__(self, library: Path) -> None:
        self.library = library
        self.runner = CliRunner()

    def run(self, *args: str, code: int = 0, library: bool = True) -> Result:
        prefix = ["--library", str(self.library)] if library else []
        result = self.runner.invoke(cli, [*prefix, *args])
        assert result.exit_code == code, (
            f"{args}: exit {result.exit_code}, wanted {code}\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}\n{result.exception!r}"
        )
        return result

    def doc(self, command: str, *args: str, code: int = 0) -> dict[str, Any]:
        """Run ``command`` (such as "roots add") with --json; stdout must be one document."""
        result = self.run("--json", *command.split(), *args, code=code)
        document: dict[str, Any] = json.loads(result.stdout)
        assert document["schema"] == "mediasift/1"
        assert document["command"] == command
        return document


@pytest.fixture
def libdir(tmp_path: Path) -> Path:
    return tmp_path / "library"


@pytest.fixture
def cli_run(libdir: Path) -> Cli:
    return Cli(libdir)


@pytest.fixture
def tree(tmp_path: Path) -> Iterator[Expected]:
    yield build_tree(tmp_path / "media")


@pytest.fixture
def scanned(cli_run: Cli, tree: Expected) -> Expected:
    """The tree added as a root and scanned."""
    cli_run.run("scan", str(tree.root))
    return tree
