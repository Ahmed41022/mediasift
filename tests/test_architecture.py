"""Layering rules from docs/PLAN.md 1.1."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src" / "mediasift"
GUI_LIBRARIES = {"PySide6", "PySide2", "PyQt5", "PyQt6", "tkinter", "wx", "gi"}

# Higher layers may import lower ones, never the other way round.
LAYER = {
    "core": 0,
    "db": 1,
    "jobs": 1,
    "readers": 2,
    "hashing": 2,
    "scan": 3,
    "match": 3,
    "keeper": 4,
    "actions": 5,
    "organize": 5,
    "importer": 5,
    "report": 5,
    "api": 6,
    "cli": 7,
    "gui": 7,
    "web": 7,
}
INTERFACES = {"cli", "gui", "web"}
INTERFACE_MAY_USE = {"api", "core"}

FILES = sorted(SRC.rglob("*.py"))


def part_of(path: Path) -> str:
    return path.relative_to(SRC).parts[0].removesuffix(".py")


def imports(path: Path) -> set[str]:
    found: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            assert node.level == 0, f"{path}: use absolute imports"
            found.add(node.module)
            if node.module == "mediasift":
                found.update(f"mediasift.{alias.name}" for alias in node.names)
    return found


@pytest.mark.parametrize("path", FILES, ids=lambda p: p.relative_to(SRC).as_posix())
def test_layering(path: Path) -> None:
    part = part_of(path)
    if part in {"__init__", "__main__"}:
        return
    assert part in LAYER, f"add {part} to LAYER"
    for name in imports(path):
        top = name.split(".")[0]
        assert part == "gui" or top not in GUI_LIBRARIES, f"{path}: GUI import {name}"
        if top != "mediasift" or "." not in name:
            continue
        target = name.split(".")[1]
        if target not in LAYER or target == part:
            continue
        assert LAYER[target] < LAYER[part], f"{path}: {part} must not import {target}"
        if part in INTERFACES:
            assert target in INTERFACE_MAY_USE, f"{path}: interfaces use mediasift.api only"
