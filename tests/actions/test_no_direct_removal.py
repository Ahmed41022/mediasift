"""Only a few reviewed places may delete, rename or replace files (docs/PLAN.md C1).

Everything else moves files through ``mediasift.actions.fsops``. A new call anywhere in the
package fails this test until it has been reviewed and added to ``ALLOWED``.
"""

from __future__ import annotations

import ast
from collections import Counter
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[2] / "src" / "mediasift"

# Dangerous whatever they are called on.
ANY_OWNER = {
    "unlink",
    "rmdir",
    "removedirs",
    "rmtree",
    "rename",
    "renames",
    "truncate",
    "ftruncate",
}
# Dangerous on these modules only: lists have ``remove``, strings ``replace``.
MODULE_ONLY = {"remove", "replace", "move", "copy", "copy2", "copyfile", "copytree"}
MODULES = {"os", "shutil", "posix", "nt", "pathlib"}
# System calls reached through ctypes or a library, never directly.
FOREIGN = {
    "send2trash",
    "renameat2",
    "renamex_np",
    "DeleteFileW",
    "MoveFileW",
    "MoveFileExW",
    "SHFileOperationW",
    "RemoveDirectoryW",
}
BANNED = ANY_OWNER | MODULE_ONLY | FOREIGN

# (file, function, name): how often. Each entry was reviewed; keep this list short.
ALLOWED = Counter(
    {
        # The no-replace rename of Windows; the target is checked before and after.
        ("actions/fsops.py", "_rename_no_replace", "rename"): 1,
        ("actions/fsops.py", "_libc_renameat2", "renameat2"): 1,
        # Link fallback: the source name goes into a new private folder, which is removed.
        ("actions/fsops.py", "_move_by_link", "rename"): 1,
        ("actions/fsops.py", "_move_by_link", "rmdir"): 2,
        ("actions/fsops.py", "_return_stranger", "rmdir"): 1,
        # Drops a name only while the same file has another (guard release, link fallback).
        ("actions/fsops.py", "release_extra_name", "unlink"): 1,
        # A .trashinfo record that MediaSift wrote, checked before.
        ("actions/backends/trash.py", "_drop_info", "unlink"): 1,
        # The shell's Recycle Bin, behind the guard hard link.
        ("actions/backends/recycle_bin.py", "shell_send", "send2trash"): 2,
        # The library's own lock file, emptied before the owner is written into it.
        ("db/connection.py", "__init__", "ftruncate"): 1,
        # A half-written database backup.
        ("db/migrate.py", "backup", "unlink"): 1,
    }
)


def find_calls(source: str) -> Counter[tuple[str, str]]:
    """(function, name) of every dangerous reference in a module."""
    tree = ast.parse(source)
    modules = set(MODULES)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] in MODULES:
                    modules.add(alias.asname or alias.name)
    found: Counter[tuple[str, str]] = Counter()
    _visit(tree, "<module>", modules, found)
    return found


def _visit(
    node: ast.AST, function: str, modules: set[str], found: Counter[tuple[str, str]]
) -> None:
    for child in ast.iter_child_nodes(node):
        inner = function
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
            inner = child.name
        name = _dangerous(child, modules)
        if name:
            found[(inner, name)] += 1
        _visit(child, inner, modules, found)


def _dangerous(node: ast.AST, modules: set[str]) -> str | None:
    if isinstance(node, ast.Attribute):
        owner = node.value.id if isinstance(node.value, ast.Name) else ""
        if node.attr in ANY_OWNER or node.attr in FOREIGN:
            return node.attr
        if node.attr in MODULE_ONLY and owner in modules:
            return node.attr
    if isinstance(node, ast.Name) and node.id in FOREIGN:
        return node.id
    if isinstance(node, ast.Call):
        return _dangerous_call(node)
    if isinstance(node, ast.ImportFrom) and node.module:
        top = node.module.split(".")[0]
        for alias in node.names:
            if top in FOREIGN or (top in MODULES and alias.name in BANNED):
                return alias.name
    if isinstance(node, ast.Import):
        for alias in node.names:
            if alias.name.split(".")[0] in FOREIGN:
                return alias.name
    if isinstance(node, ast.Constant) and node.value in ANY_OWNER | FOREIGN:
        return str(node.value)
    return None


def _dangerous_call(call: ast.Call) -> str | None:
    function = call.func
    # Path.replace(target) has one argument; str.replace has two.
    if isinstance(function, ast.Attribute) and function.attr == "replace":
        return "replace" if len(call.args) == 1 and not call.keywords else None
    if isinstance(function, ast.Name) and function.id == "getattr" and len(call.args) > 1:
        name = call.args[1]
        if isinstance(name, ast.Constant) and name.value in MODULE_ONLY:
            return str(name.value)
    return None


def test_only_reviewed_places_remove_files() -> None:
    found: Counter[tuple[str, str, str]] = Counter()
    for path in sorted(SRC.rglob("*.py")):
        rel = path.relative_to(SRC).as_posix()
        for (function, name), count in find_calls(path.read_text(encoding="utf-8")).items():
            found[(rel, function, name)] += count
    unexpected = {key: n for key, n in found.items() if n > ALLOWED[key]}
    assert not unexpected, f"new ways to delete or move files, review them: {unexpected}"
    stale = {key: n for key, n in ALLOWED.items() if found[key] != n}
    assert not stale, f"the allow-list is out of date: {stale}"


@pytest.mark.parametrize(
    "source",
    [
        "import os\ndef f(p):\n    os.remove(p)",
        "from os import unlink as u",
        "from shutil import move",
        "import shutil as sh\ndef f(p):\n    sh.rmtree(p)",
        "def f(p):\n    p.unlink(missing_ok=True)",
        "def f(p, q):\n    p.replace(q)",
        "import os\ndef f(p, q):\n    os.replace(p, q)",
        "def f(p, q):\n    p.rename(q)",
        "import os\ndef f(p):\n    getattr(os, 'unlink')(p)",
        "import os\ndef f(p, q):\n    getattr(os, 'replace')(p, q)",
        "import send2trash",
        "from send2trash import send2trash",
        "import ctypes\nctypes.windll.kernel32.DeleteFileW",
    ],
)
def test_detector_sees_the_usual_spellings(source: str) -> None:
    assert find_calls(source)


@pytest.mark.parametrize(
    "source",
    [
        "def f(s):\n    return s.replace('a', 'b')",
        "def f(items, x):\n    items.remove(x)",
        "def f(record):\n    return replace(record, id=1)",
        "def f(d):\n    return d.replace(year=2000)",
        "def f(b):\n    return b.decode('utf-8', 'replace')",
    ],
)
def test_detector_leaves_ordinary_code_alone(source: str) -> None:
    assert not find_calls(source)
