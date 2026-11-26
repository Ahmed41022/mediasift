"""``mediasift roots``: the folders in the library."""

from __future__ import annotations

from collections.abc import Callable

import click

from mediasift.api import Library, RootInfo
from mediasift.cli.common import (
    Leaf,
    emit_json,
    env_of,
    open_library,
    out,
    root_json,
    root_line,
)


@click.group()
def roots() -> None:
    """Choose the folders MediaSift looks at. Nothing outside them is ever touched."""


@roots.command("add", cls=Leaf)
@click.argument("path", type=click.Path(file_okay=False))
@click.option("--read-only", is_flag=True, help="Match files here, but never act on them.")
@click.option(
    "--rank", "prefer_rank", type=int, default=0, help="Higher keeps this folder's files."
)
def add(path: str, read_only: bool, prefer_rank: int) -> None:
    """Add a folder. A folder that contains or lies in another one is refused."""
    with open_library() as library:
        root = library.add_root(path, read_only=read_only, prefer_rank=prefer_rank)
    if env_of().json:
        emit_json("roots add", {"root": root_json(root)})
    else:
        out(f"Added folder {root_line(root).strip()}")


@roots.command("list", cls=Leaf)
def list_roots() -> None:
    """Show the folders."""
    with open_library(write=False) as library:
        found = library.list_roots()
    if env_of().json:
        emit_json("roots list", {"roots": [root_json(r) for r in found]})
    elif found:
        for root in found:
            out(root_line(root))
    else:
        out("No folders yet. Add one with: mediasift roots add PATH")


def _change(command: str, root_id: int, verb: str, action: Callable[[Library], RootInfo]) -> None:
    with open_library() as library:
        root = action(library)
    if env_of().json:
        emit_json(command, {"root": root_json(root)})
    else:
        out(f"{verb} folder {root_line(root).strip()}")


@roots.command("disable", cls=Leaf)
@click.argument("root_id", type=int)
def disable(root_id: int) -> None:
    """Leave a folder out of scans and matching. Its files stay in the index."""
    _change("roots disable", root_id, "Disabled", lambda lib: lib.disable_root(root_id))


@roots.command("enable", cls=Leaf)
@click.argument("root_id", type=int)
def enable(root_id: int) -> None:
    """Include a disabled folder again."""
    _change("roots enable", root_id, "Enabled", lambda lib: lib.enable_root(root_id))


@roots.command("accept-volume", cls=Leaf)
@click.argument("root_id", type=int)
def accept_volume(root_id: int) -> None:
    """Accept the drive now at a folder as the right one.

    A scan refuses a folder whose drive changed, because an unplugged drive would look like
    every file was deleted. Use this after replacing or reformatting the drive on purpose.
    """
    _change(
        "roots accept-volume",
        root_id,
        "Accepted the drive of",
        lambda lib: lib.set_root_volume(root_id),
    )
