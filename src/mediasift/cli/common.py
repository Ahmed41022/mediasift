"""What every command shares: options, output, opening the library."""

from __future__ import annotations

import json
import logging
import re
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import click

from mediasift.api import Library, RootInfo
from mediasift.core import paths

SCHEMA = "mediasift/1"

# Human output shows paths as they are. Wrapping Arabic names in Unicode isolates would keep a
# line in order in a bidi-aware terminal, but plain terminals and logs print the marks as
# boxes, and copied paths would carry them. The HTML report has dir="auto" for this instead.


@dataclass
class Env:
    """Global options, kept on the click context."""

    library: str | None = None
    json: bool = False
    debug: bool = False
    verbose: int = 0
    command: str = ""


class Leaf(click.Command):
    """A command that tells the global error handler its full name, such as ``roots add``."""

    def invoke(self, ctx: click.Context) -> Any:
        names: list[str] = []
        node: click.Context | None = ctx
        while node is not None and node.parent is not None:
            names.append(node.info_name or "")
            node = node.parent
        env_of(ctx).command = " ".join(reversed(names))
        return super().invoke(ctx)


def env_of(ctx: click.Context | None = None) -> Env:
    found = (ctx or click.get_current_context()).find_object(Env)
    return found if found is not None else Env()


def prepare_streams() -> None:
    """Never crash on a name the console cannot show (legacy Windows code pages).

    Unencodable characters become ``?`` in human output; JSON switches to escapes instead
    (see ``emit_json``), so no data is lost there.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(errors="replace")


def configure_logging(env: Env) -> None:
    """``-v`` logs info and ``-vv`` or ``--debug`` logs debug messages to stderr."""
    if not (env.debug or env.verbose):
        return
    level = logging.DEBUG if env.debug or env.verbose > 1 else logging.INFO
    logger = logging.getLogger("mediasift")
    for handler in list(logger.handlers):
        if getattr(handler, "_console", False):
            logger.removeHandler(handler)
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    handler.setLevel(level)
    handler._console = True  # type: ignore[attr-defined]
    logger.addHandler(handler)


def open_library(*, write: bool = True) -> Library:
    """Open the library the user chose, and apply the logging options."""
    env = env_of()
    library = Library.open(env.library, write=write)
    if env.debug or env.verbose > 1:
        logging.getLogger("mediasift").setLevel(logging.DEBUG)
    return library


def say(text: str = "") -> None:
    """A human message for the user: stderr, so that stdout stays the result."""
    click.echo(text, err=True)


def out(text: str = "") -> None:
    click.echo(text)


def emit_json(command: str, payload: Mapping[str, Any]) -> None:
    """Print the one JSON document of a command.

    Non-ASCII text stays readable (Arabic names) unless stdout cannot encode it; then the
    same document is written with escapes, which parse to the same values.
    """
    document = {"schema": SCHEMA, "command": command, **payload}
    text = json.dumps(document, ensure_ascii=False, indent=2)
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    try:
        text.encode(encoding)
    except (UnicodeEncodeError, LookupError):
        text = json.dumps(document, ensure_ascii=True, indent=2)
    click.echo(text)


def finish(code: int) -> None:
    """End the command with an exit code, unless it is 0."""
    if code:
        click.get_current_context().exit(code)


def show(path: Path) -> str:
    """A path as text for people: always encodable, one line."""
    return paths.display_path(path)


def iso(value: datetime) -> str:
    return value.isoformat(timespec="seconds")


def root_json(root: RootInfo) -> dict[str, Any]:
    return {
        "id": root.id,
        "path": show(root.path),
        "enabled": root.enabled,
        "read_only": root.read_only,
        "prefer_rank": root.prefer_rank,
        "volume_id": root.volume_id,
    }


def root_line(root: RootInfo) -> str:
    flags = [] if root.enabled else ["disabled"]
    if root.read_only:
        flags.append("read-only")
    note = f"  ({', '.join(flags)})" if flags else ""
    return f"{root.id:>3}  {show(root.path)}{note}"


_UNITS = {"": 1, "k": 1024, "m": 1024**2, "g": 1024**3, "t": 1024**4}
_SIZE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*([kmgt]?)(?:i?b)?\s*$", re.IGNORECASE)


def parse_size(ctx: click.Context, param: click.Parameter, value: str | int) -> int:
    """Option callback: a size in bytes from ``500``, ``10k``, ``5MB`` or ``1.5GiB``.

    Every unit is a power of 1024, matching what the reports print.
    """
    if isinstance(value, int):
        return value
    match = _SIZE.match(value)
    if match is None:
        raise click.BadParameter(f"{value!r} is not a size such as 500, 10k or 5MB")
    return int(float(match.group(1)) * _UNITS[match.group(2).lower()])
