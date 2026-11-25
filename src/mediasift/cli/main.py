"""Command-line entry point."""

from __future__ import annotations

import click

from mediasift import __version__


@click.group()
@click.version_option(__version__, prog_name="mediasift")
def cli() -> None:
    """Find duplicate media and clean up safely."""


def main() -> None:
    """Run the command line."""
    cli()
