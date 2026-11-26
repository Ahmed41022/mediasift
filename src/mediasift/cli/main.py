"""Command-line entry point."""

from __future__ import annotations

from typing import Any

import click

from mediasift import __version__
from mediasift.api import ExitCode, MediaSiftError
from mediasift.cli.common import Env, configure_logging, emit_json, env_of, prepare_streams, say
from mediasift.cli.dupes import dupes
from mediasift.cli.roots import roots
from mediasift.cli.scan import scan
from mediasift.cli.stats import report, stats


class MediaSiftGroup(click.Group):
    """Turns expected errors into a message and an exit code, never a traceback."""

    def invoke(self, ctx: click.Context) -> Any:
        try:
            return super().invoke(ctx)
        except (click.ClickException, click.exceptions.Exit, click.Abort):
            raise
        except KeyboardInterrupt:
            say("Interrupted.")
            ctx.exit(int(ExitCode.CANCELLED))
        except MediaSiftError as exc:
            self._fail(ctx, str(exc), exc.exit_code, type(exc).__name__)
        except Exception as exc:
            if env_of(ctx).debug:
                raise
            self._fail(
                ctx,
                f"unexpected error: {exc} (run again with --debug for details)",
                ExitCode.FAILURE,
                type(exc).__name__,
            )

    @staticmethod
    def _fail(ctx: click.Context, message: str, code: ExitCode, kind: str) -> None:
        if env_of(ctx).json:
            emit_json(
                env_of(ctx).command or ctx.invoked_subcommand or "",
                {"error": {"message": message, "type": kind, "exit_code": int(code)}},
            )
        say(f"error: {message}")
        ctx.exit(int(code))


@click.group(cls=MediaSiftGroup)
@click.version_option(__version__, prog_name="mediasift")
@click.option(
    "--library", metavar="PATH", help="Folder of the library (default: your data folder)."
)
@click.option("--json", "as_json", is_flag=True, help="Print one JSON document on stdout.")
@click.option("--debug", is_flag=True, help="Show tracebacks and debug messages.")
@click.option("-v", "--verbose", count=True, help="Log what happens to stderr (twice: in detail).")
@click.pass_context
def cli(ctx: click.Context, library: str | None, as_json: bool, debug: bool, verbose: int) -> None:
    """Find duplicate media and clean up safely."""
    prepare_streams()
    ctx.obj = Env(library, as_json, debug, verbose)
    configure_logging(ctx.obj)


cli.add_command(roots)
cli.add_command(scan)
cli.add_command(dupes)
cli.add_command(stats)
cli.add_command(report)


def main() -> None:
    """Run the command line."""
    cli()
