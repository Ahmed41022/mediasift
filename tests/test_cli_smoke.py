from __future__ import annotations

from click.testing import CliRunner

from mediasift.cli.main import cli


def test_version() -> None:
    result = CliRunner().invoke(cli, ["--version"])
    assert result.exit_code == 0
    assert "mediasift" in result.output
