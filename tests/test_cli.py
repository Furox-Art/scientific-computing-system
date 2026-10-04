"""Basic smoke tests for the CDS CLI (no heavy output parsing).

The CLI is driven via :func:`cds.cli.main` with an explicit ``argv`` and its
output captured through pytest's ``capsys`` fixture. This replaces the
in-process ``typer.testing.CliRunner`` shim used previously; no subprocess is
spawned and no third-party CLI library is required.
"""

from __future__ import annotations

import pytest

from cds.cli import main


def test_help_shows_main_commands(capsys: pytest.CaptureFixture[str]) -> None:
    rc = main(["--help"])
    out = capsys.readouterr().out
    # argparse prints help to stdout and SystemExit is raised with code 0.
    assert rc == 0
    assert "modules" in out
    assert "hypothesis" in out


def test_modules_runs(capsys: pytest.CaptureFixture[str]) -> None:
    rc = main(["modules"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "Scientific Modules" in out
    assert "cds.hypothesis" in out


def test_hypothesis_runs_with_default(capsys: pytest.CaptureFixture[str]) -> None:
    rc = main(["hypothesis", "test question"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "Generating hypotheses for" in out


def test_version_flag(capsys: pytest.CaptureFixture[str]) -> None:
    rc = main(["--version"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "version" in out.lower()


def test_version_command(capsys: pytest.CaptureFixture[str]) -> None:
    rc = main(["version"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "version" in out.lower()


def test_build_parser_back_compat_alias_is_exported() -> None:
    """`_build_parser` is a historical alias kept for backwards compatibility.

    It used to be defined in ``cds.cli._parser``, where nothing referenced it --
    static analysis correctly reported it as an unused global. It now lives in
    ``cds.cli``, the module that actually exports it, so the back-compat name
    keeps working from the only path callers use. Nothing asserted that before,
    which is how the alias drifted into a module that had no use for it.
    """
    import cds.cli
    from cds.cli import _build_parser, build_parser

    assert _build_parser is build_parser
    assert "_build_parser" in cds.cli.__all__
    assert cds.cli._build_parser is cds.cli.build_parser
    # The alias must be a working entry point, not just an equal object.
    assert type(_build_parser()).__name__ == type(build_parser()).__name__
    assert _build_parser().format_help() == build_parser().format_help()
