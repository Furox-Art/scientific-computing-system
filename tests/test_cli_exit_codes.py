"""M-3: the CLI's exit-code contract, walked end to end.

The CLI documented no exit codes at all, yet every failure came back as a bare
``1``. A caller comparing statuses could not tell "your numbers were bad" from
"the dashboard script is missing" from "streamlit is not installed", and the
suite pinned those statuses only by accident — one ``assert rc == 1`` per
command, with nothing tying them together.

``tests/test_cli_exit_codes.py`` pins the mapping from each representative
failure to its documented code by driving :func:`cds.cli.main` the way a user
does, and separately asserts that no handler returns a code outside the
documented set.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

from cds.cli import (
    EXIT_CODES,
    EXIT_ERROR,
    EXIT_MISSING_DEPENDENCY,
    EXIT_OK,
    EXIT_USAGE,
    describe,
    is_handler_code,
    main,
)


class TestDocumentedCodes:
    """The contract table itself is data, not prose."""

    def test_every_code_has_a_description(self) -> None:
        assert set(EXIT_CODES) == {EXIT_OK, EXIT_ERROR, EXIT_USAGE, EXIT_MISSING_DEPENDENCY}
        for code, text in EXIT_CODES.items():
            assert isinstance(code, int)
            assert text and text == text.strip()

    def test_describe_matches_the_table(self) -> None:
        assert describe(EXIT_OK) == "command completed and produced its output"
        assert describe(EXIT_ERROR).startswith("command was understood")
        assert describe(EXIT_USAGE).startswith("command line was rejected")
        assert describe(EXIT_MISSING_DEPENDENCY).startswith("command needs an optional extra")

    def test_describe_rejects_non_integers(self) -> None:
        """``bool`` is an ``int`` subclass in Python, but not an exit code."""
        assert "undocumented" in describe(99)
        for not_a_code in (True, False, "0", 1.0, None):
            with pytest.raises(TypeError):
                describe(not_a_code)  # type: ignore[arg-type]

    def test_handler_codes_are_a_subset_of_the_documented_set(self) -> None:
        for code in (EXIT_OK, EXIT_ERROR, EXIT_MISSING_DEPENDENCY):
            assert is_handler_code(code)
        # EXIT_USAGE belongs to argparse, before a handler ever runs.
        assert not is_handler_code(EXIT_USAGE)


class TestSuccessContract:
    """Every exit-0 path, driven through ``main``."""

    @pytest.mark.parametrize(
        "argv",
        [
            ["version"],
            ["modules"],
            ["info"],
            ["constants"],
            ["stats", "1,2,3,4,5"],
            ["stats", "5"],
            ["sample", "gaussian", "-n", "3", "--seed", "1"],
            ["sample", "poisson", "-n", "2", "--seed", "0"],
            ["integrate", "x2", "--a", "0", "--b", "1", "-n", "100"],
            ["integrate", "unit", "--a", "0", "--b", "2", "--method", "trap", "-n", "10"],
            ["hypothesis", "test question"],
            ["plot", "1,2,3"],
            ["--version"],
        ],
    )
    def test_successful_commands_return_exit_ok(
        self, argv: list[str], capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert main(list(argv)) == EXIT_OK
        assert capsys.readouterr().out.strip()

    def test_help_and_bare_invocation_return_exit_ok(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert main(["--help"]) == EXIT_OK
        capsys.readouterr()
        assert main(["nonexistent-subcommand"]) == EXIT_USAGE


class TestFailureContract:
    """Each failure class maps to its own documented code."""

    def test_malformed_values_map_to_exit_error(self, capsys: pytest.CaptureFixture[str]) -> None:
        """A comma-separated list that is not numbers: bad input, exit 1."""
        assert main(["stats", "1,x,3"]) == EXIT_ERROR
        assert "Error" in capsys.readouterr().out

        assert main(["plot", "1,not-a-number"]) == EXIT_ERROR
        assert "Error" in capsys.readouterr().out

    def test_out_of_domain_parameters_map_to_exit_error(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Distribution/quadrature constraints are input errors, not usage."""
        assert main(["sample", "uniform", "-n", "2", "--a", "1", "--b", "0"]) == EXIT_ERROR
        capsys.readouterr()
        assert main(["sample", "gaussian", "-n", "1", "--sigma", "0"]) == EXIT_ERROR
        capsys.readouterr()
        assert main(["sample", "exponential", "-n", "1", "--lam", "-1"]) == EXIT_ERROR
        capsys.readouterr()
        assert main(["integrate", "sin", "-n", "0"]) == EXIT_ERROR
        capsys.readouterr()

    def test_unknown_choice_maps_to_exit_usage(self, capsys: pytest.CaptureFixture[str]) -> None:
        """argparse rejected the command line, so the code is argparse's."""
        assert main(["sample", "no-such-distribution"]) == EXIT_USAGE
        capsys.readouterr()
        assert main(["integrate", "no-such-integrand"]) == EXIT_USAGE
        capsys.readouterr()
        # --kind acf is only meaningful with --file; ASCII mode rejects it.
        assert main(["plot", "1,2", "--kind", "acf"]) == EXIT_ERROR
        capsys.readouterr()

    def test_missing_required_argument_maps_to_exit_usage(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert main(["stats"]) == EXIT_USAGE
        # argparse writes the usage message to stderr, as it always has.
        assert "usage" in capsys.readouterr().err.lower()

    def test_unknown_subcommand_maps_to_exit_usage(self) -> None:
        assert main(["definitely-not-a-command"]) == EXIT_USAGE

    def test_unparseable_number_maps_to_exit_usage(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """``--a notanumber`` never reaches a handler."""
        assert main(["integrate", "sin", "--a", "notanumber"]) == EXIT_USAGE
        capsys.readouterr()

    def test_missing_streamlit_maps_to_exit_missing_dependency(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """The remedy is ``pip install streamlit``, so this is not a usage error."""

        def fake_run(cmd: list[str], **kwargs: object) -> None:
            raise FileNotFoundError("streamlit not found")

        monkeypatch.setattr(Path, "exists", lambda self: True)
        monkeypatch.setattr(subprocess, "run", fake_run)
        code = main(["dashboard"])
        assert code == EXIT_MISSING_DEPENDENCY
        assert "Streamlit not found" in capsys.readouterr().out

    def test_missing_matplotlib_maps_to_exit_missing_dependency(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """``--file`` needs the ``[plot]`` extra; that is also not a usage error."""
        monkeypatch.setitem(sys.modules, "cds.plot", None)
        target = tmp_path / "out.png"
        code = main(["plot", "1,2,3", "--file", str(target)])
        assert code == EXIT_MISSING_DEPENDENCY
        assert "Error" in capsys.readouterr().out

    def test_missing_dashboard_file_maps_to_exit_error(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A missing repo resource, not an optional extra — still exit 1."""
        monkeypatch.setattr(Path, "exists", lambda self: False)
        assert main(["dashboard"]) == EXIT_ERROR
        assert "Dashboard file not found" in capsys.readouterr().out

    def test_dashboard_crash_maps_to_exit_error(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """streamlit present but the app failing is a run failure, not an install."""
        monkeypatch.setattr(Path, "exists", lambda self: True)
        monkeypatch.setattr(importlib.util, "find_spec", lambda *_a, **_k: object())

        def fake_run(cmd: list[str], **kwargs: object) -> None:
            raise subprocess.CalledProcessError(7, cmd)

        monkeypatch.setattr(subprocess, "run", fake_run)
        assert main(["dashboard"]) == EXIT_ERROR
        assert "status 7" in capsys.readouterr().out

    def test_bad_plot_values_map_to_exit_error(self, capsys: pytest.CaptureFixture[str]) -> None:
        assert main(["plot", "oops"]) == EXIT_ERROR
        capsys.readouterr()

    def test_ascii_hist_kind_maps_to_exit_error(self, capsys: pytest.CaptureFixture[str]) -> None:
        assert main(["plot", "1,2,3", "--kind", "hist"]) == EXIT_ERROR
        assert "--kind series" in capsys.readouterr().out


class TestNoUndocumentedCodes:
    """A handler must never return a code outside the contract."""

    @pytest.mark.parametrize(
        ("argv", "handler"),
        [
            (["stats", "1,2"], "_cmd_stats"),
            (["sample", "gaussian"], "_cmd_sample"),
            (["integrate", "sin"], "_cmd_integrate"),
            (["plot", "1,2"], "_cmd_plot"),
            (["version"], "_cmd_version"),
            (["constants"], "_cmd_constants"),
            (["modules"], "_cmd_modules"),
            (["info"], "_cmd_info"),
            (["benchmark"], "_cmd_benchmark"),
            (["hypothesis", "q"], "_cmd_hypothesis"),
            (["prompt", "q"], "_cmd_prompt"),
        ],
    )
    def test_handler_returns_a_documented_code(
        self, argv: list[str], handler: str, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Success or failure, the code must be in :data:`EXIT_CODES`."""
        code = main(list(argv))
        capsys.readouterr()
        assert code in EXIT_CODES, f"{handler} returned undocumented code {code}"

    def test_success_and_failure_are_distinguished(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """The core of M-3: 'ok' and 'not ok' must not collide."""
        assert main(["stats", "1,2,3"]) == EXIT_OK
        capsys.readouterr()
        assert main(["stats", "nope"]) != EXIT_OK
