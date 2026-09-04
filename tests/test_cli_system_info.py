"""Regression tests for the current CLI architecture/module catalog."""

from __future__ import annotations

import pytest

from cds.cli import main


def test_cli_info_reports_assurance_and_orchestration_layers(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["info"]) == 0
    output = capsys.readouterr().out
    assert "Architecture" in output
    assert "validation / uncertainty / units / provenance" in output
    assert "workflow / optional scientific tools" in output


def test_cli_modules_reports_modern_scientific_layers(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["modules"]) == 0
    output = capsys.readouterr().out
    for module in (
        "cds.data_io",
        "cds.units",
        "cds.uncertainty",
        "cds.sensitivity",
        "cds.validation",
        "cds.workflow",
        "cds.provenance",
        "cds.tools",
    ):
        assert module in output
    assert "scientific-computing-system[scientific,io,plot]" in output
