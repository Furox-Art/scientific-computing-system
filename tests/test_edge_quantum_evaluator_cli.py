"""Tests covering previously untested code paths.

Targets:
- cds.quantum.simulator.measure (state collapse)
- cds.__main__ entry point
- cds.hypothesis.evaluator new dispatch paths
"""

import math
import subprocess
import sys
from pathlib import Path
from typing import cast

import pytest

from cds import __version__
from cds.cli import main
from cds.core.models import Hypothesis
from cds.hypothesis import (
    Domain,
    EvaluationData,
    HypothesisEvaluator,
    HypothesisStatus,
    generate_hypotheses,
)
from cds.montecarlo import buffon_needle, estimate_pi
from cds.montecarlo.methods import _pi_worker
from cds.quantum import QuantumCircuit, hadamard, pauli_x
from cds.quantum.circuit import Qubit
from cds.quantum.simulator import measure, simulate


# ---------------------------------------------------------------------------
# Quantum simulator: measure() state collapse
# ---------------------------------------------------------------------------
def test_measure_collapse_to_zero() -> None:
    """A qubit in |0> collapses to 0 and stays |0>."""
    q = Qubit(alpha=1.0 + 0j, beta=0.0 + 0j)
    outcome = measure(q)
    assert outcome == 0
    assert q.alpha == 1.0
    assert q.beta == 0.0


def test_measure_collapse_to_one() -> None:
    """A qubit in |1> collapses to 1 and stays |1>."""
    q = Qubit(alpha=0.0 + 0j, beta=1.0 + 0j)
    outcome = measure(q)
    assert outcome == 1
    assert q.alpha == 0.0
    assert q.beta == 1.0


def test_measure_superposition_distribution() -> None:
    """A |+> qubit collapses ~50/50 over many samples."""
    inv = 1.0 / math.sqrt(2)
    counts = {0: 0, 1: 0}
    for _ in range(2000):
        q = Qubit(alpha=complex(inv), beta=complex(inv))
        counts[measure(q)] += 1
    assert counts[0] > 0 and counts[1] > 0


def test_simulate_seed_reproducible() -> None:
    """simulate() with the same seed gives identical counts."""
    circuit = QuantumCircuit().add(hadamard()).add(pauli_x())
    a = simulate(circuit, shots=500, seed=123)
    b = simulate(circuit, shots=500, seed=123)
    assert a == b


# ---------------------------------------------------------------------------
# __main__ entry point (coverage for cds.__main__)
# ---------------------------------------------------------------------------
def test_main_module_runs() -> None:
    """`python -m cds --version` exits 0 and prints the version."""
    import os

    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    src_path = os.path.join(root, "src")
    env = os.environ.copy()
    env["PYTHONPATH"] = (
        f"{src_path}{os.pathsep}{env['PYTHONPATH']}" if env.get("PYTHONPATH") else src_path
    )
    result = subprocess.run(
        [sys.executable, "-m", "cds", "--version"],
        capture_output=True,
        text=True,
        timeout=30,
        env=env,
    )
    assert result.returncode == 0
    assert __version__ in result.stdout


# ---------------------------------------------------------------------------
# Evaluator: new dispatch paths
# ---------------------------------------------------------------------------
def _make_hypothesis() -> Hypothesis:
    hypos = generate_hypotheses("Test question?", Domain.GENERAL_SCIENCE, n=1)
    return hypos[0]


def test_evaluator_one_sample_significant() -> None:
    hypo = _make_hypothesis()
    evaluator = HypothesisEvaluator()
    result = evaluator.evaluate(
        hypo,
        {"one_sample": [10.2, 10.5, 9.8, 11.0, 10.4], "popmean": 5.0},
    )
    assert result.test_name == "One-sample t-test"
    assert result.is_significant
    assert hypo.status == HypothesisStatus.VALIDATED


def test_evaluator_one_sample_not_significant() -> None:
    hypo = _make_hypothesis()
    evaluator = HypothesisEvaluator()
    result = evaluator.evaluate(
        hypo,
        {"one_sample": [10.0, 10.1, 9.9, 10.0, 10.1], "popmean": 10.0},
    )
    assert not result.is_significant
    assert hypo.status == HypothesisStatus.REJECTED


def test_evaluator_chi_square_gof() -> None:
    hypo = _make_hypothesis()
    evaluator = HypothesisEvaluator()
    result = evaluator.evaluate(
        hypo,
        {"chi_square_gof": {"observed": [50, 10, 10, 30]}},
    )
    assert result.test_name == "Chi-square goodness-of-fit"
    assert result.is_significant


def test_evaluator_chi_square_independence() -> None:
    hypo = _make_hypothesis()
    evaluator = HypothesisEvaluator()
    table: list[list[float]] = [[100.0, 10.0], [10.0, 100.0]]
    result = evaluator.evaluate(hypo, {"chi_square_independence": table})
    assert result.test_name == "Chi-square independence"
    assert result.is_significant


def test_evaluator_paired() -> None:
    hypo = _make_hypothesis()
    evaluator = HypothesisEvaluator()
    result = evaluator.evaluate(
        hypo,
        {"paired": ([1.0, 2.0, 3.0, 4.0], [10.0, 11.0, 12.0, 13.0])},
    )
    assert result.test_name == "Two-sample t-test"
    assert result.is_significant


def test_evaluator_unsupported_format_raises() -> None:
    hypo = _make_hypothesis()
    evaluator = HypothesisEvaluator()
    try:
        evaluator.evaluate(hypo, cast(EvaluationData, {"unknown_key": [1, 2, 3]}))
        raise AssertionError("Expected ValueError for unsupported data format")
    except ValueError as exc:
        assert "Unsupported data format" in str(exc)


def test_evaluator_compare_groups_too_few_raises() -> None:
    hypo = _make_hypothesis()
    evaluator = HypothesisEvaluator()
    try:
        evaluator.compare_groups(hypo, [[1.0, 2.0]])
        raise AssertionError("Expected ValueError for single group")
    except ValueError as exc:
        assert "2 groups" in str(exc)


def test_evaluator_goodness_of_fit_too_few_raises() -> None:
    hypo = _make_hypothesis()
    evaluator = HypothesisEvaluator()
    try:
        evaluator.goodness_of_fit(hypo, [5.0])
        raise AssertionError("Expected ValueError for single category")
    except ValueError as exc:
        assert "2 categories" in str(exc)


def test_evaluator_independence_bad_table_raises() -> None:
    hypo = _make_hypothesis()
    evaluator = HypothesisEvaluator()
    try:
        evaluator.test_independence(hypo, [[1.0]])
        raise AssertionError("Expected ValueError for bad contingency table")
    except ValueError as exc:
        assert "contingency" in str(exc)


def test_evaluator_one_sample_too_few_raises() -> None:
    hypo = _make_hypothesis()
    evaluator = HypothesisEvaluator()
    try:
        evaluator.compare_to_reference(hypo, [5.0], popmean=4.0)
        raise AssertionError("Expected ValueError for single observation")
    except ValueError as exc:
        assert "2 observations" in str(exc)


# ---------------------------------------------------------------------------
# Evaluator: effect-size reporting on each dispatch path
# ---------------------------------------------------------------------------
def test_evaluator_two_sample_reports_cohens_d() -> None:
    hypo = _make_hypothesis()
    evaluator = HypothesisEvaluator()
    result = evaluator.evaluate(
        hypo,
        {"groups": [[1.0, 2.0, 3.0, 4.0, 5.0], [6.0, 7.0, 8.0, 9.0, 10.0]]},
    )
    assert result.test_name == "Two-sample t-test"
    assert result.effect_size is not None
    assert result.effect_size_label == "Cohen's d"
    assert abs(result.effect_size) > 2.0


def test_evaluator_anova_reports_eta_squared() -> None:
    hypo = _make_hypothesis()
    evaluator = HypothesisEvaluator()
    result = evaluator.evaluate(
        hypo,
        {"groups": [[1.0, 2.0, 3.0], [2.0, 3.0, 4.0], [4.0, 5.0, 6.0]]},
    )
    assert result.test_name == "One-way ANOVA"
    assert result.effect_size is not None
    assert result.effect_size_label == "eta-squared"
    assert 0.0 <= result.effect_size <= 1.0


def test_evaluator_one_sample_reports_cohens_d() -> None:
    hypo = _make_hypothesis()
    evaluator = HypothesisEvaluator()
    result = evaluator.evaluate(
        hypo,
        {"one_sample": [10.2, 10.5, 9.8, 11.0, 10.4], "popmean": 5.0},
    )
    assert result.test_name == "One-sample t-test"
    assert result.effect_size is not None
    assert result.effect_size_label == "Cohen's d"
    assert result.effect_size > 5.0


def test_evaluator_independence_reports_cramers_v() -> None:
    hypo = _make_hypothesis()
    evaluator = HypothesisEvaluator()
    table: list[list[float]] = [[100.0, 10.0], [10.0, 100.0]]
    result = evaluator.evaluate(hypo, {"chi_square_independence": table})
    assert result.test_name == "Chi-square independence"
    assert result.effect_size is not None
    assert result.effect_size_label == "Cramer's V"
    assert 0.0 <= result.effect_size <= 1.0


def test_evaluator_goodness_of_fit_has_no_effect_size() -> None:
    hypo = _make_hypothesis()
    evaluator = HypothesisEvaluator()
    result = evaluator.evaluate(hypo, {"chi_square_gof": {"observed": [50, 10, 10, 30]}})
    assert result.test_name == "Chi-square goodness-of-fit"
    assert result.effect_size is None
    assert result.effect_size_label is None


def test_evaluator_conclusion_mentions_effect_size() -> None:
    hypo = _make_hypothesis()
    evaluator = HypothesisEvaluator()
    result = evaluator.evaluate(
        hypo,
        {"groups": [[1.0, 2.0, 3.0, 4.0, 5.0], [6.0, 7.0, 8.0, 9.0, 10.0]]},
    )
    assert "Effect size: Cohen's d" in result.conclusion


def test_evaluator_paired_reports_cohens_d() -> None:
    hypo = _make_hypothesis()
    evaluator = HypothesisEvaluator()
    result = evaluator.evaluate(
        hypo,
        {"paired": ([1.0, 2.0, 3.0, 4.0], [10.0, 11.0, 12.0, 13.0])},
    )
    assert result.test_name == "Two-sample t-test"
    assert result.effect_size is not None
    assert result.effect_size_label == "Cohen's d"


# ---------------------------------------------------------------------------
# Evaluator: evaluate_batch (Bonferroni multiple-comparison correction)
# ---------------------------------------------------------------------------
def test_evaluator_batch_applies_bonferroni() -> None:
    h1 = _make_hypothesis()
    h2 = _make_hypothesis()
    h3 = _make_hypothesis()
    evaluator = HypothesisEvaluator(alpha=0.05)
    results = evaluator.evaluate_batch(
        [h1, h2, h3],
        [
            {"groups": [[1.0, 2.0, 3.0, 4.0, 5.0], [20.0, 21.0, 22.0, 23.0, 24.0]]},
            {"groups": [[1.0, 2.0, 3.0, 4.0, 5.0], [3.0, 4.0, 5.0, 6.0, 7.0]]},
            {"groups": [[1.0, 2.0, 3.0, 4.0, 5.0], [1.5, 2.5, 3.0, 3.5, 4.5]]},
        ],
    )
    assert len(results) == 3
    for r in results:
        assert "alpha=0.016" in r.conclusion


def test_evaluator_batch_single_comparison_matches_evaluate() -> None:
    hypo = _make_hypothesis()
    evaluator = HypothesisEvaluator()
    single = evaluator.evaluate(
        hypo,
        {"groups": [[1.0, 2.0, 3.0, 4.0, 5.0], [6.0, 7.0, 8.0, 9.0, 10.0]]},
    )
    hypo2 = _make_hypothesis()
    batched = evaluator.evaluate_batch(
        [hypo2],
        [{"groups": [[1.0, 2.0, 3.0, 4.0, 5.0], [6.0, 7.0, 8.0, 9.0, 10.0]]}],
    )
    assert batched[0].is_significant == single.is_significant


def test_evaluator_batch_length_mismatch_raises() -> None:
    evaluator = HypothesisEvaluator()
    with pytest.raises(ValueError, match="same length"):
        evaluator.evaluate_batch([_make_hypothesis()], [])


def test_evaluator_batch_empty_raises() -> None:
    evaluator = HypothesisEvaluator()
    with pytest.raises(ValueError, match="at least one"):
        evaluator.evaluate_batch([], [])


# ---------------------------------------------------------------------------
# CLI commands: plot, constants, dashboard error path
# ---------------------------------------------------------------------------
def test_cli_plot_valid(capsys: pytest.CaptureFixture[str]) -> None:
    rc = main(["plot", "1,5,3,8", "--title", "Data"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "Data" in out


def test_cli_plot_invalid(capsys: pytest.CaptureFixture[str]) -> None:
    rc = main(["plot", "1,abc,3"])
    out = capsys.readouterr().out
    assert rc == 1
    assert "Error" in out


def test_cli_plot_file_png(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    pytest.importorskip("matplotlib")
    path = tmp_path / "cli_plot.png"
    rc = main(["plot", "1,5,3,8,2", "--title", "PNG", "--file", str(path)])
    captured = capsys.readouterr().out
    assert rc == 0
    assert "Saved" in captured
    assert path.is_file()
    assert path.stat().st_size > 0


def test_cli_plot_file_success_without_matplotlib(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    import cds.plot as plot_mod

    saved: list[str] = []

    class _Fig:
        def savefig(self, path: str, dpi: int = 120) -> None:
            saved.append(path)
            Path(path).write_bytes(b"png")

    def _fake_series(*_args: object, **_kwargs: object) -> _Fig:
        return _Fig()

    monkeypatch.setattr(plot_mod, "plot_series", _fake_series)
    monkeypatch.setattr(
        plot_mod, "save_figure", lambda fig, path, dpi=120: fig.savefig(path, dpi=dpi) or path
    )
    path = tmp_path / "mocked.png"
    rc = main(["plot", "1,2,3,4", "--title", "Mock", "--file", str(path)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "Saved" in out
    assert path.is_file()


def test_cli_plot_kind_hist_mocked(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    import cds.plot as plot_mod

    class _Fig:
        def savefig(self, path: str, dpi: int = 120) -> None:
            Path(path).write_bytes(b"png")

    monkeypatch.setattr(plot_mod, "plot_histogram", lambda *a, **k: _Fig())
    monkeypatch.setattr(
        plot_mod, "save_figure", lambda fig, path, dpi=120: fig.savefig(path, dpi=dpi) or path
    )
    path = tmp_path / "hist.png"
    rc = main(["plot", "1,2,2,3", "--kind", "hist", "--file", str(path)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "hist" in out
    assert path.is_file()


def test_cli_plot_kind_acf_mocked(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    import cds.plot as plot_mod

    class _Fig:
        def savefig(self, path: str, dpi: int = 120) -> None:
            Path(path).write_bytes(b"png")

    monkeypatch.setattr(plot_mod, "plot_acf", lambda *a, **k: _Fig())
    monkeypatch.setattr(
        plot_mod, "save_figure", lambda fig, path, dpi=120: fig.savefig(path, dpi=dpi) or path
    )
    path = tmp_path / "acf.png"
    data = ",".join(str(i % 5) for i in range(30))
    rc = main(["plot", data, "--kind", "acf", "--file", str(path)])
    assert rc == 0
    assert path.is_file()


def test_cli_plot_ascii_rejects_hist(capsys: pytest.CaptureFixture[str]) -> None:
    rc = main(["plot", "1,2,3", "--kind", "hist"])
    out = capsys.readouterr().out
    assert rc == 1
    assert "ASCII" in out


def test_cli_plot_file_missing_matplotlib(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    import builtins
    import sys

    real_import = builtins.__import__

    def fake_import(name: str, *args, **kwargs):  # type: ignore[no-untyped-def]
        if name == "matplotlib" or name.startswith("matplotlib."):
            raise ImportError("simulated missing matplotlib")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.delitem(sys.modules, "matplotlib", raising=False)
    monkeypatch.delitem(sys.modules, "matplotlib.pyplot", raising=False)

    path = tmp_path / "nope.png"
    rc = main(["plot", "1,2,3", "--file", str(path)])
    out = capsys.readouterr().out
    assert rc == 1
    assert "Error" in out
    assert "optional dependency" in out or "matplotlib" in out.lower()


def test_cli_constants(capsys: pytest.CaptureFixture[str]) -> None:
    rc = main(["constants"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "Physical Constants" in out


def test_cli_dashboard_missing_file(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(Path, "exists", lambda self: False)
    rc = main(["dashboard"])
    out = capsys.readouterr().out
    assert rc == 1
    assert "not found" in out.lower()


def test_cli_calc_unknown_formula(capsys: pytest.CaptureFixture[str]) -> None:
    rc = main(["calc", "unknown"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "Unknown formula" in out


def test_cli_no_command_shows_help(capsys: pytest.CaptureFixture[str]) -> None:
    rc = main([])
    out = capsys.readouterr().out
    assert "help" in out.lower() or "Usage" in out
    assert rc == 0


def test_cli_hypothesis_show_prompt(capsys: pytest.CaptureFixture[str]) -> None:
    rc = main(["hypothesis", "Test question", "--show-prompt"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "Test question" in out


def test_cli_calc_input_error(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("builtins.input", lambda _: "not_a_number")
    rc = main(["calc", "ke"])
    out = capsys.readouterr().out
    assert rc == 1
    assert "Error" in out


def test_cli_calc_generic_exception(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def bad_prompt(_: str) -> None:
        raise RuntimeError("surprise")

    monkeypatch.setattr("builtins.input", bad_prompt)
    rc = main(["calc", "ke"])
    out = capsys.readouterr().out
    assert rc == 1
    assert "Error" in out


def test_cli_dashboard_launch(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def fake_run(cmd: list[str], **kwargs: object) -> None:
        raise KeyboardInterrupt()

    monkeypatch.delenv("PYTHONPATH", raising=False)
    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(Path, "exists", lambda self: True)
    rc = main(["dashboard"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "Dashboard stopped" in out


def test_cli_dashboard_streamlit_missing(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def fake_run(cmd: list[str], **kwargs: object) -> None:
        raise FileNotFoundError("streamlit not found")

    monkeypatch.delenv("PYTHONPATH", raising=False)
    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(Path, "exists", lambda self: True)
    rc = main(["dashboard"])
    out = capsys.readouterr().out
    assert rc == 1
    assert "Streamlit not found" in out


def test_cli_modules_end_line(capsys: pytest.CaptureFixture[str]) -> None:
    rc = main(["modules"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "See examples/" in out


def test_import_main() -> None:
    import cds.__main__  # noqa: F401


def test_import_cli_main_module() -> None:
    import cds.cli.__main__  # noqa: F401


# ---------------------------------------------------------------------------
# Monte Carlo: _pi_worker (parallel worker) and edge cases
# ---------------------------------------------------------------------------
def test_pi_worker_counts_inside() -> None:
    inside = _pi_worker((1000, 42))
    assert 700 < inside < 850


def test_pi_worker_no_seed() -> None:
    inside = _pi_worker((500, None))
    assert 0 <= inside <= 500


def test_estimate_pi_zero_samples() -> None:
    """A zero-sample Monte Carlo estimate is undefined and rejected."""
    with pytest.raises(ValueError, match="n_samples"):
        estimate_pi(n_samples=0)


def test_buffon_needle_no_crossings() -> None:
    """Zero crossings make the inverse-probability estimator undefined."""
    with pytest.raises(ArithmeticError, match="zero crossings"):
        buffon_needle(needle_length=0.01, line_spacing=2.0, n_throws=5, seed=1)
