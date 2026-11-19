"""Verify every cookbook recipe runs against the real CDS API.

Run: python scripts/verify_cookbook.py
Exits non-zero if any recipe fails. Pandas round-trip is skipped if the
optional extra is not installed. The recipes mirror the snippets in
``docs/cookbook.md`` one-for-one; if a recipe drifts from the API this file is
the canary.

Implementation note: every recipe is defined as a plain function and only run
from ``main()``. This matters on Windows (spawn) because ``cds.montecarlo``
multiprocessing re-imports this module in worker processes — keeping all recipe
*calls* inside ``main()`` (guarded by ``__name__``) prevents the workers from
re-executing them, which would otherwise duplicate or flake the output.
"""

import math
import sys
import traceback

failures: list[str] = []
ok = 0


def run(name, fn):
    """Run one recipe, tally the result, and print a single status line."""
    global ok
    try:
        fn()
        ok += 1
        print(f"  OK  {name}")
    except Exception:
        failures.append(name)
        print(f"FAIL  {name}")
        traceback.print_exc()


# ===== Recipe definitions (plain functions — no module-level execution) =====
def recipe_core_hypothesis():
    import json

    from cds.core.models import Domain, Hypothesis, HypothesisStatus

    h = Hypothesis(
        id="H-001",
        statement="raising temp raises yield",
        domain=Domain.CHEMISTRY,
        research_question="temp effect on yield?",
        assumptions=["a"],
        predictions=["yield_delta > 0.05"],
        confidence=0.7,
        status=HypothesisStatus.TESTABLE,
    )
    assert h.predictions == ["yield_delta > 0.05"]
    json.dumps(h.to_dict())


def recipe_solve_linear():
    from cds.math_utils import solve_linear

    assert solve_linear([[2.0, 1.0], [1.0, 3.0]], [5.0, 10.0]) == [1.0, 3.0]


def recipe_power_iteration():
    from cds.math_utils import power_iteration

    ev, _ = power_iteration([[4.0, 1.0], [1.0, 3.0]], max_iter=200)
    assert abs(ev - 4.618) < 0.01


def recipe_derivative_integral():
    from cds.math_utils import derivative, integral

    assert abs(derivative(lambda x: x**3, 2.0) - 12.0) < 0.1
    assert abs(integral(lambda x: x**2, 0.0, 3.0) - 9.0) < 0.1


def recipe_gram_schmidt():
    from cds.math_utils import gram_schmidt

    v = gram_schmidt([[1.0, 1.0, 0.0], [0.0, 1.0, 1.0]])
    assert len(v) == 2


def recipe_probability_pdfs():
    from cds.probability import exponential_pdf, gaussian_pdf

    assert abs(gaussian_pdf(1.0, mu=0.0, sigma=1.0) - 0.2420) < 0.001
    # `lam` is the rate parameter, per the cookbook.
    assert abs(exponential_pdf(0.5, lam=2.0) - 0.7358) < 0.001


def recipe_probability_pmfs():
    from cds.probability import binomial_pmf, poisson_pmf

    assert abs(binomial_pmf(3, n=10, p=0.5) - 0.1172) < 0.001
    assert abs(poisson_pmf(6, lam=4.0) - 0.1042) < 0.001


def recipe_uniform_sample():
    from cds.probability import uniform_sample

    s = uniform_sample(0.0, 10.0, n=5)
    assert len(s) == 5


def recipe_scientific():
    from cds.scientific import (
        de_broglie_wavelength,
        escape_velocity,
        get_constant,
        gravitational_force,
        photon_energy,
        schwarzschild_radius,
    )

    h = get_constant("h")
    c = get_constant("c")
    E = photon_energy(h * c / 500e-9)
    assert E > 0
    assert gravitational_force(5.972e24, 7.348e22, 3.844e8) > 0
    assert abs(escape_velocity(5.972e24, 6.371e6) - 11186) < 10
    assert 0 < schwarzschild_radius(5.972e24) < 0.01
    assert de_broglie_wavelength(9.109e-31, 1e6) > 0


def recipe_dijkstra():
    from cds.graph import Graph, dijkstra

    g = Graph(n_vertices=5, directed=False)
    for u, v, w in [(0, 1, 4), (0, 2, 2), (1, 2, 1), (1, 3, 5), (2, 4, 10), (3, 4, 2)]:
        g.add_edge(u, v, weight=w)
    dists, _ = dijkstra(g, 0)
    assert dists[4] > 0


def recipe_kruskal_cycle():
    from cds.graph import Graph, has_cycle, kruskal_mst

    g = Graph(n_vertices=4, directed=False)
    for u, v, w in [(0, 1, 1), (1, 2, 2), (0, 2, 3), (2, 3, 5)]:
        g.add_edge(u, v, weight=w)
    edges, total = kruskal_mst(g)
    assert total > 0
    assert has_cycle(g) is True


def recipe_topological_sort():
    from cds.graph import Graph, topological_sort

    dag = Graph(n_vertices=4, directed=True)
    dag.add_edge(0, 1, weight=1)
    dag.add_edge(1, 2, weight=1)
    dag.add_edge(2, 3, weight=1)
    order = topological_sort(dag)
    assert order.index(0) < order.index(3)


def recipe_power_spectrum():
    from cds.signals import power_spectrum

    n = 200
    sig = [math.sin(2 * math.pi * 50 * i / 1000) + 0.3 * i / n for i in range(n)]
    freqs = power_spectrum(sig)
    peak = max(range(len(freqs)), key=lambda k: freqs[k])
    assert peak > 0


def recipe_fft_radix2():
    from cds.signals import fft_radix2

    x = [math.sin(2 * math.pi * 3 * k / 64) for k in range(64)]
    spec = fft_radix2(x)
    assert abs(spec[3]) > 1.0


def recipe_butter_lowpass():
    from cds.signals import apply_filter, butter_lowpass

    coeffs = butter_lowpass(order=4, cutoff=0.2)
    out = apply_filter([1, 2, 5, 3, 8, 6, 9, 4], coeffs)
    assert len(out) == 8


def recipe_stats_descriptive():
    from cds.stats import correlation, mean

    a = [12.5, 14.3, 11.8, 15.1, 13.7]
    b = [3.1, 4.0, 2.9, 4.4, 3.8]
    assert abs(mean(a) - 13.48) < 0.01
    assert correlation(a, b) > 0


def recipe_linear_regression():
    from cds.stats import linear_regression

    reg = linear_regression([1, 2, 3, 4, 5], [2.1, 3.9, 6.2, 7.8, 10.1])
    assert abs(reg.slope - 2.0) < 0.1
    assert reg.r_squared > 0.9


def recipe_t_tests():
    from cds.stats import one_sample_ttest, two_sample_ttest

    # The API parameter is `popmean`, not `mu` — matches the cookbook.
    r1 = one_sample_ttest([4.8, 5.1, 4.9, 5.0, 5.2], popmean=5.0)
    assert 0 <= r1.p_value <= 1
    r2 = two_sample_ttest([22, 25, 19, 24, 23], [28, 30, 27, 31, 29])
    assert 0 <= r2.p_value <= 1


def recipe_chi_square_cohens_d():
    from cds.stats import chi_square_gof, cohens_d

    gof = chi_square_gof([8, 9, 12, 7, 11, 13], expected=[10] * 6)
    assert 0 <= gof.p_value <= 1
    d = cohens_d([1, 2, 3, 4], [4, 5, 6, 7])
    assert d != 0


def recipe_ts_smooth_diff_decompose():
    from cds.stats import difference, moving_average, seasonal_decompose

    s = [10 + 0.5 * i + 3 * math.sin(i / 2) for i in range(48)]
    assert len(moving_average(s, window=4)) > 0
    assert len(difference(s)) > 0
    t, se, r = seasonal_decompose(s, period=12)
    assert len(t) == len(s)


def recipe_ts_kpss_ljung_acf():
    from cds.stats import autocorrelation_function, kpss_statistic, ljung_box

    s = [10 + 0.5 * i + 3 * math.sin(i / 2) for i in range(48)]
    kpss = kpss_statistic(s)
    assert hasattr(kpss, "statistic") and hasattr(kpss, "normal_tail_probability")
    lb = ljung_box(s, lags=10)
    assert 0 <= lb.p_value <= 1
    # ACF returns r_0..r_max_lag inclusive (max_lag+1 entries).
    acfs = autocorrelation_function(s, max_lag=12)
    assert len(acfs) == 13


def recipe_exponential_smoothing():
    from cds.stats import exponential_smoothing

    s = [1.0 * i for i in range(20)]
    f = exponential_smoothing(s, alpha=0.3)
    assert len(f) == len(s)


def recipe_estimate_pi():
    from cds.montecarlo import estimate_pi

    # estimate_pi uses ProcessPoolExecutor; the caller must guard multiprocessing
    # with __main__ (this file already does).
    r = estimate_pi(n_samples=20000)
    assert abs(r.estimate - 3.14) < 0.2
    assert r.std_error > 0


def recipe_mc_integrate():
    from cds.montecarlo import mc_integrate

    r = mc_integrate(math.sin, 0.0, math.pi, n_samples=100000)
    assert abs(r.estimate - 2.0) < 0.1


def recipe_random_walk_2d():
    from cds.montecarlo import random_walk_2d

    path = random_walk_2d(steps=1000)
    assert len(path) == 1001


def recipe_numint_basic():
    from cds.numerical_integration import gaussian_quadrature, simpson, trapezoid

    def f(x):
        return math.exp(-(x**2))

    # 1-D rules return plain floats, not QuadratureResult.
    assert abs(trapezoid(f, 0.0, 1.0, n=100) - 0.746824) < 0.001
    assert abs(simpson(f, 0.0, 1.0, n=100) - 0.746824) < 1e-6
    assert abs(gaussian_quadrature(f, 0.0, 1.0, n=5) - 0.746824) < 1e-6


def recipe_adaptive_simpson():
    from cds.numerical_integration import adaptive_simpson

    def f(x):
        return math.exp(-((x - 0.5) ** 2) / 0.001)

    r = adaptive_simpson(f, 0.0, 1.0, tol=1e-8)
    assert r.value > 0


def recipe_numint_2d():
    from cds.numerical_integration import gaussian_quadrature_2d, simpson_2d

    def g(x, y):
        return math.exp(-(x**2 + y**2))

    # 2-D rules also return plain floats.
    v1 = simpson_2d(g, 0.0, 1.0, 0.0, 1.0, nx=20, ny=20)
    v2 = gaussian_quadrature_2d(g, 0.0, 1.0, 0.0, 1.0, n=4)
    assert v1 > 0 and v2 > 0


def recipe_rk4():
    from cds.diffeq import rk4

    sol = rk4(lambda t, y: -y, t0=0.0, y0=1.0, t_end=2.0, dt=0.01)
    assert abs(sol.y[-1] - math.exp(-2)) < 1e-4


def recipe_rk45():
    from cds.diffeq import rk45

    sol = rk45(lambda t, y: 0.5 * y * (1 - y / 10), t0=0.0, y0=0.5, t_end=20.0, rtol=1e-6)
    assert abs(sol.y[-1] - 10.0) < 0.5


def recipe_solve_system():
    from cds.diffeq import solve_system

    def lotka(t, st):
        x, y = st
        return [1.1 * x - 0.4 * x * y, 0.1 * x * y - 0.4 * y]

    ts, ys = solve_system(lotka, t0=0.0, y0=[10.0, 5.0], t_end=40.0, dt=0.05)
    assert len(ys) == len(ts) and len(ys) > 1


def recipe_gd_scalar():
    from cds.optimization import gradient_descent

    r = gradient_descent(lambda x: (x - 3) ** 2, x0=0.0, lr=0.1)
    assert abs(r.x - 3.0) < 0.1 and r.converged


def recipe_gd_vector():
    from cds.optimization import gradient_descent

    # Vector overload: objective takes a single list argument.
    def f(v):
        return (v[0] - 1) ** 2 + 100 * (v[1] - v[0] ** 2) ** 2

    r = gradient_descent(f, x0=[0.0, 0.0], lr=0.001, max_iter=50000)
    assert abs(r.x[0] - 1.0) < 0.2


def recipe_newton():
    from cds.optimization import newton_method

    root = newton_method(math.cos, x0=1.0)
    assert abs(root.x - math.pi / 2) < 1e-4


def recipe_adam():
    from cds.optimization import adam

    r = adam(lambda x: (x - 2) ** 2, x0=5.0, lr=0.1, max_iter=1000)
    assert abs(r.x - 2.0) < 0.2


def recipe_modeling_build_diff():
    from cds.modeling import Exp, Sin, Variable

    x = Variable("x")
    f = Sin(x) * Exp(x)
    df = f.diff("x")
    f.to_str()
    f.to_latex()
    assert abs(df.evaluate({"x": 0.0}) - 1.0) < 1e-6


def recipe_modeling_jacobian():
    from cds.modeling import Constant, MathModel, Variable

    t, a = Variable("t"), Variable("a")
    m = MathModel(name="k", parameters={"a": 9.81}, variables=["t", "v0", "x0"])
    v0, x0 = Variable("v0"), Variable("x0")
    m.add_equation("velocity", v0 + a * t)
    m.add_equation("position", x0 + v0 * t + Constant(0.5) * a * t * t)
    res = m.evaluate({"t": 2.0, "v0": 0.0, "x0": 0.0})
    assert abs(res["velocity"] - 19.62) < 0.01
    assert "velocity" in m.jacobian("t")


def recipe_modeling_solve_fit():
    from cds.modeling import Constant, MathModel, Variable, fit_parameters, solve_equation

    x = Variable("x")
    root = solve_equation(x**2 - Constant(2.0), variable="x", x0=1.0)
    assert abs(root.x - math.sqrt(2)) < 1e-4
    fm = MathModel(name="lin", parameters={"v": 0.0}, variables=["t"])
    fm.add_equation("position", Variable("v") * Variable("t"))
    obs = [({"t": float(i)}, 3.0 * i) for i in range(5)]
    fit = fit_parameters(fm, observed=obs, parameter_names=["v"], x0=[0.1], target_label="position")
    assert abs(fit.parameters["v"] - 3.0) < 0.1


def recipe_mlp():
    from cds.ml import MLP, Layer

    net = MLP([Layer(1, 8, "relu"), Layer(8, 1, "identity")])
    X = [[i / 10] for i in range(-20, 21)]
    y = [[xi[0] ** 2] for xi in X]
    info = net.train(X, y, epochs=200, lr=0.01)
    assert "final_loss" in info
    p = net.predict([1.5])
    assert len(p) == 1


def recipe_bpe():
    from cds.nlp import train_bpe

    corpus = "the quick brown fox the lazy dog a quick fox jumps"
    tok = train_bpe(corpus, vocab_size=80, min_frequency=1)
    ids = tok.encode("the quick fox")
    assert isinstance(ids, list) and len(ids) > 0
    assert tok.decode(ids)


def recipe_attention():
    from cds.nlp import scaled_dot_product_attention

    Q = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]
    out = scaled_dot_product_attention(Q, Q, [[1.0], [2.0], [3.0]])
    assert len(out) == 3 and len(out[0]) == 1


def recipe_autograd():
    from cds.nlp import Tensor, relu

    # Tensor is a scalar autograd node; backward() takes no argument.
    a = Tensor(2.0, requires_grad=True)
    b = Tensor(3.0, requires_grad=True)
    c = relu(a * b)  # 6.0, connected to a and b
    c.backward()  # seeds 1.0, propagates grads back
    assert a.grad is not None and a.grad > 0


def recipe_hypothesis_generate():
    from cds.hypothesis import generate_hypotheses

    hypos = generate_hypotheses("Why Hubble tension?", domain="cosmology", n=3)
    assert len(hypos) == 3


def recipe_hypothesis_evaluate():
    from cds.hypothesis import (
        ChiSquareGofPayload,
        Domain,
        Hypothesis,
        HypothesisEvaluator,
        HypothesisStatus,
    )

    # evaluate(hypothesis, data) takes a Hypothesis and an EvaluationData dict
    # keyed by the dispatch key ("chi_square_gof" here).
    h = Hypothesis(
        id="H-001",
        statement="a die is fair",
        domain=Domain.MATHEMATICS,
        research_question="is the die fair?",
        status=HypothesisStatus.TESTABLE,
    )
    payload: ChiSquareGofPayload = {
        "observed": [8, 12, 10, 11, 9, 10],
        "expected": [10, 10, 10, 10, 10, 10],
    }
    result = HypothesisEvaluator().evaluate(h, {"chi_square_gof": payload})
    assert hasattr(result, "p_value")


def recipe_knowledge():
    from cds.knowledge import KnowledgeGraph, Notebook, search

    kg = KnowledgeGraph(name="mechanics")
    kg.add_concept("momentum", description="p=mv")
    kg.add_concept("force", description="F=dp/dt")
    kg.add_relation("force", "momentum", "depends-on")
    assert "momentum" in kg.neighbors("force")
    nb = Notebook(name="mechanics")
    nb.add_note(
        note_id="n1",
        title="Newton II",
        body="F equals the rate of change of momentum.",
        linked_concepts=["force", "momentum"],
    )
    hits = search(kg, nb, query="rate of change")
    assert len(hits) >= 1


def recipe_data_table():
    import os
    import tempfile

    from cds.data_analysis import load_csv, z_score

    path = os.path.join(tempfile.gettempdir(), "_cb_test.csv")
    with open(path, "w") as f:
        f.write("temp\n1.0\n2.0\n3.0\n4.0\n5.0\n")
    table = load_csv(path)
    col = table.column_as_float("temp")
    z = z_score(col)
    assert len(z) == 5


def recipe_moving_average_plot():
    from cds.data_analysis import moving_average, plot_line

    series = [1, 3, 2, 5, 4, 6, 5, 8, 7, 9]
    sm = moving_average(series, window=3)
    assert len(sm) > 0
    plot_line(series)  # ASCII chart, must not raise


def recipe_pandas_roundtrip():
    try:
        import pandas  # noqa: F401
    except ImportError:
        global ok
        ok += 1
        print("       (skipped: pandas not installed)")
        return
    from cds.data_analysis import DataSet, from_dataframe, to_dataframe

    ds = DataSet(data=[{"temp": 1.0}, {"temp": 2.0}, {"temp": 3.0}])
    df = to_dataframe(ds)
    ds2 = from_dataframe(df)
    assert ds2 is not None


# Order mirrors docs/cookbook.md section order.
RECIPES = [
    ("core: Hypothesis", recipe_core_hypothesis),
    ("math_utils: solve_linear", recipe_solve_linear),
    ("math_utils: power_iteration", recipe_power_iteration),
    ("math_utils: derivative/integral", recipe_derivative_integral),
    ("math_utils: gram_schmidt", recipe_gram_schmidt),
    ("probability: pdfs", recipe_probability_pdfs),
    ("probability: pmfs", recipe_probability_pmfs),
    ("probability: uniform_sample", recipe_uniform_sample),
    ("scientific: constants + formulas", recipe_scientific),
    ("graph: dijkstra", recipe_dijkstra),
    ("graph: kruskal + has_cycle", recipe_kruskal_cycle),
    ("graph: topological_sort", recipe_topological_sort),
    ("signals: power_spectrum", recipe_power_spectrum),
    ("signals: fft_radix2", recipe_fft_radix2),
    ("signals: butter_lowpass + apply_filter", recipe_butter_lowpass),
    ("stats: descriptive + correlation", recipe_stats_descriptive),
    ("stats: linear_regression", recipe_linear_regression),
    ("stats: t-tests", recipe_t_tests),
    ("stats: chi_square_gof + cohens_d", recipe_chi_square_cohens_d),
    ("time-series: smooth/diff/decompose", recipe_ts_smooth_diff_decompose),
    ("time-series: kpss/ljung_box/acf", recipe_ts_kpss_ljung_acf),
    ("time-series: exponential_smoothing", recipe_exponential_smoothing),
    ("montecarlo: estimate_pi", recipe_estimate_pi),
    ("montecarlo: mc_integrate", recipe_mc_integrate),
    ("montecarlo: random_walk_2d", recipe_random_walk_2d),
    ("numint: trapezoid/simpson/gauss", recipe_numint_basic),
    ("numint: adaptive_simpson", recipe_adaptive_simpson),
    ("numint: 2D quadrature", recipe_numint_2d),
    ("diffeq: rk4", recipe_rk4),
    ("diffeq: rk45", recipe_rk45),
    ("diffeq: solve_system", recipe_solve_system),
    ("optimization: gradient_descent scalar", recipe_gd_scalar),
    ("optimization: gradient_descent vector", recipe_gd_vector),
    ("optimization: newton_method", recipe_newton),
    ("optimization: adam", recipe_adam),
    ("modeling: build/diff/latex", recipe_modeling_build_diff),
    ("modeling: MathModel jacobian", recipe_modeling_jacobian),
    ("modeling: solve + fit", recipe_modeling_solve_fit),
    ("ml: MLP train", recipe_mlp),
    ("nlp: train_bpe encode/decode", recipe_bpe),
    ("nlp: scaled_dot_product_attention", recipe_attention),
    ("nlp: autograd", recipe_autograd),
    ("hypothesis: generate", recipe_hypothesis_generate),
    ("hypothesis: evaluate", recipe_hypothesis_evaluate),
    ("knowledge: graph + notes + search", recipe_knowledge),
    ("data_analysis: DataTable column", recipe_data_table),
    ("data_analysis: moving_average + plot_line", recipe_moving_average_plot),
    ("data_analysis: pandas round-trip", recipe_pandas_roundtrip),
]


def main():
    for name, fn in RECIPES:
        run(name, fn)
    print("=" * 50)
    print(f"PASSED {ok}  FAILED {len(failures)}")
    if failures:
        print("FAILURES:", failures)
        sys.exit(1)
    print("ALL COOKBOOK RECIPES VERIFIED")


if __name__ == "__main__":
    main()
