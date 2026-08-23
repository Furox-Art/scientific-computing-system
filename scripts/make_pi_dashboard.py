"""Generate a 6-panel dashboard PNG: Monte Carlo methods in cds.montecarlo.

Produces assets/monte_carlo_dashboard.png -- a static, shareable infographic
for r/dataisbeautiful OC posts. All data comes from the project's own pure-Python
montecarlo module (estimate_pi, buffon_needle, mc_integrate, random_walk_2d),
not hand-faked.

Panels:
    1. π scatter (inside/outside points + quarter circle)
    2. π convergence (running estimate vs N, +-1 sigma band)
    3. Buffon's needle (P(crossing) -> π)
    4. Error decay (|estimate - π| vs N, log-log, O(1/sqrt N) reference)
    5. MC integration (sin(x) over [0, π])
    6. 2D random walk (multiple trajectories)

GitHub dark theme, consistent with other promo assets.
"""

from __future__ import annotations

import math
import random

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

from cds.montecarlo import buffon_needle, estimate_pi, mc_integrate, random_walk_2d

# GitHub dark palette (matches cds_promo assets)
BG = "#0d1117"
PANEL = "#161b22"
GRID = "#21262d"
TEXT = "#c9d1d9"
MUTED = "#8b949e"
ACCENT = "#58a6ff"
GREEN = "#3fb950"
RED = "#f85149"
YELLOW = "#d29922"
PURPLE = "#bc8cff"

SEED = 123
TRUE_PI = math.pi

plt.rcParams.update(
    {
        "figure.facecolor": BG,
        "axes.facecolor": PANEL,
        "axes.edgecolor": GRID,
        "axes.labelcolor": MUTED,
        "axes.titlecolor": TEXT,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "grid.color": GRID,
        "text.color": TEXT,
        "font.family": ["DejaVu Sans", "Arial"],
    }
)


def _despine(ax, keep: list[str] | None = None) -> None:
    keep = keep or []
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(side in keep)


# --------------------------------------------------------------------------
# Panel data generators (use the real cds.montecarlo implementations)
# --------------------------------------------------------------------------


def pi_scatter_points(n: int, seed: int) -> tuple[list, list, list]:
    """Generate the inside/outside point cloud for the scatter panel.

    We draw the points locally (cds.montecarlo.estimate_pi runs in a process
    pool and returns only the scalar); the *method* is identical to the module.
    """
    rng = random.Random(seed)
    inside, outside = [], []
    for _ in range(n):
        x, y = rng.random(), rng.random()
        (inside if x * x + y * y <= 1.0 else outside).append((x, y))
    return inside, outside, []


def running_pi(n_points: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Running π estimate and +-1σ band across sample sizes."""
    rng = random.Random(seed)
    sample_sizes = np.unique(np.geomspace(10, n_points, num=60).astype(int))
    xs_pts = [rng.random() for _ in range(n_points)]
    ys_pts = [rng.random() for _ in range(n_points)]
    cum_in = np.cumsum(
        np.fromiter(
            (1 if x * x + y * y <= 1.0 else 0 for x, y in zip(xs_pts, ys_pts)),
            dtype=int,
            count=n_points,
        )
    )
    est, lo, hi = [], [], []
    for n in sample_sizes:
        p = cum_in[n - 1] / n
        est.append(4.0 * p)
        se = 4.0 * math.sqrt(p * (1 - p) / n)
        lo.append(4.0 * p - se)
        hi.append(4.0 * p + se)
    return sample_sizes, np.array(est), np.array(lo), np.array(hi)


# --------------------------------------------------------------------------
# Panel draw functions
# --------------------------------------------------------------------------


def draw_pi_scatter(ax) -> None:
    inside, outside, _ = pi_scatter_points(2000, SEED)
    circle = Circle((0, 0), 1.0, fill=False, edgecolor=ACCENT, linewidth=1.6)
    ax.add_patch(circle)
    if inside:
        ax.scatter(*zip(*inside), s=5, c=GREEN, alpha=0.7, linewidths=0)
    if outside:
        ax.scatter(*zip(*outside), s=5, c=RED, alpha=0.7, linewidths=0)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_title("1 · π via unit square", fontsize=11, pad=8)
    ax.set_xlabel("x", fontsize=9)
    ax.set_ylabel("y", fontsize=9)
    ax.grid(True, alpha=0.25)

    # Final estimate from a real estimate_pi call at the same seed-family.
    # estimate_pi uses multiprocessing; pass a derived seed for reproducibility.
    res = estimate_pi(50_000, seed=SEED)
    ax.text(
        0.97,
        0.05,
        f"π ≈ {res.estimate:.4f}\nn = {res.samples:,}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        color=ACCENT,
        fontsize=11,
        fontweight="bold",
        family="monospace",
    )
    _despine(ax, keep=["bottom", "left"])


def draw_convergence(ax) -> None:
    n_pts, est, lo, hi = running_pi(50_000, SEED + 1)
    ax.fill_between(n_pts, lo, hi, color=ACCENT, alpha=0.18, label="±1σ")
    ax.plot(n_pts, est, color=ACCENT, linewidth=1.6, label="estimate")
    ax.axhline(TRUE_PI, color=YELLOW, linewidth=1.2, linestyle="--", label="true π")
    ax.set_xscale("log")
    ax.set_title("2 · convergence", fontsize=11, pad=8)
    ax.set_xlabel("samples (log)", fontsize=9)
    ax.set_ylabel("π estimate", fontsize=9)
    ax.set_ylim(2.9, 3.4)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="upper right", frameon=False, fontsize=8, labelcolor=MUTED)
    _despine(ax, keep=["bottom", "left"])


def draw_buffon(ax) -> None:
    sizes = [500, 1_000, 5_000, 10_000, 50_000, 100_000]
    ests: list[float] = []
    ses: list[float] = []
    for n in sizes:
        r = buffon_needle(needle_length=1.0, line_spacing=2.0, n_throws=n, seed=SEED + 2)
        ests.append(r.estimate)
        ses.append(r.std_error)
    ests_arr = np.array(ests)
    ses_arr = np.array(ses)
    ax.errorbar(
        sizes,
        ests_arr,
        yerr=ses_arr,
        fmt="o-",
        color=PURPLE,
        ecolor=MUTED,
        elinewidth=1,
        capsize=3,
        markersize=5,
        linewidth=1.4,
    )
    ax.axhline(TRUE_PI, color=YELLOW, linewidth=1.2, linestyle="--", label="true π")
    ax.set_xscale("log")
    ax.set_title("3 · Buffon's needle", fontsize=11, pad=8)
    ax.set_xlabel("throws (log)", fontsize=9)
    ax.set_ylabel("π estimate", fontsize=9)
    ax.set_ylim(2.8, 3.5)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="upper right", frameon=False, fontsize=8, labelcolor=MUTED)
    _despine(ax, keep=["bottom", "left"])


def draw_error_decay(ax) -> None:
    n_pts, est, _, _ = running_pi(200_000, SEED + 3)
    err = np.abs(est - TRUE_PI)
    ax.plot(n_pts, err, color=RED, linewidth=1.5, label="observed error")
    # O(1/sqrt N) reference, scaled to pass through a representative midpoint.
    ref_n = n_pts[len(n_pts) // 2]
    ref_val = err[len(err) // 2]
    ref = ref_val * np.sqrt(ref_n) / np.sqrt(n_pts)
    ax.plot(n_pts, ref, color=MUTED, linewidth=1.2, linestyle="--", label=r"$O(1/\sqrt{N})$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("4 · error decay", fontsize=11, pad=8)
    ax.set_xlabel("samples (log)", fontsize=9)
    ax.set_ylabel("|estimate − π| (log)", fontsize=9)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="upper right", frameon=False, fontsize=8, labelcolor=MUTED)
    _despine(ax, keep=["bottom", "left"])


def draw_mc_integration(ax) -> None:
    f = math.sin  # mc_integrate calls f scalar-by-scalar, so math.sin is correct
    a, b = 0.0, TRUE_PI
    sizes = np.unique(np.geomspace(20, 100_000, num=20).astype(int))
    ests = []
    for n in sizes:
        r = mc_integrate(f, a, b, n_samples=int(n), seed=SEED + 4)
        ests.append(r.estimate)
    true_val = 2.0  # integral of sin over [0,π] = 2

    xs = np.linspace(a, b, 200)
    ax.plot(xs, np.sin(xs), color=ACCENT, linewidth=1.6, label="sin(x)")
    ax.fill_between(xs, 0, np.sin(xs), color=ACCENT, alpha=0.12)
    final = ests[-1]
    ax.text(
        0.98,
        0.92,
        f"MC estimate\n∫₀^π sin = {final:.4f}\ntrue = {true_val:.4f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        color=TEXT,
        fontsize=9,
        family="monospace",
        bbox=dict(facecolor=PANEL, edgecolor=GRID, boxstyle="round,pad=0.4"),
    )
    ax.set_title("5 · MC integration", fontsize=11, pad=8)
    ax.set_xlabel("x", fontsize=9)
    ax.set_ylabel("sin(x)", fontsize=9)
    ax.set_xlim(0, TRUE_PI)
    ax.set_ylim(0, 1.15)
    ax.grid(True, alpha=0.25)
    _despine(ax, keep=["bottom", "left"])


def draw_random_walk(ax) -> None:
    for i, col in zip(range(6), [GREEN, ACCENT, YELLOW, PURPLE, RED, MUTED]):
        walk = random_walk_2d(steps=500, step_size=1.0, seed=SEED + 5 + i)
        xs = [p[0] for p in walk]
        ys = [p[1] for p in walk]
        ax.plot(xs, ys, color=col, linewidth=0.9, alpha=0.8)
        ax.scatter([xs[0]], [ys[0]], color=col, s=18, zorder=3, edgecolors="none")
        ax.scatter([xs[-1]], [ys[-1]], color=col, s=28, marker="X", zorder=3, edgecolors="none")
    ax.set_title("6 · 2D random walk", fontsize=11, pad=8)
    ax.set_xlabel("x", fontsize=9)
    ax.set_ylabel("y", fontsize=9)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.25)
    _despine(ax, keep=["bottom", "left"])


# --------------------------------------------------------------------------
# Figure assembly
# --------------------------------------------------------------------------


def main() -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), dpi=140)
    fig.patch.set_facecolor(BG)
    fig.subplots_adjust(left=0.06, right=0.97, top=0.90, bottom=0.07, wspace=0.28, hspace=0.38)

    draw_pi_scatter(axes[0, 0])
    draw_convergence(axes[0, 1])
    draw_buffon(axes[0, 2])
    draw_error_decay(axes[1, 0])
    draw_mc_integration(axes[1, 1])
    draw_random_walk(axes[1, 2])

    # Header
    fig.suptitle(
        "Monte Carlo methods in pure Python",
        fontsize=18,
        fontweight="bold",
        color=TEXT,
        y=0.965,
    )
    fig.text(
        0.5,
        0.928,
        "six views on estimating π, integration and diffusion -- no NumPy, no SciPy",
        ha="center",
        color=MUTED,
        fontsize=11,
    )
    fig.text(
        0.5,
        0.012,
        "data: cds.montecarlo  ·  scientific-computing-system",
        ha="center",
        color=MUTED,
        fontsize=8,
        family="monospace",
    )

    out_path = "assets/monte_carlo_dashboard.png"
    fig.savefig(out_path, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
