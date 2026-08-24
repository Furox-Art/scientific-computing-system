"""Interpolation demo — 1-D linear vs nearest and 2-D bilinear against exact values."""

import math

from cds.interpolate import interp1d, interp2d


def main() -> None:
    # --- 1-D: sine sampled coarsely ---
    print("=== 1-D Interpolation: coarse sine samples ===")
    n_knots = 7
    xs = [2.0 * math.pi * i / (n_knots - 1) for i in range(n_knots)]
    ys = [math.sin(x) for x in xs]
    linear = interp1d(xs, ys, kind="linear")
    nearest = interp1d(xs, ys, kind="nearest")

    print(f"{'query':>8} {'exact':>10} {'linear':>10} {'nearest':>10}")
    for q in [0.3, 1.0, math.pi / 2, 2.5, 4.0, 5.5]:
        print(f"{q:8.3f} {math.sin(q):10.6f} {linear(q):10.6f} {nearest(q):10.6f}")

    dense = [2.0 * math.pi * i / 400 for i in range(401)]
    lin_err = max(abs(math.sin(q) - linear(q)) for q in dense)
    near_err = max(abs(math.sin(q) - nearest(q)) for q in dense)
    print("\nMax |error| over 401 query points:")
    print(f"  linear : {lin_err:.6f}")
    print(f"  nearest: {near_err:.6f}")

    # --- 2-D: bilinear on a small grid, evaluated off-grid ---
    print("\n=== 2-D Bilinear Interpolation ===")

    def surface(x: float, y: float) -> float:
        return math.sin(x) * math.cos(y)

    gx = [0.0, 0.5, 1.0, 1.5, 2.0]
    gy = [-1.0, -0.5, 0.0, 0.5, 1.0]
    z = [[surface(x, y) for y in gy] for x in gx]
    bilinear = interp2d(gx, gy, z)

    qx, qy = 0.83, -0.37
    exact = surface(qx, qy)
    approx = bilinear(qx, qy)
    print(f"Off-grid point ({qx}, {qy}):")
    print(f"  exact    = {exact:+.6f}")
    print(f"  bilinear = {approx:+.6f}")
    print(f"  error    = {abs(approx - exact):.6f}")

    nx_node = 1
    ny_node = 2
    node_exact = surface(gx[nx_node], gy[ny_node])
    node_approx = bilinear(gx[nx_node], gy[ny_node])
    print(f"Grid node ({gx[nx_node]}, {gy[ny_node]}):")
    print(f"  exact    = {node_exact:+.6f}")
    print(f"  bilinear = {node_approx:+.6f} (reproduces the table exactly)")


if __name__ == "__main__":
    main()
