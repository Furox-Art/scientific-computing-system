"""PDE solver demo — heat diffusion of a Gaussian blob and a plucked string."""

import math

from cds.pde import solve_heat, solve_wave


def main() -> None:
    # --- Heat equation: Gaussian blob ---
    print("=== Heat equation: Gaussian blob (alpha=0.01) ===")
    nx = 101
    length = 1.0
    alpha = 0.01

    def blob(x: float) -> float:
        return math.exp(-(((x - 0.5) / 0.08) ** 2))

    u0 = [blob(length * i / (nx - 1)) for i in range(nx)]
    dx = length / (nx - 1)

    hot = solve_heat(u0, alpha, length, t_final=0.4, nx=nx, boundary="dirichlet")
    print(f"Dirichlet ends: auto dt={hot.dt:.6f} ({hot.n_steps} steps)")
    print(f"  peak temperature: {max(u0):.6f} -> {max(hot.u_final):.6f}")

    insulated = solve_heat(u0, alpha, length, t_final=0.4, nx=nx, boundary="neumann")
    mass0 = sum(u0) * dx
    mass1 = sum(insulated.u_final) * dx
    drift = abs(mass1 - mass0) / mass0
    print(f"Neumann ends   : auto dt={insulated.dt:.6f} ({insulated.n_steps} steps)")
    print(f"  total heat mass: {mass0:.6f} -> {mass1:.6f} (relative drift {drift:.2e})")

    # --- Wave equation: plucked string ---
    print("\n=== Wave equation: plucked string (c=1.0) ===")
    nx_w = 81
    c = 1.0
    xs_w = [length * i / (nx_w - 1) for i in range(nx_w)]
    center = length / 2.0
    pluck = 0.2

    def triangle(x: float) -> float:
        if x <= center:
            return 2.0 * pluck * x / length
        return 2.0 * pluck * (1.0 - x / length)

    u0_w = [triangle(x) for x in xs_w]
    v0_w = [0.0] * nx_w
    t_final_w = 0.8
    string = solve_wave(u0_w, v0_w, c, length, t_final=t_final_w, nx=nx_w)
    courant = c * string.dt / (length / (nx_w - 1))
    print(
        f"Auto dt={string.dt:.6f} chosen by CFL safety rule "
        f"(Courant number C = {courant:.3f}, {string.n_steps} steps)"
    )
    peak0 = max(abs(u) for u in u0_w)
    peak1 = max(abs(u) for u in string.u_final)
    print(f"Peak |displacement| at t={t_final_w}: {peak0:.4f} -> {peak1:.4f}")
    print(f"Reflected wave inverted the profile (final min = {min(string.u_final):+.4f})")
    print(
        f"String still pinned at both ends: "
        f"{abs(string.u_final[0]) < 1e-12 and abs(string.u_final[-1]) < 1e-12}"
    )


if __name__ == "__main__":
    main()
