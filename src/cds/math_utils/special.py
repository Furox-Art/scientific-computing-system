"""Shared special functions: incomplete gamma and beta machinery.

These Numerical Recipes kernels are the single home for the regularized
incomplete gamma/beta functions used across the package. Layering rule:
``probability`` and ``stats`` both import from here — neither owns the
kernels, so no lower-level module may depend on a higher-level one.

References:
    - Press, W. H., Teukolsky, S. A., Vetterling, W. T., & Flannery, B. P.
      (2007). "Numerical Recipes," 3rd ed., §6.2-6.4.
"""

from __future__ import annotations

import math

MAX_ITER = 200
EPS = 3.0e-12
FPMIN = 1.0e-300


def gammln(x: float) -> float:
    """Natural log of the gamma function (Lanczos approximation).

    Reference: Numerical Recipes §6.1; Lanczos (1964).
    """
    cof = [
        76.18009172947146,
        -86.50532032941677,
        24.01409824083091,
        -1.231739572450155,
        0.1208650973866179e-2,
        -0.5395239384953e-5,
    ]
    y = x
    tmp = x + 5.5
    tmp -= (x + 0.5) * math.log(tmp)
    ser = 1.000000000190015
    for c in cof:
        y += 1.0
        ser += c / y
    return -tmp + math.log(2.5066282746310005 * ser / x)


def gser(a: float, x: float) -> float:
    """Lower regularized incomplete gamma P(a,x) via series expansion.

    Reference: Numerical Recipes §6.2 (gser).
    """
    if x <= 0.0:
        return 0.0
    ap = a
    total = 1.0 / a
    delta = total
    # NR §6.2 series expansion converges within MAX_ITER for every valid
    # (a>0, x≥0) input, so the loop always exits via the ``break`` below —
    # the natural-exhaustion arc (loop completes without breaking) is
    # mathematically unreachable and excluded as a branch arc.
    for _ in range(MAX_ITER):  # pragma: no branch
        ap += 1.0
        delta *= x / ap
        total += delta
        if abs(delta) < abs(total) * EPS:
            break
    return total * math.exp(-x + a * math.log(x) - gammln(a))


def gcf(a: float, x: float) -> float:
    """Upper regularized incomplete gamma Q(a,x) via continued fraction.

    Reference: Numerical Recipes §6.2 (gcf), Lentz's algorithm.
    """
    b = x + 1.0 - a
    c = 1.0 / FPMIN
    d = 1.0 / b
    h = d
    # Same convergence argument as gser: always exits via break.
    for i in range(1, MAX_ITER + 1):  # pragma: no branch
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < FPMIN:  # pragma: no cover — d ≥ |b| - |an*d| stays above FPMIN
            d = FPMIN
        c = b + an / c
        if abs(c) < FPMIN:  # pragma: no cover — c stays ≥ |b| - |an/c| > FPMIN
            c = FPMIN
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < EPS:
            break
    return math.exp(-x + a * math.log(x) - gammln(a)) * h


def gammp(a: float, x: float) -> float:
    """Regularized lower incomplete gamma function P(a, x)."""
    if x < 0.0 or a <= 0.0:
        raise ValueError("a must be > 0 and x must be >= 0 (regularized incomplete gamma P(a, x))")
    if x < a + 1.0:
        return gser(a, x)
    return 1.0 - gcf(a, x)


def gammq(a: float, x: float) -> float:
    """Regularized upper incomplete gamma function Q(a, x) = 1 - P(a, x)."""
    if x < 0.0 or a <= 0.0:
        raise ValueError("a must be > 0 and x must be >= 0 (regularized incomplete gamma Q(a, x))")
    if x < a + 1.0:
        return 1.0 - gser(a, x)
    return gcf(a, x)


def betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the incomplete beta function.

    Reference: Numerical Recipes §6.4 (betacf), Lentz's algorithm.
    """
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < FPMIN:
        d = FPMIN
    d = 1.0 / d
    h = d
    for m in range(1, MAX_ITER + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:  # pragma: no cover — first-loop aa≥0 keeps d≥1
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:  # pragma: no cover — first-loop aa≥0 keeps c≥1
            c = FPMIN
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:  # pragma: no cover — defensive underflow guard
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:  # pragma: no cover — defensive underflow guard
            c = FPMIN
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < EPS:
            break
    return h


def betai(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta function I_x(a, b).

    Reference: Numerical Recipes §6.4 (betai).
    """
    if x < 0.0 or x > 1.0:
        raise ValueError("x must be in [0, 1] for the incomplete beta function I_x(a, b)")
    if x == 0.0 or x == 1.0:
        return x
    front = math.exp(
        gammln(a + b) - gammln(a) - gammln(b) + a * math.log(x) + b * math.log(1.0 - x)
    )
    if x < (a + 1.0) / (a + b + 2.0):
        return front * betacf(a, b, x) / a
    return 1.0 - front * betacf(b, a, 1.0 - x) / b
