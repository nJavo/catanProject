"""Small statistics helpers (no scipy)."""
from __future__ import annotations

import math


def wilson(k: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion k/n."""
    if n <= 0:
        return (0.0, 1.0)
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, centre - half), min(1.0, centre + half))


def mean_ci(xs, z: float = 1.96) -> dict:
    """Mean, sample sd and normal-approximation 95% CI of the mean."""
    n = len(xs)
    if n == 0:
        return {"mean": None, "sd": None, "ci95": None, "n": 0}
    m = sum(xs) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1)) if n > 1 else 0.0
    h = z * sd / math.sqrt(n) if n > 1 else 0.0
    return {"mean": m, "sd": sd, "ci95": [m - h, m + h], "n": n}


def binom_two_sided_p(k: int, n: int, p: float = 0.5) -> float:
    """Exact two-sided binomial test p-value (sum of outcomes no more likely than k)."""
    if n == 0:
        return 1.0
    pmf = [math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(n + 1)]
    pk = pmf[k]
    return min(1.0, sum(x for x in pmf if x <= pk * (1 + 1e-9)))
