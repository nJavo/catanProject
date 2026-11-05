"""Prioritized fictitious self-play weights for 4-player 1v3 tables.

A 1v3 win rate p is mapped to a pairwise "beats a given opponent seat"
probability q = p / (p + (1 - p) / 3): equal strength (p = 1/4) gives q = 1/2,
like the 2-player setting PFSP was defined for (AlphaStar, Vinyals et al. 2019).

  hard:     f(q) = (1 - q)^2       focus on opponents we lose to
  variance: f(q) = q (1 - q)       focus on opponents near our level
"""
from __future__ import annotations


def pairwise(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return p / (p + (1 - p) / 3)


def weight(p: float, kind: str = "hard") -> float:
    q = pairwise(p)
    if kind == "hard":
        return (1 - q) ** 2
    if kind == "variance":
        return q * (1 - q)
    if kind == "uniform":
        return 1.0
    raise ValueError(kind)


def distribution(rates: dict, kind: str = "hard", floor: float = 0.02) -> dict:
    """rates: mode -> 1v3 win rate. Returns normalized weights (each >= floor share)."""
    if not rates:
        return {}
    w = {m: weight(p, kind) for m, p in rates.items()}
    s = sum(w.values()) or 1.0
    w = {m: max(v / s, floor) for m, v in w.items()}
    s = sum(w.values())
    return {m: v / s for m, v in w.items()}
