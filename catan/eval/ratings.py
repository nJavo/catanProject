"""Ratings from multiplayer results.

Each game is a ranking (winner first, then final VP; equal VP tie). It is
decomposed into all pairwise outcomes and a Bradley-Terry model is fitted by
the MM algorithm (Hunter 2004). Strengths are reported on the Elo scale
(400 * log10), centred so the mean rating is 1500. Unlike sequential Elo
(``elo_system/elo.py``) the fit does not depend on game order. A bootstrap
over games gives 95% intervals.
"""
from __future__ import annotations

import math
import random


def pairwise_outcomes(rankings):
    """rankings: list of [(label, score), ...] per game (higher score is better).
    Returns {(a, b): wins of a over b} with ties counted 0.5 each way.
    Same-label pairs are skipped."""
    w: dict = {}
    for game in rankings:
        for i, (a, sa) in enumerate(game):
            for b, sb in game[i + 1:]:
                if a == b:
                    continue
                if sa > sb:
                    w[(a, b)] = w.get((a, b), 0) + 1
                elif sb > sa:
                    w[(b, a)] = w.get((b, a), 0) + 1
                else:
                    w[(a, b)] = w.get((a, b), 0) + 0.5
                    w[(b, a)] = w.get((b, a), 0) + 0.5
    return w


def bradley_terry(wins: dict, labels, iters: int = 500, prior: float = 0.5) -> dict:
    """MM fit. ``prior`` adds that many virtual wins and losses vs every other
    label so ratings stay finite when someone never loses."""
    labels = list(labels)
    gam = {x: 1.0 for x in labels}
    W = {x: 0.0 for x in labels}
    N = {}
    for a in labels:
        for b in labels:
            if a == b:
                continue
            wab = wins.get((a, b), 0) + prior
            W[a] += wab
            N[(a, b)] = N.get((a, b), 0) + wab
            N[(b, a)] = N.get((b, a), 0) + wab
    for _ in range(iters):
        new = {}
        for a in labels:
            den = sum(N[(a, b)] / (gam[a] + gam[b]) for b in labels if b != a)
            new[a] = W[a] / den if den > 0 else gam[a]
        g = math.exp(sum(math.log(v) for v in new.values()) / len(new))
        delta = max(abs(new[a] / g - gam[a]) for a in labels)
        gam = {a: new[a] / g for a in labels}
        if delta < 1e-10:
            break
    return {a: 1500 + 400 * math.log10(gam[a]) for a in labels}


def ratings(rankings, labels, bootstrap: int = 200, seed: int = 0) -> dict:
    labels = sorted(set(labels))
    if len(labels) < 2:
        return {x: {"elo": 1500.0, "ci95": [1500.0, 1500.0]} for x in labels}
    point = bradley_terry(pairwise_outcomes(rankings), labels)
    rng = random.Random(seed)
    samples = {x: [] for x in labels}
    for _ in range(bootstrap):
        rs = [rankings[rng.randrange(len(rankings))] for _ in rankings]
        r = bradley_terry(pairwise_outcomes(rs), labels, iters=200)
        for x in labels:
            samples[x].append(r[x])
    out = {}
    for x in labels:
        s = sorted(samples[x])
        ci = [s[int(0.025 * len(s))], s[min(len(s) - 1, int(0.975 * len(s)))]] if s else [point[x]] * 2
        out[x] = {"elo": round(point[x], 1), "ci95": [round(ci[0], 1), round(ci[1], 1)]}
    return out
