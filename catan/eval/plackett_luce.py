"""Plackett-Luce ratings from full 4-player finishing orders.

A game is a ranking [(label, score), ...] (one entry per seat, higher score is better; the
``tournament.rank_scores`` key: winner +100, then VP). Plackett-Luce reads it as a sequence of
choices: the winner is picked from all seats with probability g_w / sum(g), second from the rest,
and so on. Bradley-Terry over the 6 pairwise outcomes (catan.eval.ratings) uses the same order but
treats the 6 pairs as independent games.

Ties (equal VP among losers) use Breslow's rule: each member of a tied group is "chosen" from the
same risk set. A tie among all remaining seats carries no order information and is skipped.
Same-label seats (1v3 tables) are separate items with one shared strength.

Fit: MM (Hunter 2004, section 5). ``prior`` adds that many virtual wins and losses per label
against a virtual player of average strength, the same convention as ``bradley_terry``'s prior.
Ratings are on the Elo scale (400 log10 g), mean 1500. In a 2-player game PL and BT coincide, so
the two scales are comparable. ``seat_effects`` also fits a multiplicative strength per seat
(item strength = g_label * s_seat), which needs the seat order of every ranking.
"""
from __future__ import annotations

import math

import numpy as np


class Stages:
    """Sparse choice stages. One row per (stage, item in its risk set): stage index, label index,
    seat, and whether the item is chosen at that stage. ``game[s]`` maps stage -> game index."""

    def __init__(self, rankings, labels, max_stages: int | None = None):
        idx = {x: i for i, x in enumerate(labels)}
        st, lab, seat, ch, game = [], [], [], [], []
        s = 0
        for g, rk in enumerate(rankings):
            rest = sorted([(idx[a], sc, q) for q, (a, sc) in enumerate(rk) if a in idx], key=lambda t: -t[1])
            k = 0
            while len(rest) > 1 and (max_stages is None or k < max_stages):
                top = rest[0][1]
                ng = sum(1 for t in rest if t[1] == top)
                if ng == len(rest):
                    break  # all remaining tied: no order information
                for j, (i, sc, q) in enumerate(rest):
                    st.append(s)
                    lab.append(i)
                    seat.append(q)
                    ch.append(j < ng)
                game.append(g)
                rest = rest[ng:]
                s += 1
                k += 1
        self.st, self.lab, self.seat = np.array(st, int), np.array(lab, int), np.array(seat, int)
        self.ch = np.array(ch, bool)
        self.game = np.array(game, int)
        self.n_stages, self.n_labels = s, len(labels)
        self.n_seats = int(self.seat.max()) + 1 if len(self.seat) else 1
        self.k = np.bincount(self.st[self.ch], minlength=s).astype(float)  # chosen per stage


def fit_stages(S: Stages, wts=None, prior=0.5, iters=1000, seat_effects=False, tol=1e-10, init=None):
    """MM fit with per-stage weights (bootstrap multiplicities). Returns mean-centred
    (log10 label strengths, log10 seat strengths). ``init``: starting log10 strengths (warm start)."""
    L, Q = S.n_labels, S.n_seats
    wts = np.ones(S.n_stages) if wts is None else wts
    gam, seat = (np.ones(L) if init is None else 10.0 ** np.asarray(init, float)), np.ones(Q)
    wi = wts[S.st]
    W = np.bincount(S.lab[S.ch], wi[S.ch], minlength=L) + prior
    WS = np.bincount(S.seat[S.ch], wi[S.ch], minlength=Q)
    kw = wts * S.k
    for _ in range(iters):
        d = np.bincount(S.st, gam[S.lab] * seat[S.seat], minlength=S.n_stages)
        f = np.where(d > 0, kw / np.maximum(d, 1e-300), 0.0)
        den = np.bincount(S.lab, f[S.st] * seat[S.seat], minlength=L) + prior * 2.0 / (gam + 1.0)
        new = np.where(den > 0, W / np.maximum(den, 1e-300), gam)
        new /= math.exp(np.log(new).mean())
        delta = float(np.abs(np.log(new / gam)).max())  # relative change (log scale)
        gam = new
        if seat_effects:
            d = np.bincount(S.st, gam[S.lab] * seat[S.seat], minlength=S.n_stages)
            f = np.where(d > 0, kw / np.maximum(d, 1e-300), 0.0)
            ds = np.bincount(S.seat, f[S.st] * gam[S.lab], minlength=Q)
            ns = np.where(ds > 0, WS / np.maximum(ds, 1e-300), 1.0)
            ns /= math.exp(np.log(ns).mean())
            delta = max(delta, float(np.abs(np.log(ns / seat)).max()))
            seat = ns
        if delta < tol:
            break
    lg, ls = np.log10(gam), np.log10(seat)
    return lg - lg.mean(), ls - ls.mean()


def fit(rankings, labels, prior: float = 0.5, iters: int = 1000, seat_effects: bool = False,
        max_stages: int | None = None) -> dict:
    """Point fit: {"elo": {label: rating, mean 1500}, "seat_elo": per-seat offsets (0 without seat_effects)}.
    ``max_stages=1`` uses only the winner (a multinomial-logit "who won" model)."""
    labels = list(labels)
    lg, ls = fit_stages(Stages(rankings, labels, max_stages), None, prior, iters, seat_effects)
    return {"elo": {x: 1500 + 400 * float(v) for x, v in zip(labels, lg)},
            "seat_elo": [round(400 * float(v), 1) for v in ls]}


def ratings(rankings, labels, bootstrap: int = 200, seed: int = 0, prior: float = 0.5) -> dict:
    """PL ratings with a bootstrap over games (95% percentile CI), mean 1500.
    Same shape as catan.eval.ratings.ratings: {label: {"elo", "ci95"}}."""
    labels = sorted(set(labels))
    if len(labels) < 2:
        return {x: {"elo": 1500.0, "ci95": [1500.0, 1500.0]} for x in labels}
    S = Stages(rankings, labels)
    lg, _ = fit_stages(S, None, prior)
    rng = np.random.default_rng(seed)
    n = len(rankings)
    B = []
    for _ in range(bootstrap):
        mult = np.bincount(rng.integers(0, n, n), minlength=n).astype(float)
        # warm start from the point fit: 60 MM steps land within 0.05 Elo of convergence
        B.append(1500 + 400 * fit_stages(S, mult[S.game], prior, 60, tol=2e-5, init=lg)[0])
    B = np.array(B)
    out = {}
    for j, x in enumerate(labels):
        pt = 1500 + 400 * float(lg[j])
        ci = [float(np.percentile(B[:, j], 2.5)), float(np.percentile(B[:, j], 97.5))] if len(B) else [pt, pt]
        out[x] = {"elo": round(pt, 1), "ci95": [round(ci[0], 1), round(ci[1], 1)]}
    return out


def win_probs(game_labels, elo: dict, seat_elo=None) -> np.ndarray:
    """P(seat q wins) = g_q / sum g under PL (also the natural 4-player read of BT ratings)."""
    v = np.array([elo.get(a, 1500.0) + (seat_elo[q] if seat_elo else 0.0) for q, a in enumerate(game_labels)])
    g = 10 ** ((v - v.max()) / 400)
    return g / g.sum()
