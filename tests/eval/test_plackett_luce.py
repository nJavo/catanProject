"""Plackett-Luce ratings (catan.eval.plackett_luce)."""
import random

import numpy as np

from catan.eval import plackett_luce as pl


def simulate(true, seat, n, rng, size=4):
    labs = list(true)
    games = []
    for _ in range(n):
        t = rng.choice(labs, size, replace=False)
        g = np.array([10 ** ((true[a] + seat[q]) / 400) for q, a in enumerate(t)])
        rem, sc = list(range(size)), [0] * size
        r = size
        while rem:
            k = rem.pop(rng.choice(len(rem), p=g[rem] / g[rem].sum()))
            sc[k] = r
            r -= 1
        games.append([(a, s) for a, s in zip(t, sc)])
    return games


def test_pl_recovers_known_ratings_and_seat_effects():
    rng = np.random.default_rng(1)
    true = {f"a{i}": r for i, r in enumerate([1250, 1400, 1450, 1500, 1550, 1600, 1750, 1500])}
    seat = [60, 20, -30, -50]
    games = simulate(true, seat, 3000, rng)
    m = np.mean(list(true.values()))
    f = pl.fit(games, list(true), prior=0, seat_effects=True)
    for a, r in true.items():
        assert abs(f["elo"][a] - (r - m + 1500)) < 25, (a, f["elo"][a], r)
    assert np.allclose(f["seat_elo"], np.array(seat) - np.mean(seat), atol=15)
    # CIs cover the truth for most labels
    ci = pl.ratings(games, list(true), bootstrap=60, prior=0)
    cover = sum(ci[a]["ci95"][0] - 5 <= r - m + 1500 <= ci[a]["ci95"][1] + 5 for a, r in true.items())
    assert cover >= 7


def test_pl_ties_duplicates_and_two_player_bt():
    from catan.eval.ratings import bradley_terry, pairwise_outcomes
    # 1v3 tables with ties among losers: must fit, and the 3 copies share one strength
    games = [[("x", 110), ("b", 5), ("b", 5), ("b", 3)], [("b", 110), ("x", 7), ("b", 4), ("b", 4)],
             [("x", 110), ("b", 6), ("b", 6), ("b", 6)]] * 20
    f = pl.fit(games, ["x", "b"], prior=0.5)
    assert f["elo"]["x"] > f["elo"]["b"]
    s = pl.Stages([[("x", 110), ("b", 6), ("b", 6), ("b", 6)]], ["x", "b"])
    assert s.n_stages == 1  # the 3-way tie at the bottom carries no order information
    # two-player games: PL is BT
    rng = random.Random(0)
    g2 = [[("p", 1), ("q", 0)] if rng.random() < 0.7 else [("p", 0), ("q", 1)] for _ in range(400)]
    a = pl.fit(g2, ["p", "q"], prior=0.5)["elo"]
    b = bradley_terry(pairwise_outcomes(g2), ["p", "q"], prior=0.5)
    assert abs((a["p"] - a["q"]) - (b["p"] - b["q"])) < 1.0


def test_winner_only_model_uses_one_stage():
    s = pl.Stages([[("a", 110), ("b", 5), ("c", 4), ("d", 3)]], list("abcd"), max_stages=1)
    assert s.n_stages == 1 and s.ch.sum() == 1


def _toy_game(rng, skill, V):
    """Race: 2 players add d6 rolls for 10 rounds, player 0 gets +skill per round; z = player 0 wins.
    Returns (z, luck of player 0) with luck from baseline V(lead, rounds_left)."""
    lead, lk = 0.0, 0.0
    for t in range(10, 0, -1):
        for p in (0, 1):
            before = lead
            r = rng.integers(1, 7)
            lead = lead + (r + skill if p == 0 else -r)
            exp = np.mean([V(before + (k + skill if p == 0 else -k), t - (p == 1)) for k in range(1, 7)])
            lk += V(lead, t - (p == 1)) - exp
    return float(lead > 0), lk
