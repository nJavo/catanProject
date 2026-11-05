import json
import math
import os
import random
from collections import Counter

import pytest

from catan.agents.base import Agent
from catan.agents.registry import register
from catan.eval.ratings import bradley_terry, pairwise_outcomes, ratings
from catan.eval.stats import binom_two_sided_p, mean_ci, wilson
from catan.eval.tournament import lineups, parse_spec, run_tournament


def test_wilson_known_values():
    lo, hi = wilson(5, 10)
    assert math.isclose(lo, 0.2366, abs_tol=1e-3) and math.isclose(hi, 0.7634, abs_tol=1e-3)
    lo, hi = wilson(0, 10)
    assert lo == 0 and math.isclose(hi, 0.2775, abs_tol=1e-3)
    assert wilson(0, 0) == (0.0, 1.0)


def test_wilson_coverage():
    rng = random.Random(0)
    p, n, hits = 0.3, 200, 0
    for _ in range(500):
        k = sum(rng.random() < p for _ in range(n))
        lo, hi = wilson(k, n)
        hits += lo <= p <= hi
    assert 0.92 <= hits / 500 <= 0.98


def test_mean_ci_and_binom():
    m = mean_ci([1, 2, 3, 4])
    assert m["mean"] == 2.5 and m["n"] == 4
    assert math.isclose(binom_two_sided_p(5, 10), 1.0)
    assert binom_two_sided_p(10, 10) < 0.002


def test_parse_spec():
    assert parse_spec("basic") == ("basic", {})
    assert parse_spec("rl:checkpoint=runs/a.pt,temp=0.5,greedy=true,n=3") == (
        "rl", {"checkpoint": "runs/a.pt", "temp": 0.5, "greedy": True, "n": 3})
    with pytest.raises(ValueError):
        parse_spec("rl:checkpoint")


def test_lineups_balanced():
    cyc = lineups(["a", "b", "c", "d"], 4, "cyclic")
    assert len(cyc) == 4
    for seat in range(4):
        assert sorted(x[seat] for x in cyc) == ["a", "b", "c", "d"]
    assert len(lineups(["a", "b", "c", "d"], 4, "all")) == 24
    two = lineups(["a", "b"], 4, "cyclic")
    assert all(Counter(x) == Counter({"a": 2, "b": 2}) for x in two)
    one_v3 = lineups(["x", "r", "r", "r"], 4, "cyclic")
    assert sorted(x.index("x") for x in one_v3) == [0, 1, 2, 3]


def test_bradley_terry_orders_strength():
    rng = random.Random(1)
    strength = {"a": 4.0, "b": 2.0, "c": 1.0}
    games = []
    for _ in range(300):
        x, y = rng.sample(list(strength), 2)
        px = strength[x] / (strength[x] + strength[y])
        games.append([(x, 1), (y, 0)] if rng.random() < px else [(x, 0), (y, 1)])
    r = bradley_terry(pairwise_outcomes(games), strength)
    assert r["a"] > r["b"] > r["c"]
    # a is 2x b: ~120 Elo
    assert 60 < r["a"] - r["b"] < 200
    rb = ratings(games, list(strength), bootstrap=30)
    assert rb["a"]["ci95"][0] <= rb["a"]["elo"] <= rb["a"]["ci95"][1]


class IllegalAgent(Agent):
    name = "illegal"

    def act(self, game, legal):
        return legal[0]._replace(type="END_TURN") if game.phase != "MAIN" else legal[0]


register("illegal_test", lambda seed=None, **_: IllegalAgent())


def test_tournament_json_and_determinism(tmp_path):
    kw = dict(games=8, seed=3, num_players=4, max_turns=300, out_dir=str(tmp_path))
    r1 = run_tournament(["random", "basic", "illegal_test", "basic:seed=5"], workers=1, tid="t1", **kw)
    r2 = run_tournament(["random", "basic", "illegal_test", "basic:seed=5"], workers=2, tid="t2",
                        save_replays=1, **kw)
    assert r1["game_summaries"] == r2["game_summaries"]
    d = json.load(open(tmp_path / "tournaments" / "t1.json"))
    for k in ("id", "agents", "games", "seeds", "per_agent", "pairwise", "created", "ratings"):
        assert k in d
    pa = d["per_agent"]
    assert set(pa) == {"random", "basic", "illegal_test", "basic:seed=5"}
    assert pa["illegal_test"]["illegal_actions"] > 0 and pa["basic"]["illegal_actions"] == 0
    assert d["illegal_log"]
    for lab in pa:
        assert pa[lab]["games"] == 8 and 0 <= pa[lab]["ci95"][0] <= pa[lab]["win_rate"] <= pa[lab]["ci95"][1]
        assert sum(s["games"] for s in pa[lab]["seats"].values()) == 8
    assert sum(pa[x]["wins"] for x in pa) + sum(g["winner"] < 0 for g in d["game_summaries"]) == 8
    assert os.listdir(tmp_path / "replays")
