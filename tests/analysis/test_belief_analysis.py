"""Belief-weighted hand sampling in the analysis engine and API."""
import json

import pytest
from fastapi.testclient import TestClient

from catan.analysis import determinize, position_from_game
from catan.analysis.engine import analyze, analyze_game
from catan.bots.belief import HandPosterior
from catan.engine import Phase
from catan.server.app import create_app

from .helpers import play_until


@pytest.fixture(scope="module")
def mid():
    g = play_until(3, lambda g: g.turn >= 30 and g.phase == Phase.MAIN and g.actor == 0)
    post, _ = HandPosterior.from_history(g.seed, g.config, list(g.action_log), 0)
    return g, position_from_game(g, 0), post


def test_belief_determinize_draws_from_posterior(mid):
    g, spec, post = mid
    assert not post.lost
    for d in determinize(spec, 20, 1, belief=post):
        hands = tuple(tuple(d.players[q].hand) for q in post.opps)
        assert hands in post.states
        assert d.players[0].hand == g.players[0].hand and d.bank == g.bank
    # the default path is unchanged
    a, b = determinize(spec, 3, 5), determinize(spec, 3, 5, belief=None)
    assert [[p.hand for p in x.players] for x in a] == [[p.hand for p in x.players] for x in b]


def test_analyze_reports_belief(mid):
    g, spec, post = mid
    res = analyze(spec, depth="quick", belief=post)
    assert res["determinization"]["hand_sampling"] == "belief"
    assert "posterior" in res["determinization"]["assumptions"][0]
    b = res["belief"]
    assert b["label"] == "estimate from public events"
    for q in post.opps:
        o = b["opponents"][str(q)]
        assert abs(sum(o["expected"]) - sum(g.players[q].hand)) < 1e-3
        assert all(abs(sum(d.values()) - 1) < 1e-3 for d in o["dist"])
    assert analyze(spec, depth="quick")["belief"] is None


@pytest.fixture
def client(tmp_path, mid):
    g = mid[0]
    (tmp_path / "results" / "replays").mkdir(parents=True)
    (tmp_path / "results" / "replays" / "r1.json").write_text(json.dumps(
        {"seed": g.seed, "config": g.config.to_dict(), "actions": [a.to_json() for a in g.action_log]}))
    return TestClient(create_app(data_dir=tmp_path, static_dir=tmp_path / "nodist"))


def test_api_belief(client, mid):
    g, spec, post = mid
    body = {"source": {"kind": "replay", "ref": "r1.json"}, "viewer": 0, "depth": "quick", "belief": True}
    r = client.post("/api/analysis/evaluate", json=body)
    assert r.status_code == 200, r.text
    b = r.json()["belief"]
    assert b["states"] == len(post.states) and set(b["opponents"]) == {"1", "2", "3"}
    r = client.post("/api/analysis/evaluate", json={"position": spec, "depth": "quick", "belief": True})
    assert r.status_code == 400 and "source" in r.json()["detail"]
    r = client.post("/api/analysis/evaluate", json=dict(body, belief="yes"))
    assert r.status_code == 400


def test_analyze_game_with_belief(mid):
    g = mid[0]
    rep = {"seed": g.seed, "config": g.config.to_dict(), "actions": [a.to_json() for a in g.action_log[:120]]}
    res = analyze_game(rep, 0, depth="quick", belief=True)
    assert res["timeline"]
