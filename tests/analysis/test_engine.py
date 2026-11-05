import pytest

from catan.analysis import analyze, analyze_game, continue_position, position_from_game, validate_position
from catan.analysis.engine import verdict, parse_policy
from catan.engine import Action, Phase
from catan.eval.stats import wilson

from .helpers import CKPT, play_until


@pytest.fixture(scope="module")
def mid():
    g = play_until(3, lambda g: g.turn >= 30 and g.phase == Phase.MAIN and g.actor == 0)
    return g, position_from_game(g, 0)


def _strip(res):
    res = dict(res)
    res.pop("timing_ms")
    return res


def test_candidates_legal_and_fields(mid):
    g, spec = mid
    legal = {a.to_json().__repr__() for a in g.legal_actions()}
    res = analyze(spec, "intermediate", depth="standard", seed=1)
    assert res["legal_count"] == len(g.legal_actions())
    assert res["actor"] == 0 and res["rollout_policy"] == "basic"
    for c in res["candidates"]:
        assert repr(c["action"]) in legal
        assert c["policy_prob"] is None and c["value_estimate"] is None  # not an RL policy
        assert isinstance(c["search_score"], float)
        ro = c["rollout"]
        assert ro["n"] == 32 and 0 <= ro["ci95"][0] <= ro["win_rate"] <= ro["ci95"][1] <= 1
        assert c["verdict"] in ("best", "good", "inaccuracy", "mistake", "blunder", "unrated")
    assert res["candidates"][0]["verdict"] == "best"
    assert [c["rank"] for c in res["candidates"]] == list(range(1, len(res["candidates"]) + 1))
    assert res["position_eval"]["value_estimate"] is None and res["position_eval"]["rollout"]["n"] == 32


def test_quick_has_no_rollouts(mid):
    res = analyze(mid[1], "intermediate", depth="quick")
    assert res["rollout_policy"] is None and res["position_eval"]["rollout"] is None
    assert all(c["rollout"] is None and c["verdict"] == "unrated" for c in res["candidates"])


def test_random_policy_uses_pruning_and_reports_no_scores(mid):
    res = analyze(mid[1], "random", depth="quick")
    assert res["candidate_source"] == "intermediate_pruning"
    assert all(c["search_score"] is None and c["policy_prob"] is None for c in res["candidates"])


def test_determinism_given_seed(mid):
    a = analyze(mid[1], "intermediate", depth="standard", seed=4)
    assert _strip(a) == _strip(analyze(mid[1], "intermediate", depth="standard", seed=4))


def test_viewer_must_be_actor(mid):
    g, _ = mid
    spec = position_from_game(g, 2)
    with pytest.raises(ValueError, match="viewer"):
        analyze(spec, depth="quick")


def test_wilson_and_verdicts():
    lo, hi = wilson(5, 10)
    assert abs(lo - 0.2366) < 1e-3 and abs(hi - 0.7634) < 1e-3
    assert wilson(0, 0) == (0.0, 1.0)
    lo, hi = wilson(0, 32)
    assert lo == 0 and abs(hi - 0.1072) < 1e-3

    def ro(k, n):
        lo, hi = wilson(k, n)
        return {"win_rate": k / n, "ci95": [lo, hi], "n": n}
    best = ro(180, 200)
    assert verdict(ro(176, 200), best) == "good"            # gap < 5 points
    assert verdict(ro(14, 20), ro(18, 20)) == "unrated"     # gap 20 points but CIs overlap
    assert verdict(ro(160, 200), best) == "inaccuracy"
    assert verdict(ro(150, 200), best) == "mistake"
    assert verdict(ro(100, 200), best) == "blunder"
    assert verdict(None, best) == "unrated"


def test_parse_policy():
    assert parse_policy("rl:checkpoint=runs/a.pt,deterministic=true") == \
        ("rl", {"checkpoint": "runs/a.pt", "deterministic": True})
    assert parse_policy("search:rollouts=4") == ("search", {"rollouts": 4})
    with pytest.raises(ValueError):
        parse_policy("search:rollouts")


def winning_spec():
    """Seat 0 to move in MAIN, holding exactly a city's cost, one VP short."""
    g = play_until(7, lambda g: g.turn >= 12 and g.phase == Phase.MAIN and g.actor == 0
                   and sum(g.players[0].dev) == 0 and g.players[0].settlements
                   and g.bank[3] + g.players[0].hand[3] >= 2 and g.bank[4] + g.players[0].hand[4] >= 3)
    spec = position_from_game(g, 0)
    hand, bank = spec["players"][0]["hand"], spec["bank"]
    want = [0, 0, 0, 2, 3]
    for r in range(5):
        bank[r] += hand[r] - want[r]
        hand[r] = want[r]
    vp = len(g.players[0].settlements) + 2 * len(g.players[0].cities) + 2 * (g.longest_road_owner == 0) \
        + 2 * (g.largest_army_owner == 0)
    spec["config"]["vp_to_win"] = vp + 1
    assert validate_position(spec) == [], validate_position(spec)
    return spec


def test_obvious_winning_move_ranked_best():
    spec = winning_spec()
    res = analyze(spec, "basic", depth="standard", seed=0, top_k=6)
    best = res["candidates"][0]
    assert best["action"]["type"] == "BUILD_CITY"
    assert best["rollout"]["win_rate"] == 1.0 and best["verdict"] == "best"
    others = [c for c in res["candidates"] if c["action"]["type"] != "BUILD_CITY"]
    assert others and all(c["rollout"]["win_rate"] < 1.0 for c in others)
    # the winning label is human readable
    assert best["label"].startswith("Upgrade to city at vertex")


def test_include_forces_played_move(mid):
    g, spec = mid
    end = Action(0, "END_TURN")
    res = analyze(spec, "intermediate", depth="quick", top_k=1, include=[end])
    assert any(c["action"] == end.to_json() for c in res["candidates"])
    with pytest.raises(ValueError, match="not legal"):
        analyze(spec, "intermediate", depth="quick", include=[Action(0, "ROLL")])


@pytest.mark.skipif(CKPT is None, reason="no RL checkpoint under runs/")
def test_rl_fields(mid):
    res = analyze(mid[1], f"rl:checkpoint={CKPT}", depth="quick")
    probs = [c["policy_prob"] for c in res["candidates"]]
    assert probs == sorted(probs, reverse=True) and 0 < sum(probs) <= 1.0001
    assert all(isinstance(c["value_estimate"], float) and c["search_score"] is None for c in res["candidates"])
    assert isinstance(res["position_eval"]["value_estimate"], float)


def test_analyze_game_timeline():
    ag_seed = 9
    g = play_until(ag_seed, lambda g: False, agent="basic", max_turns=24)
    tl = analyze_game(g, 1, "intermediate", depth="quick")
    s = tl["summary"]
    assert s["decisions"] == len(tl["timeline"]) > 0 and s["skipped_forced"] > 0
    n_viewer = sum(1 for a in g.action_log if a.player == 1)
    assert s["decisions"] + s["skipped_forced"] <= n_viewer
    for t in tl["timeline"]:
        assert t["actor"] == 1 and g.action_log[t["index"]].to_json() == t["played"]
        assert t["legal_count"] > 1 and t["verdict"] in ("best", "unrated")
        if t["played"] == t["best"]:
            assert t["verdict"] == "best"
    # replay-dict source gives the same timeline
    rep = {"seed": g.seed, "config": g.config.to_dict(), "actions": [a.to_json() for a in g.action_log]}
    tl2 = analyze_game(rep, 1, "intermediate", depth="quick")
    assert tl2["timeline"] == tl["timeline"]


def test_continue_position(mid):
    _, spec = mid
    r = continue_position(spec, {"player": 0, "type": "END_TURN", "value": None}, "basic", seed=1)
    assert r["actions"][0]["label"] == "End turn"
    assert r["position"]["turn"] == spec["turn"] + 1 and r["position"]["viewer"] == 0
    assert all(e.get("visible_to") is None or 0 in e["visible_to"] for e in r["events"])
    r2 = continue_position(spec, {"player": 0, "type": "END_TURN", "value": None}, "basic", until="game_end", seed=1)
    assert r2["done"]
    r3 = continue_position(spec, {"player": 0, "type": "END_TURN", "value": None}, "basic", steps=3, seed=1)
    assert len(r3["actions"]) == 4
    with pytest.raises(ValueError, match="illegal"):
        continue_position(spec, {"player": 0, "type": "ROLL", "value": None})


def test_parallel_rollouts_match_serial(mid):
    kw = dict(policy="intermediate", depth="deep", seed=5, top_k=3, samples=2, rollouts=3)
    a = analyze(mid[1], workers=1, **kw)
    b = analyze(mid[1], workers=2, **kw)
    assert _strip(a) == _strip(b)


VT = "jointb_gated:sims=8"


def test_search_policy_reports_its_own_choice_and_search_values(monkeypatch):
    """The champion (jointb_gated) is scored by its own search, not by intermediate's heuristic,
    and it decides on the viewer's information-set view (no peeking)."""
    pytest.importorskip("torch")
    import os
    from catan.analysis import engine
    from catan.engine.infoset import InfoSetGame
    if not os.path.isfile("models/value_best.pt"):
        pytest.skip("models/value_best.pt not available")
    orig = engine.get_agent(VT, engine._QUICK_AGENTS)
    games = []
    monkeypatch.setattr(orig, "act_with_info", lambda g, legal, _f=orig.act_with_info, **kw: (games.append(g), _f(g, legal))[1])
    g = play_until(3, lambda g: g.phase == Phase.SETUP_ROAD and g.actor == 0)  # a road choice, as in the bug report
    res = analyze(position_from_game(g, 0), VT, depth="quick")
    assert games and all(isinstance(g, InfoSetGame) for g in games)
    assert res["policy_kind"] == "search" and res["candidate_source"] == "search_root"
    assert res["agent_choice"] is not None and res["search_value_kind"].startswith("value-net win probability")
    picked = [c for c in res["candidates"] if c["agent_choice"]]
    assert len(picked) == 1 and picked[0]["action"] == res["agent_choice"] and picked[0]["rank"] == 1
    vals = [c["search_value"] for c in res["candidates"] if c["search_value"] is not None]
    assert vals and all(0.0 <= v <= 1.0 for v in vals)
    assert all(c["search_value_calibrated"] is False for c in res["candidates"] if c["search_value"] is not None)
    assert sum(c["search_visits"] or 0 for c in res["candidates"]) == res["agent_info"]["sims"]
