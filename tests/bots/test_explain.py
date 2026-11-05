"""Decision explanations: terms add up, decisions are unchanged, payload is JSON-safe and small."""
import json
import random

import pytest

from catan.agents.registry import make_agent
from catan.bots import common as C
from catan.bots.explain import evaluate_terms, resource_value_parts
from catan.bots.intermediate import W, evaluate
from catan.engine import Action, Game, GameConfig

FAST = {"basic": {}, "intermediate": {}, "search": {"rollouts": 3, "candidates": 4}}
MAX_BYTES = 16000  # MAIN decisions carry provenance sentences for the chosen action and runner-up


def lineup(name, seed):
    agents = [make_agent(name, seed=seed * 10 + i, **FAST[name]) for i in range(4)]
    for i, a in enumerate(agents):
        a.reset(i, seed * 10 + i)
    return agents


def play(name, seed, explain, max_turns=80, on_info=None):
    agents = lineup(name, seed)
    g = Game(config=GameConfig(max_turns=max_turns), seed=seed)
    while not g.is_over:
        legal = g.legal_actions()
        if explain:
            a, info = agents[g.actor].act_with_info(g, legal)
            if on_info:
                on_info(g, info)
        else:
            a = agents[g.actor].act(g, legal)
        g.step(a)
    return g.action_log


@pytest.mark.parametrize("name", ["basic", "intermediate", "search"])
def test_explanations_do_not_change_decisions(name):
    for seed in range(2 if name == "search" else 8):  # search: ~10 s per seed pair
        assert play(name, seed, explain=False) == play(name, seed, explain=True), seed


@pytest.mark.parametrize("name", ["basic", "intermediate", "search"])
def test_terms_sum_to_score_and_payload_bounded(name):
    seen, declines = set(), []

    def check(g, info):
        assert "explain_error" not in info, info["explain_error"]
        raw = json.dumps(info, allow_nan=False)
        assert len(raw) < MAX_BYTES, (g.phase, len(raw))
        e = json.loads(raw)["explain"]
        seen.add(e["phase"])
        assert e["summary"] and e["summary_source"] == "generated from heuristic terms"
        assert "()" not in e["summary"], e["summary"]  # DS2-10: zero-score defaults list no empty terms
        if e["phase"] == "TRADE_RESPONSE" and e["candidates"][0]["action"]["type"] == "REJECT_TRADE":
            declines.append(e["summary"])
            # DS31-03: trade_delta's -9 can't-pay sentinel read as a hand score ("Hand gain -9.00")
            assert "-9.00" not in e["summary"] and "\u22129" not in e.get("context", {}).get("text", ""), e["summary"]
        assert e["candidates"][0]["action"] == info["top"][0]["action"] or e["candidates"][0]["score"] == info["top"][0]["score"]
        for c in e["candidates"]:
            assert "unattributed" not in c["terms"], (g.phase, c)
            total = c.get("heuristic", c["score"])
            assert abs(sum(c["terms"].values()) - total) < 2e-3, (g.phase, c)
            if "deltas" in c:
                for k, d in c["deltas"].items():
                    assert abs(c["terms"].get(k, 0) - e["baseline_terms"].get(k, 0) - d) < 2e-4
        if "search" in e and e["search"]["ran"]:
            assert all(c["rollout"]["n"] == FAST["search"]["rollouts"] for c in e["candidates"])

    for seed in range(2):
        play(name, seed, explain=True, max_turns=120, on_info=check)
    assert {"SETUP_SETTLEMENT", "SETUP_ROAD", "MAIN", "MOVE_ROBBER"} <= seen
    if name != "basic":  # basic never offers, so nobody declines
        assert declines and all(s.startswith("Decline trade: score 0.00.") for s in declines), declines[:2]


def test_evaluate_terms_match_evaluate_exactly():
    rng = random.Random(0)
    agents = lineup("intermediate", 5)
    g = Game(config=GameConfig(max_turns=150), seed=5)
    n = 0
    while not g.is_over:
        bi = C.board_info(g.layout)
        for p in range(g.n):
            assert abs(sum(evaluate_terms(g, p, bi, W).values()) - evaluate(g, p, bi, W)) < 1e-9
            rv = C.resource_values(g, p, bi)
            for part, v in zip(resource_value_parts(g, p, bi), rv):
                assert abs(part["scarcity"] * part["phase"] * part["own_factor"] - v) < 1e-3
            n += 1
        g.step(agents[g.actor].act(g, g.legal_actions()) if rng.random() < 0.9 else rng.choice(g.legal_actions()))
    assert n > 200


def test_search_reports_rollouts_or_skip_reason():
    a = make_agent("search", seed=1, rollouts=3, candidates=4)
    a.reset(0, 1)
    g = Game(config=GameConfig(max_turns=120), seed=2)
    others = lineup("basic", 2)
    ran = skipped = 0
    while not g.is_over and not (ran and skipped):
        legal = g.legal_actions()
        if g.actor == 0:
            act, info = a.act_with_info(g, legal)
            s = info["explain"].get("search")
            if s:
                assert "leaf" in s and "policy" in s
                if s["ran"]:
                    ran += 1
                    r = info["explain"]["candidates"][0]["rollout"]
                    assert r["mean"] == info["explain"]["candidates"][0]["score"] and r["diff_vs_chosen"] == 0
                else:
                    skipped += 1
                    assert s["skip_reason"]
        else:
            act = others[g.actor].act(g, legal)
        g.step(act)
    assert ran and skipped


# ------------------------------------------------------------------ provenance
def recompute(term, inp, rv):
    """Term value from the provenance inputs alone (independent of explain.py's code)."""
    if term.startswith("spot"):
        return 0.0 if inp["vertex"] is None else inp["weight"] * sum(p * v for p, v in zip(inp["pips"], rv))
    if term == "hand":
        scores = sorted((gl["weight"] / (1 + 1.3 * sum(m * v for m, v in zip(gl["missing"], rv))) for gl in inp["goals"]),
                        reverse=True)
        h = inp["hand"]
        s = (scores[0] if scores else 0) + (0.4 * scores[1] if len(scores) > 1 else 0) + 0.03 * sum(n * v for n, v in zip(h, rv))
        return inp["weight"] * (s - 0.06 * max(0, sum(h) - 7))
    if term == "production":
        prod = [0.0] * 5
        for b in inp["buildings"]:
            for r, pips, robbed in b["hexes"]:
                prod[r] += 0 if robbed else pips * b["level"]
        return inp["weight"] * sum(p * v for p, v in zip(prod, rv))
    if term == "diversity":
        return inp["weight"] * sum(1 for x in inp["prod"] if x > 0)
    if term == "vp":
        return inp["settlements"] + 2 * inp["cities"] + 2 * inp["longest_road"] + 2 * inp["largest_army"] + inp["vp_cards"]
    if term == "dev_held":
        return inp["weight"] * inp["count"]
    if term == "knights":
        return inp["weight"] * min(inp["count"], 4)
    if term == "road_len":
        return 0.0 if inp["holds_lr"] else inp["weight"] * min(inp["length"], 6)
    return None


def heuristic_pick(g):
    """Independent re-score: a fresh intermediate agent's choice on the (modified) copy."""
    return make_agent("intermediate").act(g, g.legal_actions())


@pytest.mark.parametrize("name", ["intermediate", "search"])
def test_provenance_reproduces_terms_and_counterfactuals_hold(name):
    stats = {"terms": 0, "cf": 0, "flips": 0}

    def check(g, info):
        e = info["explain"]
        if e.get("kind") != "value_model" or e["phase"] != "MAIN":
            return
        rv0 = [p["value"] for p in e["resource_values"]]
        sets = [(e["baseline_why"], e["baseline_terms"], rv0)]
        sets += [(c["why"], c["terms"], c["rv_after"]) for c in e["candidates"] if "why" in c]
        for why, terms, rv in sets:
            for k, p in why.items():
                v = recompute(k, p["inputs"], rv)
                if v is not None:
                    assert abs(v - terms.get(k, 0.0)) < 3e-3, (k, v, terms.get(k), p)
                    assert p["text"]
                    stats["terms"] += 1
        for k in e["candidates"][0]["deltas"]:  # every changed term of the chosen action is explained
            assert k in e["candidates"][0]["why"], k
        cf = e["counterfactuals"]
        me = g.actor
        assert heuristic_pick(g) == Action.from_json(cf["baseline_choice"])
        if name == "intermediate":
            assert cf["baseline_choice"] == e["candidates"][0]["action"]
        for it in cf["items"]:
            if it["kind"] == "add_card":
                c = g.clone()
                c.bank[it["res"]] -= 1
                c.players[me].hand[it["res"]] += 1
                assert heuristic_pick(c).to_json() == it["choice"] != cf["baseline_choice"]
                stats["flips"] += 1
            elif it["kind"] == "add_card_none":
                for r in it["checked"]:
                    c = g.clone()
                    c.bank[r] -= 1
                    c.players[me].hand[r] += 1
                    assert heuristic_pick(c).to_json() == cf["baseline_choice"]
            else:
                c = g.clone()
                c._place_settlement(C.leader(g, me), it["vertex"])
                assert heuristic_pick(c).to_json() == it["choice"]
                assert it["changes"] == (it["choice"] != cf["baseline_choice"])
            stats["cf"] += 1

    play(name, 1, explain=True, max_turns=50 if name == "search" else 80, on_info=check)
    assert stats["terms"] > 100 and stats["cf"] > 20 and stats["flips"] > 0, stats
