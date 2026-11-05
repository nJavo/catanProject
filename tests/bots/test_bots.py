import copy
import random

import pytest

from catan.agents.registry import available_agents, make_agent
from catan.engine import Game, GameConfig
from catan.engine.actions import Phase
from catan.bots import common as C
from catan.bots.intermediate import IntermediateAgent

FAST = {"basic": {}, "intermediate": {}, "search": {"rollouts": 2, "candidates": 3}}


def play(lineup, seed, max_turns=500):
    agents = [make_agent(n, seed=seed * 10 + i, **FAST[n]) for i, n in enumerate(lineup)]
    for i, a in enumerate(agents):
        a.reset(i, seed * 10 + i)
    g = Game(config=GameConfig(max_turns=max_turns), seed=seed)
    while not g.is_over:
        legal = g.legal_actions()
        a = agents[g.actor].act(g, legal)
        assert a in legal, (a, g.phase)
        g.step(a)
    return g


def test_registered():
    assert {"random", "basic", "intermediate", "search", "intermediate_trader", "search_trader"} <= set(available_agents())


@pytest.mark.parametrize("name", ["basic", "intermediate"])
def test_selfplay_finishes_legal(name):
    for seed in range(6):
        g = play([name] * 4, seed)
        assert g.winner >= 0 and not g.truncated


def test_mixed_games_finish_legal():
    for seed in range(4):
        g = play(["basic", "intermediate", "search", "basic"][seed % 4:] + ["basic", "intermediate", "search", "basic"][:seed % 4], seed)
        assert g.is_over


@pytest.mark.parametrize("name", ["basic", "intermediate", "search"])
def test_deterministic_given_seed(name):
    a = play([name, "basic", "intermediate", "basic"], 3, max_turns=60)
    b = play([name, "basic", "intermediate", "basic"], 3, max_turns=60)
    assert a.action_log == b.action_log


def test_act_with_info_reports_scores():
    g = Game(seed=5)
    for name in ("basic", "intermediate", "search"):
        a, info = make_agent(name, seed=1, **FAST[name]).act_with_info(g, g.legal_actions())
        assert info["score_kind"] == "heuristic_score"
        assert info["top"] and info["top"][0]["action"]["type"] == "BUILD_SETTLEMENT"


def resample_hidden(g, me, rng):
    """Independent resampler: keeps everything public (hand sizes, per-resource
    totals via the bank, dev counts, unseen dev composition) and changes the rest."""
    c = g.clone()
    opps = [q for q in range(g.n) if q != me]
    pool = [r for q in opps for r in range(5) for _ in range(g.players[q].hand[r])]
    rng.shuffle(pool)
    i = 0
    for q in opps:
        k = sum(g.players[q].hand)
        h = [0] * 5
        for r in pool[i:i + k]:
            h[r] += 1
        c.players[q].hand = h
        i += k
    dpool = list(g.dev_deck) + [t for q in opps for t in range(5) for _ in range(g.players[q].dev[t])]
    rng.shuffle(dpool)
    i = 0
    for q in opps:
        k = sum(g.players[q].dev)
        nk = sum(g.players[q].dev_new)
        d, dn = [0] * 5, [0] * 5
        for j, t in enumerate(dpool[i:i + k]):
            d[t] += 1
            dn[t] += j < nk
        c.players[q].dev, c.players[q].dev_new = d, dn
        i += k
    c.dev_deck = dpool[i:]
    c.rng = random.Random(rng.random())
    return c


def invariance_mismatches(name, every, make=None):
    """(checked decisions, mismatches) when only hidden info is resampled."""
    rng = random.Random(0)
    checked = bad = 0
    for seed in (11, 12):
        lineup = ["basic", "intermediate", name, "random"]
        agents = [make_agent(n, seed=seed + i, **FAST.get(n, {})) for i, n in enumerate(lineup)]
        if make:
            agents[2] = make()
        for i, a in enumerate(agents):
            a.reset(i, seed + i)
        g = Game(config=GameConfig(max_turns=200), seed=seed)
        step = 0
        while not g.is_over:
            p = g.actor
            legal = g.legal_actions()
            if p == 2 and g.phase not in (Phase.SETUP_SETTLEMENT, Phase.SETUP_ROAD) and step % every == 0:
                want = copy.deepcopy(agents[p]).act(g, legal)
                for _ in range(3):
                    h = resample_hidden(g, p, rng)
                    bad += copy.deepcopy(agents[p]).act(h, h.legal_actions()) != want
                checked += 1
            step += 1
            evs = g.step(agents[p].act(g, legal))
            for i, a in enumerate(agents):
                if hasattr(a, "observe"):
                    a.observe(g, [e for e in evs if "visible_to" not in e or i in e["visible_to"]])
    return checked, bad


@pytest.mark.parametrize("name,every", [("basic", 1), ("intermediate", 2), ("search", 8)])
def test_hidden_info_invariance(name, every):
    """Bots must choose the same action when only hidden info changes."""
    checked, bad = invariance_mismatches(name, every)
    assert checked > 20 and bad == 0


class Peeker(IntermediateAgent):
    """Negative control: steals from whoever holds the most ore (hidden)."""

    def steal(self, g, me, legal):
        return [(g.players[a.value].hand[4], a) for a in legal]

    def discard(self, g, me, legal):
        nxt = (me + 1) % g.n
        return [(g.players[nxt].hand[a.value], a) for a in legal]


@pytest.mark.parametrize("name,every", [("intermediate", 1), ("search", 6)])
def test_hidden_info_invariance_trader(name, every):
    kw = dict(FAST[name], trader=True)
    checked, bad = invariance_mismatches(name, every, make=lambda: make_agent(name, seed=0, **kw))
    assert checked > 20 and bad == 0


def test_invariance_check_catches_peeking():
    checked, bad = invariance_mismatches("intermediate", 1, make=lambda: Peeker(seed=0))
    assert bad > 0


def test_determinize_preserves_public_info():
    rng = random.Random(1)
    g = Game(seed=2)
    r = random.Random(0)
    for _ in range(400):
        g.step(r.choice(g.legal_actions()))
    me = g.actor
    for _ in range(10):
        c = C.determinize(g, me, rng)
        assert c.players[me].hand == g.players[me].hand and c.players[me].dev == g.players[me].dev
        assert c.bank == g.bank
        for q in range(g.n):
            assert sum(c.players[q].hand) == sum(g.players[q].hand)
            assert sum(c.players[q].dev) == sum(g.players[q].dev)
        tot = lambda x: sorted(x.dev_deck + [t for q in range(x.n) if q != me for t in range(5)
                                             for _ in range(x.players[q].dev[t])])
        assert tot(c) == tot(g)
