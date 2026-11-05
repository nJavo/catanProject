"""TR-002 acceptance at scale.

1. The belief tracker stays consistent with the truth over 200 games: rows match
   the public hand sizes, columns the public per-resource totals, and while no
   steal is unseen the estimate is the true hand.
2. The trader's caches (partner value context, d50224b; tracker estimate) are
   decision-neutral: identical action logs with the caches on and off.
"""
import json

import pytest

from catan.agents.registry import make_agent
from catan.bots import trading as TR
from catan.engine import Game, GameConfig

from test_trading import run_tracked

LINEUP = ("intermediate", "random", "basic", "intermediate:trader=true")


def test_tracker_consistent_over_200_games():
    checked = exact = 0
    for seed in range(200):
        me = seed % 4
        for k, (g, t) in enumerate(run_tracked(seed, me=me, lineup=LINEUP[me:] + LINEUP[:me])):
            if k % 3:
                continue
            est = t.estimate(g)
            assert not t.lost
            for q in range(g.n):
                assert abs(sum(est[q]) - sum(g.players[q].hand)) < 0.01, (seed, q)
                assert min(est[q]) >= 0
            for r in range(5):
                assert abs(sum(est[q][r] for q in range(g.n)) - (19 - g.bank[r])) < 0.01, (seed, r)
            if not t.unseen_steals:
                assert all(abs(est[q][r] - g.players[q].hand[r]) < 0.02 for q in range(g.n) for r in range(5))
                exact += 1
            checked += 1
    assert checked > 20_000 and exact > 2_000


def _play(seed, caches: bool):
    ags = [make_agent("intermediate_trader", seed=seed * 10 + i) for i in range(2)] + \
          [make_agent("intermediate", seed=seed * 10 + 2), make_agent("basic", seed=seed * 10 + 3)]
    for i, a in enumerate(ags):
        a.reset(i, seed * 10 + i)
    g = Game(config=GameConfig(max_turns=150), seed=seed)
    while not g.is_over:
        evs = g.step(ags[g.actor].act(g, g.legal_actions()))
        for i, a in enumerate(ags):
            if hasattr(a, "observe"):
                a.observe(g, [e for e in evs if "visible_to" not in e or i in e["visible_to"]])
            if not caches:
                a._pctx = None
                if getattr(a, "tracker", None) is not None:
                    a.tracker._cache_key = None
    return json.dumps([a.to_json() for a in g.action_log])


@pytest.fixture
def no_caches(monkeypatch):
    """Clear the per-step caches before every lookup, not just between steps."""
    pg, est = TR.TraderMixin.partner_gain, TR.BeliefTracker.estimate

    def partner_gain(self, *a, **k):
        self._pctx = None
        return pg(self, *a, **k)

    def estimate(self, g):
        self._cache_key = None
        return est(self, g)
    monkeypatch.setattr(TR.TraderMixin, "partner_gain", partner_gain)
    monkeypatch.setattr(TR.BeliefTracker, "estimate", estimate)


def test_trader_caches_are_decision_neutral(request):
    with_caches = [_play(s, True) for s in range(12)]
    request.getfixturevalue("no_caches")
    without = [_play(s, False) for s in range(12)]
    assert with_caches == without
