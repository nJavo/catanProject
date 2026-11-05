"""Exact hand posterior (catan/bots/belief.py): exact, info-safe, consistent with public counts."""
import random

from catan.agents.registry import make_agent
from catan.bots import common as C
from catan.bots.belief import HandPosterior, visible
from catan.engine import Game, GameConfig

from test_bots import FAST, invariance_mismatches, resample_hidden

LINEUP = ["intermediate", "basic", "intermediate_trader", "random"]


def games(seeds, max_turns=200):
    """Yield (game, events) after every step of seeded mixed-bot games."""
    for seed in seeds:
        ags = [make_agent(x, seed=seed * 4 + i) for i, x in enumerate(LINEUP)]
        for i, a in enumerate(ags):
            a.reset(i, seed * 4 + i)
        g = Game(config=GameConfig(max_turns=max_turns), seed=seed)
        while not g.is_over:
            ev = g.step(ags[g.actor].act(g, g.legal_actions()))
            yield g, ev


def test_posterior_is_exact_and_consistent():
    posts, steals, last = {}, 0, None
    for g, ev in games(range(6)):
        if g is not last:
            posts, last = {m: HandPosterior(g.n, m) for m in range(g.n)}, g
        for m, P in posts.items():
            P.observe(g, visible(ev, m))
            assert not P.lost
            true = tuple(tuple(g.players[q].hand) for q in P.opps)
            assert true in P.states  # the true hands are always in the support
            assert abs(sum(P.states.values()) - 1) < 1e-9
            cols = C.opponents_resource_totals(g, m)
            for s in P.states:  # every state matches hand sizes and per-resource totals
                assert [sum(h) for h in s] == [C.hand_count(g, q) for q in P.opps]
                assert [sum(h[r] for h in s) for r in range(5)] == cols
        steals = max(steals, max(P.unseen_steals for P in posts.values()))
    assert steals > 0  # the games exercised unseen steals


def test_posterior_ignores_hidden_events_and_hidden_state():
    """Same posterior when (a) hidden events are tampered with and (b) the game passed in has its
    hidden state resampled: the posterior reads only public events and public counts."""
    rng = random.Random(0)
    a = b = c = None
    last = None
    for g, ev in games([3]):
        if g is not last:
            a, b, c, last = HandPosterior(g.n, 0), HandPosterior(g.n, 0), HandPosterior(g.n, 0), g
        a.observe(g, ev)  # filters by visible_to itself
        tampered = [dict(e, resource=(e["resource"] + 1) % 5) if e["type"] == "stolen_detail"
                    and 0 not in e["visible_to"] else e for e in ev]
        tampered += [{"type": "dev_bought_detail", "player": 1, "card": "monopoly", "visible_to": [1]}]
        b.observe(g, tampered)
        c.observe(resample_hidden(g, 0, rng), ev)
        assert a.states == b.states == c.states


def test_belief_determinize_matches_public_counts_and_trades():
    rng = random.Random(1)
    checked = trades = 0
    last = None
    for g, ev in games(range(2)):
        if g is not last:
            P, last = HandPosterior(g.n, 0), g
        P.observe(g, visible(ev, 0))
        if g.is_over or len(g.action_log) % 5:
            continue
        for _ in range(3):
            d = C.determinize(g, 0, rng, belief=P)
            assert d.players[0].hand == g.players[0].hand and d.bank == g.bank
            assert [sum(d.players[q].hand) for q in range(g.n)] == [sum(g.players[q].hand) for q in range(g.n)]
            assert tuple(tuple(d.players[q].hand) for q in P.opps) in P.states
            if g.trade is not None and g.trade["proposer"] != 0:
                trades += 1
                assert all(d.players[g.trade["proposer"]].hand[r] >= g.trade["give"][r] for r in range(5))
            checked += 1
    assert checked > 100 and trades > 0


def test_default_determinize_is_unchanged():
    g = Game(seed=4)
    r = random.Random(0)
    for _ in range(300):
        g.step(r.choice(g.legal_actions()))
    x = C.determinize(g, g.actor, random.Random(9))
    y = C.determinize(g, g.actor, random.Random(9), belief=None)
    assert [p.hand for p in x.players] == [p.hand for p in y.players] and x.dev_deck == y.dev_deck


def test_search_beliefs_hidden_info_invariance():
    kw = dict(FAST["search"], beliefs=True)
    checked, bad = invariance_mismatches("search", 6, make=lambda: make_agent("search", seed=0, **kw))
    assert checked > 20 and bad == 0


def test_search_beliefs_uses_posterior_when_observed():
    ag = make_agent("search", seed=0, rollouts=2, candidates=3, beliefs=True)
    others = [make_agent("basic", seed=i) for i in range(3)]
    ag.reset(0, 0)
    g = Game(config=GameConfig(max_turns=40), seed=5)
    while not g.is_over:
        p = g.actor
        a = ag.act(g, g.legal_actions()) if p == 0 else others[p - 1].act(g, g.legal_actions())
        ag.observe(g, visible(g.step(a), 0))
    assert ag.posterior is not None and not ag.posterior.lost and ag.belief_fallbacks == 0


def test_from_history_matches_incremental():
    g = Game(seed=7)
    r = random.Random(2)
    inc = HandPosterior(g.n, 1)
    acts = []
    for _ in range(250):
        a = r.choice(g.legal_actions())
        acts.append(a)
        inc.observe(g, visible(g.step(a), 1))
    post, g2 = HandPosterior.from_history(7, g.config, acts, 1)
    assert post.states == inc.states and [p.hand for p in g2.players] == [p.hand for p in g.players]
