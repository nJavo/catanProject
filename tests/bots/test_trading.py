import random

from catan.agents.registry import make_agent
from catan.bots.trading import BeliefTracker, bundles, prob_at_least
from catan.engine import Game, GameConfig
from catan.engine.game import TRADE_TEMPLATES
from test_bots import resample_hidden


def visible(evs, q):
    return [e for e in evs if "visible_to" not in e or q in e["visible_to"]]


def run_tracked(seed, me=0, lineup=("intermediate", "random", "basic", "intermediate:trader=true"), steps=None):
    """Yield (game, tracker) after each step of a game, tracker fed seat ``me``'s events."""
    agents = []
    for i, spec in enumerate(lineup):
        name, _, kw = spec.partition(":")
        kwargs = dict(x.split("=") for x in kw.split(",")) if kw else {}
        kwargs = {k: v == "true" if v in ("true", "false") else v for k, v in kwargs.items()}
        a = make_agent(name, seed=seed * 10 + i, **kwargs)
        a.reset(i, seed * 10 + i)
        agents.append(a)
    g = Game(config=GameConfig(max_turns=300), seed=seed)
    t = BeliefTracker(g.n, me)
    k = 0
    while not g.is_over and (steps is None or k < steps):
        evs = g.step(agents[g.actor].act(g, g.legal_actions()))
        for i, a in enumerate(agents):
            if hasattr(a, "observe"):
                a.observe(g, visible(evs, i))
        t.observe(g, visible(evs, me))
        k += 1
        yield g, t


def test_tracker_exact_until_unseen_steal():
    checked = 0
    for seed in range(4):
        for g, t in run_tracked(seed):
            if t.unseen_steals:
                break
            est = t.estimate(g)
            for q in range(g.n):
                assert all(abs(est[q][r] - g.players[q].hand[r]) < 0.02 for r in range(5)), (q, est[q])
            checked += 1
    assert checked > 200


def test_tracker_consistent_and_better_than_prior():
    err_t = err_p = 0.0
    for seed in range(3):
        for g, t in run_tracked(seed):
            est = t.estimate(g)
            prior = BeliefTracker(g.n, 0)
            prior.lost = True
            pe = prior.estimate(g)
            for q in range(1, g.n):
                assert abs(sum(est[q]) - sum(g.players[q].hand)) < 0.01
                err_t += sum(abs(est[q][r] - g.players[q].hand[r]) for r in range(5))
                err_p += sum(abs(pe[q][r] - g.players[q].hand[r]) for r in range(5))
            for r in range(5):  # per-resource totals match the bank
                tot = sum(est[q][r] for q in range(g.n))
                assert abs(tot - (19 - g.bank[r])) < 0.01
    assert err_t < 0.5 * err_p


def test_tracker_info_safe():
    rng = random.Random(0)
    n = 0
    for g, t in run_tracked(5, steps=400):
        if n % 20 == 0:
            h = resample_hidden(g, 0, rng)
            assert t.estimate(g) == t.estimate(h)
        n += 1


def test_tracker_detects_missed_steps():
    g = Game(seed=1)
    rng = random.Random(0)
    t = BeliefTracker(4, 0)
    g.step(rng.choice(g.legal_actions()))
    g.step(rng.choice(g.legal_actions()))
    t.observe(g, [])
    assert t.lost


def test_prob_and_bundles():
    assert prob_at_least(0, 5, 1) == 0.0 and prob_at_least(5, 5, 5) == 1.0
    assert 0 < prob_at_least(1.0, 4, 1) < 1
    b = bundles((2, 2, 2, 2, 2), 2, 2)
    assert set(TRADE_TEMPLATES) <= set(b)
    assert all(not any(x[i] and y[i] for i in range(5)) for x, y in b)
    assert bundles((0, 0, 0, 0, 0)) == []


def test_trader_games_legal_and_use_counters():
    counters = 0
    for g, t in run_tracked(7):
        pass
    for a in g.action_log:
        counters += a.type == "COUNTER_OFFER" and a.player in (0, 3)
    assert g.is_over and counters > 0


def test_negotiation_features_shape_and_safety():
    from catan.bots.trading import negotiation_feature_names, negotiation_features
    rng = random.Random(1)
    k = 0
    for g, t in run_tracked(3, steps=300):
        if g.is_over or k % 25:
            k += 1
            continue
        me = g.actor
        tt = t if me == 0 else BeliefTracker(g.n, me)
        if me != 0:
            tt.lost = True
        x = negotiation_features(g, me, tt.estimate(g))
        assert len(x) == len(negotiation_feature_names(g.n))
        h = resample_hidden(g, me, rng)
        assert x == negotiation_features(h, me, tt.estimate(h))
        k += 1
