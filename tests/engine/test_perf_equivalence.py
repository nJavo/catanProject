"""Performance rewrites vs. the original (pre-optimization) engine code.

``ref_legal_actions`` / ``ref_legal_road_edges`` are verbatim copies of the
straightforward implementations the fast paths replaced. Any difference in
content, order or value types is a semantics bug.
"""
import random

from catan.engine import A, Action, Phase, Game, GameConfig, COST, TOPOLOGY as T
from catan.engine.game import (KNIGHT, ROAD_BUILDING, YEAR_OF_PLENTY, MONOPOLY, SUPPLY,
                               TRADE_TEMPLATES)

from .test_oracle import _positions

VE, EV = T.vertex_edges, T.edge_vertices


def ref_legal_road_edges(g, p):
    pl = g.players[p]
    if len(pl.roads) >= SUPPLY["road"]:
        return []
    cand = set()
    for e in pl.roads:
        a, b = EV[e]
        cand.update(VE[a])
        cand.update(VE[b])
    for v in pl.settlements:
        cand.update(VE[v])
    for v in pl.cities:
        cand.update(VE[v])
    return sorted(e for e in cand if g._road_ok(p, e))


def ref_legal_actions(g):
    ph = g.phase
    p = g.actor
    if ph == Phase.GAME_OVER:
        return []
    if ph == Phase.SETUP_SETTLEMENT:
        return [Action(p, A.BUILD_SETTLEMENT, v) for v in range(T.num_vertices) if g._settlement_spot_free(v)]
    if ph == Phase.SETUP_ROAD:
        return [Action(p, A.BUILD_ROAD, e) for e in T.vertex_edges[g.last_settlement]
                if g.edge_owner[e] == -1]
    if ph == Phase.DISCARD:
        h = g.players[p].hand
        return [Action(p, A.DISCARD, r) for r in range(5) if h[r] > 0]
    if ph == Phase.MOVE_ROBBER:
        return [Action(p, A.MOVE_ROBBER, h) for h in range(T.num_hexes) if h != g.robber]
    if ph == Phase.STEAL:
        return [Action(p, A.STEAL, v) for v in g._robber_victims(p)]
    if ph == Phase.ROAD_BUILDING:
        return [Action(p, A.BUILD_ROAD, e) for e in ref_legal_road_edges(g, p)]
    if ph == Phase.TRADE_RESPONSE:
        tr = g.trade
        out = [Action(p, A.REJECT_TRADE)]
        if g.can_afford(p, tr["get"]):
            out.append(Action(p, A.ACCEPT_TRADE))
        h = g.players[p].hand
        cur = (tr["give"], tr["get"])
        for give, get in TRADE_TEMPLATES:
            ri = next(i for i in range(5) if get[i])
            if h[ri] >= get[ri] and (give, get) != cur:
                out.append(Action(p, A.COUNTER_OFFER, (give, get)))
        return out
    if ph == Phase.TRADE_CONFIRM:
        out = [Action(p, A.CANCEL_TRADE)]
        for q, resp in sorted(g.trade["responses"].items()):
            if resp is not None and g.can_afford(p, resp[0]):
                out.append(Action(p, A.CONFIRM_TRADE, q))
        return out
    out = []
    pl = g.players[p]
    if g._playable_dev(p, KNIGHT):
        out.append(Action(p, A.PLAY_KNIGHT))
    if g._playable_dev(p, ROAD_BUILDING) and ref_legal_road_edges(g, p):
        out.append(Action(p, A.PLAY_ROAD_BUILDING))
    if g._playable_dev(p, YEAR_OF_PLENTY):
        for a in range(5):
            for b in range(a, 5):
                need = [0] * 5
                need[a] += 1
                need[b] += 1
                if all(g.bank[i] >= need[i] for i in range(5)):
                    out.append(Action(p, A.PLAY_YEAR_OF_PLENTY, (a, b)))
    if g._playable_dev(p, MONOPOLY):
        out.extend(Action(p, A.PLAY_MONOPOLY, r) for r in range(5))
    if ph == Phase.PRE_ROLL:
        out.append(Action(p, A.ROLL))
        return out
    out.append(Action(p, A.END_TURN))
    if g.can_afford(p, COST["road"]):
        out.extend(Action(p, A.BUILD_ROAD, e) for e in ref_legal_road_edges(g, p))
    if g.can_afford(p, COST["settlement"]):
        out.extend(Action(p, A.BUILD_SETTLEMENT, v) for v in g.legal_settlement_vertices(p))
    if g.can_afford(p, COST["city"]) and g.supply(p, "city") > 0:
        out.extend(Action(p, A.BUILD_CITY, v) for v in pl.settlements)
    if g.can_afford(p, COST["dev"]) and g.dev_deck:
        out.append(Action(p, A.BUY_DEV))
    for gi in range(5):
        if pl.hand[gi] >= g.trade_rate(p, gi):
            out.extend(Action(p, A.MARITIME_TRADE, (gi, r)) for r in range(5) if r != gi and g.bank[r] > 0)
    if g._can_offer(p):
        for give, get in TRADE_TEMPLATES:
            gi = next(i for i in range(5) if give[i])
            if pl.hand[gi] >= give[gi]:
                out.append(Action(p, A.OFFER_TRADE, (give, get)))
    return out


def _same(a, b):
    return a == b and [type(x.value) for x in a] == [type(x.value) for x in b]


def test_legal_actions_match_reference():
    n = 0
    phases = set()
    for g, _ in _positions(range(16), steps=3000):
        assert _same(g.legal_actions(), ref_legal_actions(g)), g.phase
        for q in range(g.n):
            assert g.legal_road_edges(q) == ref_legal_road_edges(g, q)
        phases.add(g.phase)
        n += 1
    assert n > 20000
    assert phases >= {Phase.SETUP_SETTLEMENT, Phase.SETUP_ROAD, Phase.PRE_ROLL, Phase.MAIN,
                      Phase.DISCARD, Phase.MOVE_ROBBER, Phase.STEAL, Phase.ROAD_BUILDING,
                      Phase.TRADE_RESPONSE, Phase.TRADE_CONFIRM}


def test_legal_actions_match_reference_with_edited_hands_and_bank():
    """Random hands/banks hit affordability and port-rate corners that real
    games rarely reach (e.g. a counter-offer equal to the pending offer)."""
    rng = random.Random(1)
    checked = 0
    for g, i in _positions(range(6), steps=2000):
        if i % 5 or g.phase not in (Phase.MAIN, Phase.PRE_ROLL, Phase.TRADE_RESPONSE):
            continue
        c = g.clone()
        for pl in c.players:
            pl.hand = [rng.choice((0, 0, 1, 2, 3, 4, 7)) for _ in range(5)]
            pl.dev = [rng.randint(0, 2) for _ in range(5)]
            pl.dev_new = [min(d, rng.randint(0, 1)) for d in pl.dev]
        c.bank = [rng.choice((0, 1, 2, 5)) for _ in range(5)]
        c.dev_played = rng.random() < 0.3
        assert _same(c.legal_actions(), ref_legal_actions(c)), c.phase
        checked += 1
    assert checked > 300


def _play(g, rng, k):
    acts = []
    for _ in range(k):
        if g.is_over:
            break
        a = rng.choice(g.legal_actions())
        acts.append(a)
        g.step(a)
    return acts


def test_clone_has_identical_future_and_is_independent():
    for g, i in _positions(range(6), steps=1500):
        if i % 97:
            continue
        before = g.to_dict()
        c = g.clone()
        cc = c.clone()
        assert c.to_dict() == before and cc.to_dict() == before
        acts = _play(c, random.Random(i), 60)   # uses dice + steal RNG
        assert g.to_dict() == before            # source untouched
        for a in acts:
            cc.step(a)
        assert cc.to_dict() == c.to_dict()      # same actions -> same future
        assert c.rng.random() == cc.rng.random()


def test_clone_rng_copy_matches_original_draws():
    g = Game(seed=5)
    g.rng.random()
    c = g.clone()
    assert type(c.rng) is random.Random and c.rng is not g.rng
    assert [g.rng.random() for _ in range(5)] == [c.rng.random() for _ in range(5)]
    assert g.rng.gauss(0, 1) == c.rng.gauss(0, 1)


def test_copy_on_write_clone_family_matches_independent_copies():
    """Clones share RNG and action log copy-on-write. Interleave steps, rng
    draws, log reads and re-cloning across a family of games; each must stay
    identical to an independent copy (from_dict(to_dict())) made at the same
    moment and driven by the same actions."""
    rng = random.Random(7)
    for seed in range(4):
        root = Game(config=GameConfig(max_turns=200), seed=seed)
        fam = [(root, Game.from_dict(root.to_dict()))]
        for _ in range(400):
            i = rng.randrange(len(fam))
            g, ref = fam[i]
            op = rng.random()
            if op < 0.15 and len(fam) < 12:
                fam.append((g.clone(), Game.from_dict(g.to_dict())))
            elif op < 0.2:
                assert g.action_log == ref.action_log
            elif op < 0.23:
                x = g.rng.random()  # external draw counts as use
                assert x == ref.rng.random()
            elif not g.is_over:
                a = rng.choice(g.legal_actions())
                assert g.step(a) == ref.step(a)
        for g, ref in fam:
            assert g.to_dict() == ref.to_dict()
            assert g.action_log == ref.action_log


def test_rng_setter_and_from_dict_unshare():
    g = Game(seed=3)
    c = g.clone()
    c.rng = random.Random(1)  # as determinize does
    assert c.rng is not g.rng
    before = g.rng.getstate()
    c.rng.random()
    assert g.rng.getstate() == before
    d = Game.from_dict(g.clone().to_dict())
    assert d.rng.getstate() == before


def test_legal_actions_only_return_prebuilt_objects():
    """catan.rl.spaces keys a cache on id(action); that is only sound if every
    returned action is one of the immortal PREBUILT_ACTIONS objects."""
    from catan.engine.game import PREBUILT_ACTIONS
    ids = {id(a) for a in PREBUILT_ACTIONS}
    for g, _ in _positions(range(6), steps=2000):
        assert all(id(a) in ids for a in g.legal_actions()), g.phase


def test_public_longest_trail_matches_reference_with_an_extra_road():
    from catan.engine.game import longest_trail
    from .test_oracle import ref_longest_road
    n = 0
    for g, i in _positions(range(6), steps=1500):
        if i % 11:
            continue
        for q in range(g.n):
            roads = g.players[q].roads
            assert longest_trail(roads, g.vertex_owner, q) == g.longest_road(q)
            for e in g.legal_road_edges(q)[:3]:
                c = g.clone()
                c.edge_owner[e] = q
                c.players[q].roads.append(e)
                assert longest_trail(roads + [e], g.vertex_owner, q) == ref_longest_road(c, q)
                n += 1
    assert n > 300
