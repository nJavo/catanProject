"""Optimized engine paths vs. straightforward reference implementations
(the original engine code), on many random positions."""
import random


from catan.engine import A, Action, Phase, Game, GameConfig, TOPOLOGY as T, TRADE_TEMPLATES
from catan.engine.actions import ALL_ACTION_TYPES

from .test_invariants import builder_pick


def ref_longest_road(g, p):
    edges = g.players[p].roads
    if not edges:
        return 0
    eset, vo, best = set(edges), g.vertex_owner, 0

    def dfs(v, used, length):
        nonlocal best
        best = max(best, length)
        if used and vo[v] != -1 and vo[v] != p:
            return
        for e in T.vertex_edges[v]:
            if e in eset and e not in used:
                a, b = T.edge_vertices[e]
                used.add(e)
                dfs(b if a == v else a, used, length + 1)
                used.discard(e)

    for v in {v for e in edges for v in T.edge_vertices[e]}:
        dfs(v, set(), 0)
    return best


def ref_is_legal(g, a):
    """Original semantics: membership in legal_actions, plus any well-formed
    domestic offer / counter-offer."""
    if not isinstance(a, Action) or type(a.player) is not int or a.player != g.actor or g.is_over:
        return False
    v = a.value
    if a.type == A.OFFER_TRADE and g.phase == Phase.MAIN:
        return (g._can_offer(a.player) and isinstance(v, tuple) and len(v) == 2
                and g._valid_offer(v[0], v[1], a.player, None))
    if a.type == A.COUNTER_OFFER and g.phase == Phase.TRADE_RESPONSE:
        return (isinstance(v, tuple) and len(v) == 2 and v != (g.trade["give"], g.trade["get"])
                and g._valid_offer(v[0], v[1], None, a.player))
    return any(a == x and type(x.value) is type(a.value) for x in g.legal_actions())


CAND_VALUES = ([None] + list(range(-1, 73)) + [(a, b) for a in range(-1, 6) for b in range(-1, 6)]
               + list(TRADE_TEMPLATES[:12]) + [((2, 0, 0, 0, 0), (0, 0, 0, 1, 1)), (5, 5)])


def _positions(seeds, steps=4000):
    for s in seeds:
        g = Game(config=GameConfig(num_players=2 + s % 3, max_turns=300), seed=s)
        rng = random.Random(s)
        pick = builder_pick if s % 2 else (lambda r, legal: r.choice(legal))
        for i in range(steps):
            if g.is_over:
                break
            yield g, i
            g.step(pick(rng, g.legal_actions()))


def test_longest_road_matches_reference():
    n = 0
    for g, _ in _positions(range(12)):
        for q in range(g.n):
            assert g.longest_road(q) == ref_longest_road(g, q)
        n += 1
    assert n > 4000


def test_longest_road_reference_handcrafted():
    from .helpers import main_game, put_road, put_settlement
    rng = random.Random(0)
    for trial in range(300):
        g = main_game()
        for e in rng.sample(range(72), rng.randint(1, 15)):
            put_road(g, 0, e)
        for v in rng.sample(range(54), rng.randint(0, 6)):
            if g.vertex_owner[v] == -1:
                put_settlement(g, rng.randint(0, 1), v)
        assert g.longest_road(0) == ref_longest_road(g, 0), trial


def test_is_legal_matches_reference():
    checked = 0
    for g, i in _positions(range(12), steps=1500):
        if i % 17:
            continue
        p = g.actor
        for t in ALL_ACTION_TYPES:
            for v in CAND_VALUES:
                a = Action(p, t, v)
                assert g.is_legal(a) == ref_is_legal(g, a), (g.phase, a)
                checked += 1
        legal = g.legal_actions()
        assert all(g.is_legal(a) for a in legal)
    assert checked > 40000
