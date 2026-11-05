"""Invariants checked after every step of many random games."""
import json
import random

import pytest
from hypothesis import given, settings, strategies as st, HealthCheck

from catan.engine import A, Action, Phase, Game, GameConfig, IllegalAction, TOPOLOGY as T
from catan.engine.game import SUPPLY, VICTORY_POINT

PHASE_TYPES = {
    Phase.SETUP_SETTLEMENT: {A.BUILD_SETTLEMENT},
    Phase.SETUP_ROAD: {A.BUILD_ROAD},
    Phase.PRE_ROLL: {A.ROLL, A.PLAY_KNIGHT, A.PLAY_ROAD_BUILDING, A.PLAY_YEAR_OF_PLENTY, A.PLAY_MONOPOLY},
    Phase.DISCARD: {A.DISCARD},
    Phase.MOVE_ROBBER: {A.MOVE_ROBBER},
    Phase.STEAL: {A.STEAL},
    Phase.MAIN: {A.BUILD_SETTLEMENT, A.BUILD_ROAD, A.BUILD_CITY, A.BUY_DEV, A.PLAY_KNIGHT,
                 A.PLAY_ROAD_BUILDING, A.PLAY_YEAR_OF_PLENTY, A.PLAY_MONOPOLY, A.MARITIME_TRADE,
                 A.OFFER_TRADE, A.END_TURN},
    Phase.ROAD_BUILDING: {A.BUILD_ROAD},
    Phase.TRADE_RESPONSE: {A.ACCEPT_TRADE, A.REJECT_TRADE, A.COUNTER_OFFER},
    Phase.TRADE_CONFIRM: {A.CONFIRM_TRADE, A.CANCEL_TRADE},
    Phase.GAME_OVER: set(),
}


class Tracker:
    def __init__(self):
        self.dev_played = 0


def check(g: Game, tr: Tracker, legal):
    n = g.n
    # resources conserved, non-negative
    for r in range(5):
        assert g.bank[r] >= 0
        assert g.bank[r] + sum(pl.hand[r] for pl in g.players) == 19
    for pl in g.players:
        assert min(pl.hand) >= 0 and min(pl.dev) >= 0 and min(pl.dev_new) >= 0
        assert all(pl.dev_new[t] <= pl.dev[t] for t in range(5))
    # dev cards conserved
    held = sum(sum(pl.dev) for pl in g.players)
    assert len(g.dev_deck) + held + tr.dev_played == 25
    assert tr.dev_played >= sum(pl.knights for pl in g.players)
    # structures vs board and supply
    for q, pl in enumerate(g.players):
        assert len(pl.roads) <= SUPPLY["road"]
        assert len(pl.settlements) <= SUPPLY["settlement"] and len(pl.cities) <= SUPPLY["city"]
        assert sorted(pl.roads) == [e for e in range(72) if g.edge_owner[e] == q]
        assert sorted(pl.settlements + pl.cities) == [v for v in range(54) if g.vertex_owner[v] == q]
        assert all(g.vertex_level[v] == 1 for v in pl.settlements)
        assert all(g.vertex_level[v] == 2 for v in pl.cities)
        ports = {g._vertex_port[v] for v in pl.settlements + pl.cities if v in g._vertex_port}
        assert ports == pl.ports
        # every road component (ignoring blocking) contains an own building
        own = set(pl.roads)
        seen = set()
        for e0 in pl.roads:
            if e0 in seen:
                continue
            comp, stack, verts = set(), [e0], set()
            while stack:
                e = stack.pop()
                if e in comp:
                    continue
                comp.add(e)
                for v in T.edge_vertices[e]:
                    verts.add(v)
                    stack.extend(x for x in T.vertex_edges[v] if x in own and x not in comp)
            seen |= comp
            assert any(g.vertex_owner[v] == q for v in verts), (q, comp)
    # distance rule
    for v in range(54):
        if g.vertex_owner[v] != -1:
            assert g.vertex_level[v] in (1, 2)
            assert all(g.vertex_owner[u] == -1 for u in T.vertex_neighbors[v])
        else:
            assert g.vertex_level[v] == 0
    # awards and VP
    for q in range(n):
        assert g.road_length[q] == g.longest_road(q)
        pl = g.players[q]
        vp = len(pl.settlements) + 2 * len(pl.cities) + 2 * (g.longest_road_owner == q) \
            + 2 * (g.largest_army_owner == q)
        assert g.public_victory_points(q) == vp
        assert g.victory_points(q) == vp + pl.dev[VICTORY_POINT]
    if g.longest_road_owner != -1:
        lo = g.longest_road_owner
        assert g.road_length[lo] >= 5 and g.road_length[lo] == max(g.road_length)
    if g.largest_army_owner != -1:
        la = g.largest_army_owner
        k = g.players[la].knights
        assert k >= 3 and k == max(pl.knights for pl in g.players)
    if g.last_roll is not None:
        assert all(1 <= d <= 6 for d in g.last_roll)
    assert 0 <= g.robber < 19
    # no one but the current player may hold 10+ VP at a moment the game continues
    if not g.is_over and g.phase not in (Phase.SETUP_SETTLEMENT, Phase.SETUP_ROAD):
        assert g.victory_points(g.current) < g.config.vp_to_win
    # legal actions belong to actor and phase
    if g.is_over:
        assert legal == []
        assert g.winner != -1 or g.truncated
    else:
        assert legal, g.phase
        assert len(set(legal)) == len(legal)
        for a in legal:
            assert a.player == g.actor and a.type in PHASE_TYPES[g.phase], (g.phase, a)
    if g.phase == Phase.DISCARD:
        assert g.discard_remaining[g.actor] > 0
    # serializable
    json.dumps(g.to_dict())
    json.dumps(g.player_view(g.actor if not g.is_over else None))


def play(seed, n=4, max_turns=400, pick=None, check_every=1):
    g = Game(config=GameConfig(num_players=n, max_turns=max_turns), seed=seed)
    rng = random.Random(seed)
    tr = Tracker()
    i = 0
    legal = g.legal_actions()
    check(g, tr, legal)
    while not g.is_over:
        a = pick(rng, legal) if pick else rng.choice(legal)
        if i % 50 == 0:
            assert all(g.is_legal(x) for x in legal)
            other = Action((g.actor + 1) % n, a.type, a.value)
            assert not g.is_legal(other)
        ev = g.step(a)
        tr.dev_played += sum(e["type"] == "dev_played" for e in ev)
        for e in ev:
            if e["type"] == "dice_rolled":
                assert 2 <= e["total"] <= 12 and e["total"] == sum(e["dice"])
        legal = g.legal_actions()
        if i % check_every == 0 or g.is_over:
            check(g, tr, legal)
        i += 1
    with pytest.raises(IllegalAction):
        g.step(Action(0, A.END_TURN))
    return g


def builder_pick(rng, legal):
    """Bias towards building so games reach awards, supply limits and wins."""
    pri = [a for a in legal if a.type in (A.BUILD_CITY, A.BUILD_SETTLEMENT, A.BUY_DEV,
                                          A.PLAY_KNIGHT, A.PLAY_ROAD_BUILDING, A.CONFIRM_TRADE,
                                          A.ACCEPT_TRADE, A.PLAY_YEAR_OF_PLENTY, A.PLAY_MONOPOLY)]
    if pri and rng.random() < 0.9:
        return rng.choice(pri)
    roads = [a for a in legal if a.type == A.BUILD_ROAD]
    if roads and rng.random() < 0.7:
        return rng.choice(roads)
    trades = [a for a in legal if a.type == A.MARITIME_TRADE]
    if trades and rng.random() < 0.5:
        return rng.choice(trades)
    nontrade = [a for a in legal if a.type not in (A.OFFER_TRADE, A.COUNTER_OFFER)]
    return rng.choice(nontrade or legal)


@pytest.mark.parametrize("n", [2, 3, 4])
@pytest.mark.parametrize("seed", range(4))
def test_invariants_random_games(seed, n):
    play(seed, n, max_turns=120)


@pytest.mark.parametrize("seed", range(12))
def test_invariants_builder_games(seed):
    g = play(1000 + seed, 2 + seed % 3, max_turns=600, pick=builder_pick, check_every=1)
    assert g.is_over


def test_builder_games_reach_interesting_states():
    """Sanity: the biased policy actually exercises awards and wins."""
    wins = lr = la = 0
    for s in range(20):
        g = play(2000 + s, 4, max_turns=600, pick=builder_pick, check_every=10**9)
        wins += g.winner != -1
        lr += g.longest_road_owner != -1
        la += g.largest_army_owner != -1
    assert wins >= 10 and lr >= 3 and la >= 3


@settings(max_examples=25, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(seed=st.integers(0, 2**31 - 1), n=st.integers(2, 4), data=st.data())
def test_hypothesis_replay_and_clone(seed, n, data):
    g = Game(config=GameConfig(num_players=n, max_turns=200), seed=seed)
    steps = data.draw(st.integers(0, 1500))
    rng = random.Random(seed)
    for _ in range(steps):
        if g.is_over:
            break
        g.step(builder_pick(rng, g.legal_actions()))
    d = g.to_dict()
    assert Game.replay(seed, g.config, g.action_log).to_dict() == d
    assert Game.from_dict(json.loads(json.dumps(d))).to_dict() == d
    c = g.clone()
    if not c.is_over:
        c.step(c.legal_actions()[0])
    assert g.to_dict() == d


@settings(max_examples=40, deadline=None)
@given(seed=st.integers(0, 2**31 - 1), steps=st.integers(0, 800),
       ptype=st.sampled_from(sorted(set().union(*PHASE_TYPES.values()))),
       value=st.one_of(st.none(), st.integers(-5, 80), st.booleans(), st.floats(),
                       st.tuples(st.integers(-1, 5), st.integers(-1, 5)),
                       st.tuples(st.tuples(*[st.integers(-1, 3)] * 5), st.tuples(*[st.integers(-1, 3)] * 5))))
def test_hypothesis_arbitrary_actions_never_corrupt(seed, steps, ptype, value):
    g = Game(seed=seed, config=GameConfig(max_turns=100))
    rng = random.Random(seed)
    for _ in range(steps):
        if g.is_over:
            break
        g.step(rng.choice(g.legal_actions()))
    if g.is_over:
        return
    before = json.dumps(g.to_dict())
    a = Action(g.actor, ptype, value)
    if g.is_legal(a):
        g.step(a)
        tr = Tracker()
        tr.dev_played = 25 - len(g.dev_deck) - sum(sum(pl.dev) for pl in g.players)
        check(g, tr, g.legal_actions())
    else:
        with pytest.raises(IllegalAction):
            g.step(a)
        assert json.dumps(g.to_dict()) == before
