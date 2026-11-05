import copy

import pytest

from catan.analysis import determinize, position_from_game, validate_position, derived
from catan.engine import Game, Phase
from catan.engine.game import DEV_DECK

from .helpers import play_until, sampled_positions

POSITIONS = sampled_positions()


def test_sampled_positions_cover_phases():
    phases = {g.phase for g, _ in POSITIONS}
    assert {Phase.SETUP_SETTLEMENT, Phase.MAIN, Phase.PRE_ROLL} <= phases
    assert phases & {Phase.TRADE_RESPONSE, Phase.TRADE_CONFIRM}


@pytest.mark.parametrize("i", range(0, len(POSITIONS), max(1, len(POSITIONS) // 60)))
def test_real_positions_validate_and_determinize_consistently(i):
    g, v = POSITIONS[i]
    if g.is_over:
        pytest.skip("terminal")
    spec = position_from_game(g, v)
    assert validate_position(spec) == []
    dets = determinize(spec, 4, seed=i)
    for d in dets:
        assert d.phase == g.phase and d.actor == g.actor and d.current == g.current
        assert d.bank == g.bank and len(d.dev_deck) == len(g.dev_deck)
        assert d.road_length == g.road_length
        assert (d.longest_road_owner, d.largest_army_owner) == (g.longest_road_owner, g.largest_army_owner)
        assert d.players[v].hand == g.players[v].hand and d.players[v].dev == g.players[v].dev
        for q in range(g.n):
            assert sum(d.players[q].hand) == sum(g.players[q].hand)
            assert sum(d.players[q].dev) == sum(g.players[q].dev)
            assert sum(d.players[q].dev_new) == sum(g.players[q].dev_new)
        # conservation: 19 per resource, dev composition 25
        for r in range(5):
            assert d.bank[r] + sum(p.hand[r] for p in d.players) == 19
        played = [DEV_DECK[t] - sum(p.dev[t] for p in g.players) - g.dev_deck.count(t) for t in range(5)]
        for t in range(5):
            assert d.dev_deck.count(t) + sum(p.dev[t] for p in d.players) + played[t] == DEV_DECK[t]
        if g.actor == v:
            assert sorted(d.legal_actions()) == sorted(g.legal_actions())
        assert position_from_game(d, v) == spec  # round trip
        if g.trade is not None:  # pending trade stays executable
            tr = d.trade
            assert all(d.players[tr["proposer"]].hand[k] >= tr["give"][k] for k in range(5))


def test_known_hand_min_honored():
    g = play_until(1, lambda g: g.turn >= 20 and g.phase == Phase.MAIN and g.actor == 0)
    spec = position_from_game(g, 0)
    opp = max((1, 2, 3), key=lambda q: spec["players"][q]["resource_count"])
    k = spec["players"][opp]["resource_count"]
    assert k >= 2
    unseen = derived(spec)["unseen_resources"]
    r = max(range(5), key=lambda r: unseen[r])
    km = [0] * 5
    km[r] = min(2, unseen[r])
    spec["players"][opp]["known_hand_min"] = km
    assert validate_position(spec) == []
    for d in determinize(spec, 30, seed=3):
        assert d.players[opp].hand[r] >= km[r]
        assert sum(d.players[opp].hand) == k


def test_no_leak_opponent_hidden_state():
    g = play_until(2, lambda g: g.turn >= 25 and g.phase == Phase.MAIN and g.actor == 0)
    spec = position_from_game(g, 0)
    h = copy.deepcopy(g)
    # move a card between two opponents' hands keeping counts; swap a dev card with the deck
    a, b = h.players[1].hand, h.players[2].hand
    ra = next(r for r in range(5) if a[r]); rb = next((r for r in range(5) if b[r] and r != ra), None)
    if rb is not None:
        a[ra] -= 1; a[rb] += 1; b[rb] -= 1; b[ra] += 1
    for q in (1, 2, 3):
        d = h.players[q].dev
        t = next((t for t in range(5) if d[t] - h.players[q].dev_new[t] > 0), None)
        if t is not None:
            u = next((x for x in h.dev_deck if x != t), None)
            if u is not None:
                h.dev_deck.remove(u); h.dev_deck.append(t); d[t] -= 1; d[u] += 1
                break
    h.dev_deck.reverse()
    h.rng.seed(999)
    assert position_from_game(h, 0) == spec
    assert [x.to_dict() for x in determinize(spec, 3, 5)] == \
           [x.to_dict() for x in determinize(position_from_game(h, 0), 3, 5)]


def test_determinism_and_fresh_rng():
    g = play_until(3, lambda g: g.turn >= 10 and g.phase == Phase.PRE_ROLL and g.actor == 1)
    spec = position_from_game(g, 1)
    a = [x.to_dict() for x in determinize(spec, 5, 11)]
    assert a == [x.to_dict() for x in determinize(spec, 5, 11)]
    assert a != [x.to_dict() for x in determinize(spec, 5, 12)]
    assert len({x["seed"] for x in a}) == 5


def test_setup_positions():
    g = Game(seed=4)
    for _ in range(5):  # 2 settlements + 2 roads + 1 settlement placed: seat 2 must build its road
        g.step(g.legal_actions()[-1])
    assert g.phase == Phase.SETUP_ROAD
    spec = position_from_game(g, g.actor)
    assert validate_position(spec) == [] and derived(spec)["setup_index"] == 2
    d = determinize(spec, 1, 0)[0]
    assert d.actor == g.actor and sorted(d.legal_actions()) == sorted(g.legal_actions())


def test_does_not_mutate_input():
    g = play_until(5, lambda g: g.turn >= 5 and g.phase == Phase.MAIN)
    spec = position_from_game(g, g.actor)
    for p in spec["players"]:
        p.pop("knights_played")
    before = copy.deepcopy(spec)
    validate_position(spec); determinize(spec, 1, 0)
    assert spec == before


# ----------------------------------------------------------- validation
@pytest.fixture(scope="module")
def base():
    g = play_until(6, lambda g: g.turn >= 20 and g.phase == Phase.MAIN and g.actor == 0)
    spec = position_from_game(g, 0)
    assert validate_position(spec) == []
    return spec


def _err(spec, fn, needle):
    s = copy.deepcopy(spec)
    fn(s)
    errs = validate_position(s)
    assert any(needle in e for e in errs), errs


def test_validation_catches_illegal_positions(base):
    occupied = next(v for v in range(54) if base["vertex_owner"][v] == 0)
    from catan.engine import TOPOLOGY as T
    nb = T.vertex_neighbors[occupied][0]

    def adjacent(s):
        s["vertex_owner"][nb], s["vertex_level"][nb] = 1, 1
    _err(base, adjacent, "distance rule")

    far_edge = next(e for e in range(72) if base["edge_owner"][e] == -1
                    and all(base["vertex_owner"][v] == -1 and all(base["edge_owner"][x] == -1 for x in T.vertex_edges[v])
                            for v in T.edge_vertices[e]))
    _err(base, lambda s: s["edge_owner"].__setitem__(far_edge, 2), "not connected")
    _err(base, lambda s: s["vertex_level"].__setitem__(occupied, 0), "owner and level disagree")
    _err(base, lambda s: s["layout"]["hex_terrain"].__setitem__(0, "desert" if s["layout"]["hex_terrain"][0] != "desert" else "ore"), "base set")
    _err(base, lambda s: s["bank"].__setitem__(0, s["bank"][0] + 1), "conservation")
    _err(base, lambda s: s["players"][0]["hand"].__setitem__(1, 19), "> 19")
    _err(base, lambda s: s.__setitem__("dev_deck_count", (s["dev_deck_count"] + 1) % 26), "dev conservation")
    _err(base, lambda s: s.__setitem__("phase", "STEAL"), "STEAL needs")
    _err(base, lambda s: s.__setitem__("phase", "GAME_OVER"), "unsupported phase")
    _err(base, lambda s: s.__setitem__("rolled", False), "MAIN requires rolled=true")
    _err(base, lambda s: s.__setitem__("config", {"vp_to_win": 3}), "the game is over")
    _err(base, lambda s: s["players"][1].__setitem__("known_hand_min", [50, 0, 0, 0, 0]), "known_hand_min")
    _err(base, lambda s: s.__setitem__("longest_road_owner", 3), "longest_road_owner")
    _err(base, lambda s: s["vertex_owner"].pop(), "54 integers")
    assert validate_position("nope") == ["position must be an object"]
    _err(base, lambda s: s.__setitem__("actor", (s["actor"] + 1) % 4), "disagrees")
    assert base["actor"] == 0
