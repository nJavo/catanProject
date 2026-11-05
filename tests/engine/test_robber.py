import random

from catan.engine import A, Action, Phase, TOPOLOGY as T

from .helpers import pre_roll_game, main_game, put_settlement, give, check_conservation


def _roll_seven(g):
    """Force the next roll to be 7 by searching an RNG seed."""
    for s in range(1000):
        r = random.Random(s)
        if r.randint(1, 6) + r.randint(1, 6) == 7:
            g.rng = random.Random(s)
            return g.step(Action(g.current, A.ROLL))
    raise AssertionError


def test_seven_discard_only_over_limit():
    g = pre_roll_game()
    give(g, 1, (8, 0, 0, 0, 0))   # 8 -> discard 4
    give(g, 2, (3, 4, 0, 0, 0))   # 7 -> none
    give(g, 3, (2, 2, 2, 2, 1))   # 9 -> discard 4
    give(g, 0, (0, 0, 5, 5, 5))   # 15 -> discard 7
    ev = _roll_seven(g)
    assert g.phase == Phase.DISCARD
    assert g.discard_remaining == [7, 4, 0, 4]
    assert {"type": "discard_required", "amounts": [7, 4, 0, 4]} in ev
    # roller discards first, then seat order
    order = []
    while g.phase == Phase.DISCARD:
        p = g.actor
        acts = g.legal_actions()
        assert acts and all(a.type == A.DISCARD and a.player == p for a in acts)
        assert {a.value for a in acts} == {r for r in range(5) if g.players[p].hand[r] > 0}
        order.append(p)
        g.step(acts[0])
    assert order == [0] * 7 + [1] * 4 + [3] * 4
    assert [sum(pl.hand) for pl in g.players] == [8, 4, 7, 5]
    assert g.phase == Phase.MOVE_ROBBER
    check_conservation(g)


def test_discard_order_from_roller():
    g = pre_roll_game()
    g.current = 2
    give(g, 0, (8, 0, 0, 0, 0))
    give(g, 3, (0, 8, 0, 0, 0))
    _roll_seven(g)
    assert g.actor == 3
    for _ in range(4):
        g.step(g.legal_actions()[0])
    assert g.actor == 0


def test_cannot_discard_absent_resource_or_wrong_player():
    g = pre_roll_game()
    give(g, 1, (8, 0, 0, 0, 0))
    _roll_seven(g)
    assert not g.is_legal(Action(1, A.DISCARD, 1))
    assert not g.is_legal(Action(0, A.DISCARD, 0))
    assert not g.is_legal(Action(1, A.END_TURN))


def test_robber_must_move_and_no_steal_when_no_victims():
    g = pre_roll_game()
    _roll_seven(g)
    assert g.phase == Phase.MOVE_ROBBER
    legal = {a.value for a in g.legal_actions()}
    assert legal == set(range(19)) - {g.robber}
    assert not g.is_legal(Action(0, A.MOVE_ROBBER, g.robber))
    h = next(iter(legal))
    g.step(Action(0, A.MOVE_ROBBER, h))
    assert g.robber == h and g.phase == Phase.MAIN


def test_auto_steal_single_victim():
    g = main_game()
    h = next(h for h in range(19) if h != g.robber)
    put_settlement(g, 1, T.hex_vertices[h][0])
    put_settlement(g, 0, T.hex_vertices[h][3])      # own building: not a victim
    put_settlement(g, 2, next(v for v in range(54) if h not in T.vertex_hexes[v] and g.vertex_owner[v] == -1))
    give(g, 1, (0, 0, 0, 2, 0))
    g.phase = Phase.MOVE_ROBBER
    ev = g.step(Action(0, A.MOVE_ROBBER, h))
    assert g.phase == Phase.MAIN
    assert g.players[0].hand == [0, 0, 0, 1, 0] and g.players[1].hand == [0, 0, 0, 1, 0]
    pub = next(e for e in ev if e["type"] == "stolen")
    assert "resource" not in pub
    det = next(e for e in ev if e["type"] == "stolen_detail")
    assert det["visible_to"] == [0, 1] and det["resource"] == 3


def test_victims_need_cards_and_adjacency():
    g = main_game()
    h = next(h for h in range(19) if h != g.robber)
    hv = T.hex_vertices[h]
    put_settlement(g, 1, hv[0])
    put_settlement(g, 2, hv[2])
    put_settlement(g, 3, hv[4])
    give(g, 1, (1, 0, 0, 0, 0))
    give(g, 3, (0, 1, 0, 0, 0))   # p2 has no cards
    g.phase = Phase.MOVE_ROBBER
    g.step(Action(0, A.MOVE_ROBBER, h))
    assert g.phase == Phase.STEAL
    assert {a.value for a in g.legal_actions()} == {1, 3}
    assert not g.is_legal(Action(0, A.STEAL, 2))
    assert not g.is_legal(Action(0, A.STEAL, 0))
    g.step(Action(0, A.STEAL, 3))
    assert g.players[0].hand == [0, 1, 0, 0, 0] and g.phase == Phase.MAIN


def test_moving_robber_to_hex_with_only_empty_handed_opponent():
    g = main_game()
    h = next(h for h in range(19) if h != g.robber)
    put_settlement(g, 1, T.hex_vertices[h][0])
    g.phase = Phase.MOVE_ROBBER
    ev = g.step(Action(0, A.MOVE_ROBBER, h))
    assert g.phase == Phase.MAIN and not any(e["type"] == "stolen" for e in ev)
