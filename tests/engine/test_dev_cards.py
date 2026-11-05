from collections import Counter

import pytest

from catan.engine import A, Action, Phase, TOPOLOGY as T, COST, Game
from catan.engine.game import KNIGHT, VICTORY_POINT, ROAD_BUILDING, YEAR_OF_PLENTY, MONOPOLY

from .helpers import main_game, pre_roll_game, put_settlement, put_road, give, vertex_path, check_conservation


def test_deck_composition():
    for seed in range(5):
        g = Game(seed=seed)
        assert Counter(g.dev_deck) == {KNIGHT: 14, VICTORY_POINT: 5, ROAD_BUILDING: 2,
                                       YEAR_OF_PLENTY: 2, MONOPOLY: 2}
    assert Game(seed=1).dev_deck != Game(seed=2).dev_deck


def test_buy_dev_pays_draws_and_hides():
    g = main_game()
    give(g, 0, COST["dev"])
    top = g.dev_deck[-1]
    ev = g.step(Action(0, A.BUY_DEV))
    assert g.players[0].dev[top] == 1 and g.players[0].dev_new[top] == 1
    assert len(g.dev_deck) == 24 and g.players[0].hand == [0] * 5
    pub = [e for e in ev if "visible_to" not in e]
    assert all("card" not in e for e in pub)
    assert any(e.get("visible_to") == [0] for e in ev)
    check_conservation(g)


def test_cannot_buy_from_empty_deck():
    g = main_game()
    g.dev_deck = []
    give(g, 0, COST["dev"])
    assert not g.is_legal(Action(0, A.BUY_DEV))


def test_cannot_play_card_bought_this_turn():
    g = main_game()
    g.dev_deck.append(KNIGHT)
    give(g, 0, COST["dev"])
    g.step(Action(0, A.BUY_DEV))
    assert not g.is_legal(Action(0, A.PLAY_KNIGHT))
    g.step(Action(0, A.END_TURN))
    assert g.players[0].dev_new == [0] * 5
    # next own turn: playable (even before rolling)
    for _ in range(3):
        g.phase, g.rolled = Phase.MAIN, True
        g.step(Action(g.current, A.END_TURN))
    assert g.current == 0 and g.phase == Phase.PRE_ROLL
    assert g.is_legal(Action(0, A.PLAY_KNIGHT))


def test_old_copy_playable_when_new_copy_bought():
    g = main_game()
    g.players[0].dev[KNIGHT] = 1
    g.dev_deck.append(KNIGHT)
    give(g, 0, COST["dev"])
    g.step(Action(0, A.BUY_DEV))
    assert g.is_legal(Action(0, A.PLAY_KNIGHT))


def test_one_dev_card_per_turn():
    g = main_game()
    g.players[0].dev[KNIGHT] = 2
    g.players[0].dev[MONOPOLY] = 1
    g.step(Action(0, A.PLAY_KNIGHT))
    g.step(Action(0, A.MOVE_ROBBER, next(h for h in range(19) if h != g.robber)))
    assert g.phase == Phase.MAIN
    assert not g.is_legal(Action(0, A.PLAY_KNIGHT))
    assert not g.is_legal(Action(0, A.PLAY_MONOPOLY, 0))


def test_vp_cards_not_playable():
    g = main_game()
    g.players[0].dev[VICTORY_POINT] = 1
    assert not any(a.type.startswith("PLAY_") for a in g.legal_actions())
    assert g.victory_points(0) == 1 and g.public_victory_points(0) == 0


def test_knight_pre_roll_returns_to_pre_roll():
    g = pre_roll_game()
    g.players[0].dev[KNIGHT] = 1
    assert g.is_legal(Action(0, A.PLAY_KNIGHT))
    g.step(Action(0, A.PLAY_KNIGHT))
    assert g.phase == Phase.MOVE_ROBBER and g.players[0].knights == 1
    g.step(Action(0, A.MOVE_ROBBER, next(h for h in range(19) if h != g.robber)))
    assert g.phase == Phase.PRE_ROLL
    assert {a.type for a in g.legal_actions()} == {A.ROLL}


def test_knight_main_with_steal_returns_to_main():
    g = main_game()
    g.players[0].dev[KNIGHT] = 1
    h = next(h for h in range(19) if h != g.robber)
    put_settlement(g, 1, T.hex_vertices[h][0])
    put_settlement(g, 2, T.hex_vertices[h][3])
    give(g, 1, (1, 0, 0, 0, 0))
    give(g, 2, (0, 1, 0, 0, 0))
    g.step(Action(0, A.PLAY_KNIGHT))
    g.step(Action(0, A.MOVE_ROBBER, h))
    assert g.phase == Phase.STEAL
    g.step(Action(0, A.STEAL, 2))
    assert g.phase == Phase.MAIN and g.players[0].hand == [0, 1, 0, 0, 0]


@pytest.mark.parametrize("supply,expected", [(2, 2), (1, 1)])
def test_road_building_places_up_to_supply(supply, expected):
    g = main_game()
    vs, es = vertex_path(17)
    put_settlement(g, 0, vs[0])
    for e in es[: 15 - supply]:
        put_road(g, 0, e)
    g.players[0].dev[ROAD_BUILDING] = 1
    g.step(Action(0, A.PLAY_ROAD_BUILDING))
    assert g.phase == Phase.ROAD_BUILDING and g.free_roads == expected
    assert g.legal_actions() and all(a.type == A.BUILD_ROAD for a in g.legal_actions())
    for _ in range(expected):
        g.step(g.legal_actions()[0])
    assert g.phase == Phase.MAIN and g.free_roads == 0
    assert g.supply(0, "road") == 0
    assert g.players[0].hand == [0] * 5  # free


def test_road_building_with_no_roads_left_illegal():
    g = main_game()
    vs, es = vertex_path(16)
    put_settlement(g, 0, vs[0])
    for e in es[:15]:
        put_road(g, 0, e)
    g.players[0].dev[ROAD_BUILDING] = 1
    assert not g.is_legal(Action(0, A.PLAY_ROAD_BUILDING))


def test_road_building_stops_when_no_legal_edge():
    g = main_game()
    g.players[0].dev[ROAD_BUILDING] = 1
    assert not g.is_legal(Action(0, A.PLAY_ROAD_BUILDING))  # no network at all


def test_road_building_pre_roll_returns_to_pre_roll():
    g = pre_roll_game()
    put_settlement(g, 0, 0)
    g.players[0].dev[ROAD_BUILDING] = 1
    g.step(Action(0, A.PLAY_ROAD_BUILDING))
    g.step(g.legal_actions()[0])
    g.step(g.legal_actions()[0])
    assert g.phase == Phase.PRE_ROLL


def test_year_of_plenty_bank_limits():
    g = main_game()
    g.players[0].dev[YEAR_OF_PLENTY] = 1
    g.bank = [1, 0, 19, 19, 19]
    g.players[1].hand = [18, 19, 0, 0, 0]
    yop = {a.value for a in g.legal_actions() if a.type == A.PLAY_YEAR_OF_PLENTY}
    assert (0, 0) not in yop and (0, 1) not in yop and (1, 2) not in yop
    assert (0, 2) in yop and (2, 2) in yop
    assert not g.is_legal(Action(0, A.PLAY_YEAR_OF_PLENTY, (2, 0)))  # must be sorted
    g.step(Action(0, A.PLAY_YEAR_OF_PLENTY, (0, 2)))
    assert g.players[0].hand == [1, 0, 1, 0, 0] and g.bank[0] == 0
    check_conservation(g)


def test_monopoly_takes_all_of_resource():
    g = main_game()
    g.players[0].dev[MONOPOLY] = 1
    give(g, 1, (0, 0, 3, 1, 0))
    give(g, 2, (0, 0, 2, 0, 0))
    give(g, 3, (1, 0, 0, 0, 0))
    ev = g.step(Action(0, A.PLAY_MONOPOLY, 2))
    assert g.players[0].hand[2] == 5
    assert g.players[1].hand == [0, 0, 0, 1, 0] and g.players[2].hand == [0] * 5
    assert g.players[3].hand == [1, 0, 0, 0, 0]
    assert any(e["type"] == "monopoly" and e["amount"] == 5 for e in ev)
    check_conservation(g)
    assert not g.is_legal(Action(0, A.PLAY_MONOPOLY, 5))


def test_largest_army_transfer_strictly_more():
    g = main_game()
    g.players[0].dev[KNIGHT] = 4
    g.players[1].dev[KNIGHT] = 4

    def knight(p):
        g.current = p
        g.phase, g.rolled, g.dev_played = Phase.MAIN, True, False
        ev = g.step(Action(p, A.PLAY_KNIGHT))
        g.step(Action(p, A.MOVE_ROBBER, next(h for h in range(19) if h != g.robber)))
        return ev

    knight(0); knight(0)
    assert g.largest_army_owner == -1
    ev = knight(0)
    assert g.largest_army_owner == 0
    assert {"type": "award_changed", "award": "largest_army", "from": -1, "to": 0} in ev
    knight(1); knight(1); knight(1)
    assert g.largest_army_owner == 0  # 3 vs 3 -> holder keeps
    knight(1)
    assert g.largest_army_owner == 1
    assert g.public_victory_points(1) == 2 and g.public_victory_points(0) == 0
