import pytest

from catan.engine import A, Action, Phase, TOPOLOGY as T, COST, IllegalAction

from .helpers import main_game, put_settlement, put_road, give, vertex_path, check_conservation


def test_costs():
    assert COST == {"road": (1, 1, 0, 0, 0), "settlement": (1, 1, 1, 1, 0),
                    "city": (0, 0, 0, 2, 3), "dev": (0, 0, 1, 1, 1)}


def test_build_road_pays_and_requires_resources():
    g = main_game()
    vs, es = vertex_path(3)
    put_settlement(g, 0, vs[0])
    assert not g.is_legal(Action(0, A.BUILD_ROAD, es[0]))
    give(g, 0, COST["road"])
    assert g.is_legal(Action(0, A.BUILD_ROAD, es[0]))
    g.step(Action(0, A.BUILD_ROAD, es[0]))
    assert g.players[0].hand == [0] * 5 and g.edge_owner[es[0]] == 0
    check_conservation(g)


def test_road_connectivity():
    g = main_game()
    vs, es = vertex_path(4)
    put_settlement(g, 0, vs[0])
    give(g, 0, (5, 5, 0, 0, 0))
    legal = {a.value for a in g.legal_actions() if a.type == A.BUILD_ROAD}
    assert legal == set(T.vertex_edges[vs[0]])
    g.step(Action(0, A.BUILD_ROAD, es[0]))
    assert not g.is_legal(Action(0, A.BUILD_ROAD, es[2]))  # not adjacent to network
    g.step(Action(0, A.BUILD_ROAD, es[1]))  # extends from road end (no building)
    assert g.is_legal(Action(0, A.BUILD_ROAD, es[2]))


def test_road_blocked_through_opponent_settlement():
    g = main_game()
    vs, es = vertex_path(3)
    put_settlement(g, 0, vs[0])
    put_road(g, 0, es[0])
    put_settlement(g, 1, vs[1])
    give(g, 0, (2, 2, 0, 0, 0))
    assert not g.is_legal(Action(0, A.BUILD_ROAD, es[1]))
    legal = {a.value for a in g.legal_actions() if a.type == A.BUILD_ROAD}
    assert not legal & set(T.vertex_edges[vs[1]]) - {es[0]}


def test_cannot_build_on_occupied_edge():
    g = main_game()
    vs, es = vertex_path(2)
    put_settlement(g, 0, vs[0])
    put_road(g, 1, es[0])
    give(g, 0, COST["road"])
    assert not g.is_legal(Action(0, A.BUILD_ROAD, es[0]))


def test_settlement_needs_own_road_and_distance_rule():
    g = main_game()
    vs, es = vertex_path(4)
    put_settlement(g, 0, vs[0])
    put_road(g, 0, es[0])
    give(g, 0, (2, 2, 2, 2, 0))
    assert not g.is_legal(Action(0, A.BUILD_SETTLEMENT, vs[1]))  # adjacent to own settlement
    assert not g.is_legal(Action(0, A.BUILD_SETTLEMENT, vs[2]))  # no road there
    put_road(g, 0, es[1])
    assert g.is_legal(Action(0, A.BUILD_SETTLEMENT, vs[2]))
    put_settlement(g, 1, vs[3])
    assert not g.is_legal(Action(0, A.BUILD_SETTLEMENT, vs[2]))  # adjacent to opponent
    # opponent's road does not allow building
    g2 = main_game()
    put_road(g2, 1, es[0])
    give(g2, 0, (1, 1, 1, 1, 0))
    assert not g2.is_legal(Action(0, A.BUILD_SETTLEMENT, vs[0]))


def test_settlement_build_pays_and_gains_port():
    g = main_game()
    pe = g.layout.port_edges[0]
    v = T.edge_vertices[pe][0]
    e = T.vertex_edges[v][0]
    put_road(g, 0, e)
    give(g, 0, COST["settlement"])
    g.step(Action(0, A.BUILD_SETTLEMENT, v))
    assert g.players[0].hand == [0] * 5
    assert g.layout.port_types[0] in g.players[0].ports
    assert g.vertex_owner[v] == 0 and g.vertex_level[v] == 1


def test_city_upgrade_and_supply_returns_settlement():
    g = main_game()
    put_settlement(g, 0, 0)
    assert g.supply(0, "settlement") == 4
    give(g, 0, COST["city"])
    assert not g.is_legal(Action(0, A.BUILD_CITY, 5))  # not own settlement
    g.step(Action(0, A.BUILD_CITY, 0))
    assert g.vertex_level[0] == 2 and g.players[0].cities == [0] and g.players[0].settlements == []
    assert g.supply(0, "settlement") == 5 and g.supply(0, "city") == 3
    give(g, 0, COST["city"])
    assert not g.is_legal(Action(0, A.BUILD_CITY, 0))  # already a city
    check_conservation(g)


def test_supply_limits():
    g = main_game()
    assert (g.supply(0, "road"), g.supply(0, "settlement"), g.supply(0, "city")) == (15, 5, 4)
    # 15 roads used -> no road actions even with resources
    vs, es = vertex_path(16)
    put_settlement(g, 0, vs[0])
    for e in es[:15]:
        put_road(g, 0, e)
    give(g, 0, (3, 3, 1, 1, 0))
    assert g.supply(0, "road") == 0
    assert not g.is_legal(Action(0, A.BUILD_ROAD, es[15]))
    assert not any(a.type == A.BUILD_ROAD for a in g.legal_actions())
    # 5 settlements used -> no settlement
    for v in (vs[2], vs[4], vs[6], vs[8]):
        put_settlement(g, 0, v)
    assert g.supply(0, "settlement") == 0
    assert not g.is_legal(Action(0, A.BUILD_SETTLEMENT, vs[10]))
    # 4 cities used -> no city
    for v in (vs[0], vs[2], vs[4], vs[6]):
        g.players[0].settlements.remove(v)
        g.players[0].cities.append(v)
        g.vertex_level[v] = 2
    g.players[0].hand = [0] * 5
    g.bank = [19] * 5
    give(g, 0, COST["city"])
    assert g.supply(0, "city") == 0
    assert not g.is_legal(Action(0, A.BUILD_CITY, vs[8]))


def test_build_only_in_main():
    g = main_game()
    g.phase = Phase.PRE_ROLL
    g.rolled = False
    put_settlement(g, 0, 0)
    give(g, 0, (5, 5, 5, 5, 5))
    assert not g.is_legal(Action(0, A.BUILD_ROAD, T.vertex_edges[0][0]))
    assert not g.is_legal(Action(0, A.BUILD_CITY, 0))
    assert not g.is_legal(Action(0, A.BUY_DEV))
    assert not g.is_legal(Action(0, A.END_TURN))
    with pytest.raises(IllegalAction):
        g.step(Action(0, A.END_TURN))
