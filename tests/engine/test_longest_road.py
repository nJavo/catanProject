from catan.engine import A, Action, TOPOLOGY as T, COST

from .helpers import main_game, put_settlement, put_road, give, vertex_path, edge


def _road(g, p, es):
    for e in es:
        put_road(g, p, e)


def _update(g):
    ev = []
    g._update_longest_road(ev)
    return ev


def test_length_simple_path_and_under_five():
    g = main_game()
    vs, es = vertex_path(4)
    _road(g, 0, es)
    assert g.longest_road(0) == 4
    _update(g)
    assert g.longest_road_owner == -1
    put_road(g, 0, vertex_path(5)[1][4])
    assert g.longest_road(0) == 5
    ev = _update(g)
    assert g.longest_road_owner == 0
    assert ev == [{"type": "award_changed", "award": "longest_road", "from": -1, "to": 0}]
    assert g.public_victory_points(0) == 2


def test_cycle_counts_each_edge_once():
    g = main_game()
    hv = T.hex_vertices[9]  # center hex
    ring = [edge(hv[i], hv[(i + 1) % 6]) for i in range(6)]
    _road(g, 0, ring)
    assert g.longest_road(0) == 6
    # tail off the ring: 6 + 1
    v = hv[0]
    tail = next(e for e in T.vertex_edges[v] if e not in ring)
    put_road(g, 0, tail)
    assert g.longest_road(0) == 7


def test_branches_take_longest_trail():
    g = main_game()
    vs, es = vertex_path(4)
    _road(g, 0, es)  # trail vs0..vs4
    i = next(i for i in (1, 2, 3) if len(T.vertex_edges[vs[i]]) == 3)
    b1 = next(e for e in T.vertex_edges[vs[i]] if e not in es)
    mid = next(x for x in T.edge_vertices[b1] if x != vs[i])
    b2 = next(e for e in T.vertex_edges[mid] if e != b1
              and not set(T.edge_vertices[e]) & set(vs))
    put_road(g, 0, b1)
    assert g.longest_road(0) == max(4, i + 1, 4 - i + 1)
    put_road(g, 0, b2)
    assert g.longest_road(0) == max(4, i + 2, 4 - i + 2)


def test_opponent_settlement_breaks_road():
    g = main_game()
    vs, es = vertex_path(6)
    _road(g, 0, es)
    assert g.longest_road(0) == 6
    put_settlement(g, 1, vs[3])
    assert g.longest_road(0) == 3
    # own settlement does not break it
    g2 = main_game()
    _road(g2, 0, es)
    put_settlement(g2, 0, vs[3])
    assert g2.longest_road(0) == 6
    # opponent at an end does not shorten
    g3 = main_game()
    _road(g3, 0, es)
    put_settlement(g3, 1, vs[0])
    assert g3.longest_road(0) == 6


def test_tie_keeps_holder_and_strictly_longer_takes():
    g = main_game()
    _road(g, 0, vertex_path(5, start=0)[1])
    _update(g)
    assert g.longest_road_owner == 0
    _road(g, 1, vertex_path(5, start=10)[1])
    _update(g)
    assert g.longest_road_owner == 0  # tie keeps holder
    put_road(g, 1, vertex_path(6, start=10)[1][5])
    _update(g)
    assert g.longest_road_owner == 1


def test_simultaneous_tie_without_holder_nobody():
    g = main_game()
    _road(g, 0, vertex_path(5, start=0)[1])
    _road(g, 1, vertex_path(5, start=10)[1])
    _update(g)
    assert g.longest_road_owner == -1


def test_broken_road_tie_sets_card_aside_via_action():
    g = main_game()
    vs, es = vertex_path(7, start=0)
    _road(g, 0, es)  # p0: 7
    _road(g, 1, vertex_path(5, start=12)[1])  # p1: 5
    _road(g, 2, vertex_path(5, start=20)[1])  # p2: 5
    _update(g)
    assert g.longest_road_owner == 0
    # p1 builds a settlement on vs[3] using an inland road -> p0 drops to 4
    inland = next(e for e in T.vertex_edges[vs[3]] if e not in es)
    put_road(g, 1, inland)
    g.current = 1
    give(g, 1, COST["settlement"])
    assert g.is_legal(Action(1, A.BUILD_SETTLEMENT, vs[3]))
    ev = g.step(Action(1, A.BUILD_SETTLEMENT, vs[3]))
    assert g.road_length[0] == 4
    assert g.longest_road_owner == -1  # p1/p2 tied at 5 -> card set aside
    assert {"type": "award_changed", "award": "longest_road", "from": 0, "to": -1} in ev


def test_broken_road_unique_new_leader_takes():
    g = main_game()
    vs, es = vertex_path(7, start=0)
    _road(g, 0, es)
    _road(g, 1, vertex_path(5, start=12)[1])
    _update(g)
    put_settlement(g, 2, vs[3])
    _update(g)
    assert g.longest_road_owner == 1


def test_broken_road_all_below_five_nobody():
    g = main_game()
    vs, es = vertex_path(6, start=0)
    _road(g, 0, es)
    _update(g)
    assert g.longest_road_owner == 0
    put_settlement(g, 1, vs[3])
    _update(g)
    assert g.longest_road_owner == -1


def test_broken_holder_still_tied_keeps():
    # doc rule: holder keeps the card on a tie
    g = main_game()
    vs, es = vertex_path(8, start=0)
    _road(g, 0, es)  # 8
    _road(g, 1, vertex_path(5, start=15)[1])
    _update(g)
    put_settlement(g, 2, vs[3])  # p0 -> max(3, 5) = 5, tied with p1
    _update(g)
    assert g.road_length[0] == 5
    assert g.longest_road_owner == 0
