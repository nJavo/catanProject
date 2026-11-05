import pytest

from catan.engine import Game, GameConfig, Phase, A, Action, TOPOLOGY as T
from catan.engine.board import RESOURCES, DESERT


def _play_setup(g, pick=lambda acts: acts[0]):
    order = []
    while g.phase in (Phase.SETUP_SETTLEMENT, Phase.SETUP_ROAD):
        if g.phase == Phase.SETUP_SETTLEMENT:
            order.append(g.actor)
        g.step(pick(g.legal_actions()))
    return order


@pytest.mark.parametrize("n", [2, 3, 4])
def test_snake_order(n):
    g = Game(config=GameConfig(num_players=n), seed=1)
    assert _play_setup(g) == list(range(n)) + list(reversed(range(n)))
    assert g.phase == Phase.PRE_ROLL and g.actor == 0 and g.turn == 0
    for p in range(n):
        assert len(g.players[p].settlements) == 2 and len(g.players[p].roads) == 2


def test_setup_only_settlement_actions_then_road():
    g = Game(seed=2)
    acts = g.legal_actions()
    assert acts and all(a.type == A.BUILD_SETTLEMENT and a.player == 0 for a in acts)
    assert len(acts) == 54
    g.step(acts[0])
    assert g.phase == Phase.SETUP_ROAD and g.actor == 0


def test_setup_distance_rule():
    g = Game(seed=3)
    v = 20
    g.step(Action(0, A.BUILD_SETTLEMENT, v))
    g.step(g.legal_actions()[0])
    legal = {a.value for a in g.legal_actions()}
    assert v not in legal
    for u in T.vertex_neighbors[v]:
        assert u not in legal
        assert not g.is_legal(Action(1, A.BUILD_SETTLEMENT, u))
    assert len(legal) == 54 - 1 - len(T.vertex_neighbors[v])


def test_setup_road_must_touch_new_settlement():
    g = Game(seed=4)
    g.step(Action(0, A.BUILD_SETTLEMENT, 20))
    legal = {a.value for a in g.legal_actions()}
    assert legal == set(T.vertex_edges[20])
    far = next(e for e in range(72) if 20 not in T.edge_vertices[e])
    assert not g.is_legal(Action(0, A.BUILD_ROAD, far))
    # second round: road must touch the second settlement, not the first
    _play_setup_until(g, lambda: g.setup_index == 7 and g.phase == Phase.SETUP_SETTLEMENT)
    v2 = next(a.value for a in g.legal_actions())
    g.step(Action(0, A.BUILD_SETTLEMENT, v2))
    legal = {a.value for a in g.legal_actions()}
    assert legal == {e for e in T.vertex_edges[v2] if g.edge_owner[e] == -1}
    assert not any(20 in T.edge_vertices[e] for e in legal)


def _play_setup_until(g, cond):
    while not cond():
        g.step(g.legal_actions()[0])


def test_second_settlement_starting_resources():
    g = Game(seed=5)
    while g.phase in (Phase.SETUP_SETTLEMENT, Phase.SETUP_ROAD):
        p = g.actor
        if g.phase == Phase.SETUP_SETTLEMENT:
            before = g.players[p].hand[:]
            # choose a 3-hex vertex for a non-trivial check
            acts = g.legal_actions()
            a = max(acts, key=lambda a: len(T.vertex_hexes[a.value]))
            g.step(a)
            gained = [x - y for x, y in zip(g.players[p].hand, before)]
            if g.setup_index < g.n:
                assert gained == [0] * 5
            else:
                exp = [0] * 5
                for h in T.vertex_hexes[a.value]:
                    if g.layout.hex_terrain[h] != DESERT:
                        exp[RESOURCES.index(g.layout.hex_terrain[h])] += 1
                assert gained == exp
        else:
            g.step(g.legal_actions()[0])
    for r in range(5):
        assert g.bank[r] + sum(pl.hand[r] for pl in g.players) == 19


def test_no_main_actions_during_setup():
    g = Game(seed=6)
    for t in (A.ROLL, A.END_TURN, A.BUY_DEV):
        assert not g.is_legal(Action(0, t))
    assert not g.is_legal(Action(0, A.BUILD_ROAD, 0))
