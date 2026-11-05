"""Helpers to build engine positions directly (bypassing setup)."""
from __future__ import annotations

from catan.engine import Game, GameConfig, Phase, TOPOLOGY as T
from catan.engine.board import RESOURCES, DESERT


def main_game(n=4, seed=0, **cfg) -> Game:
    """Empty board, player 0 to act in MAIN (already rolled)."""
    g = Game(config=GameConfig(num_players=n, **cfg), seed=seed)
    g.setup_index = len(g.setup_order)
    g.phase = Phase.MAIN
    g.current = 0
    g.rolled = True
    return g


def pre_roll_game(n=4, seed=0, **cfg) -> Game:
    g = main_game(n, seed, **cfg)
    g.phase = Phase.PRE_ROLL
    g.rolled = False
    return g


def put_settlement(g, p, v, city=False):
    g._place_settlement(p, v)
    if city:
        g.players[p].settlements.remove(v)
        g.players[p].cities.append(v)
        g.vertex_level[v] = 2


def put_road(g, p, e):
    assert g.edge_owner[e] == -1
    g.edge_owner[e] = p
    g.players[p].roads.append(e)


def give(g, p, res):
    """Move resources (5-vector) from bank to player p."""
    for i, c in enumerate(res):
        assert g.bank[i] >= c
        g.bank[i] -= c
        g.players[p].hand[i] += c


def hand(*counts):
    return list(counts) + [0] * (5 - len(counts))


def edge(a, b):
    return T.edge_lookup[frozenset((a, b))]


def vertex_path(k, start=None):
    """Simple path of k edges as (vertices[k+1], edges[k]) along the coast ring."""
    ring = T.coast_edges
    i = 0 if start is None else start
    es = [ring[(i + j) % len(ring)] for j in range(k)]
    vs = []
    a, b = T.edge_vertices[es[0]]
    nxt = set(T.edge_vertices[es[1]]) if k > 1 else {b}
    first, second = (a, b) if b in nxt else (b, a)
    vs = [first, second]
    for e in es[1:]:
        x, y = T.edge_vertices[e]
        vs.append(y if x == vs[-1] else x)
    return vs, es


def resource_hex(g, resource=None, number=None):
    """A non-desert hex id (optionally matching resource name / number)."""
    for h in range(T.num_hexes):
        terr = g.layout.hex_terrain[h]
        if terr == DESERT:
            continue
        if resource is not None and terr != resource:
            continue
        if number is not None and g.layout.hex_number[h] != number:
            continue
        return h
    raise LookupError


def res_index(g, h):
    return RESOURCES.index(g.layout.hex_terrain[h])


def check_conservation(g):
    for r in range(5):
        assert g.bank[r] + sum(pl.hand[r] for pl in g.players) == 19
