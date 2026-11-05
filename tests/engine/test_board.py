import random
from collections import Counter

import pytest

from catan.engine import TOPOLOGY as T, Game, generate_layout
from catan.engine.board import TOKEN_SEQUENCE, DESERT, BoardLayout


def test_topology_counts():
    assert (T.num_hexes, T.num_vertices, T.num_edges) == (19, 54, 72)
    assert len(T.coast_edges) == 30
    assert len(set(T.coast_edges)) == 30


def test_vertex_hex_and_neighbor_counts():
    hex_counts = Counter(len(T.vertex_hexes[v]) for v in range(54))
    assert set(hex_counts) <= {1, 2, 3}
    # 24 interior vertices touch 3 hexes; 30 coastal vertices touch 1 or 2.
    assert hex_counts[3] == 24 and hex_counts[1] + hex_counts[2] == 30
    for v in range(54):
        assert len(T.vertex_neighbors[v]) in (2, 3)
        assert len(T.vertex_edges[v]) == len(T.vertex_neighbors[v])
    assert all(len(set(hv)) == 6 for hv in T.hex_vertices)


def test_adjacency_symmetric_and_consistent():
    for e, (a, b) in enumerate(T.edge_vertices):
        assert a != b
        assert b in T.vertex_neighbors[a] and a in T.vertex_neighbors[b]
        assert e in T.vertex_edges[a] and e in T.vertex_edges[b]
        assert T.edge_lookup[frozenset((a, b))] == e
    for h, hv in enumerate(T.hex_vertices):
        for i in range(6):
            assert frozenset((hv[i], hv[(i + 1) % 6])) in T.edge_lookup
            assert h in T.vertex_hexes[hv[i]]


def test_coast_ring_is_a_cycle():
    r = T.coast_edges
    for i in range(30):
        assert set(T.edge_vertices[r[i]]) & set(T.edge_vertices[r[(i + 1) % 30]])
    coast_vertices = {v for e in r for v in T.edge_vertices[e]}
    assert len(coast_vertices) == 30
    assert all(len(T.vertex_hexes[v]) < 3 for v in coast_vertices)


@pytest.mark.parametrize("seed", range(30))
def test_layout_distribution(seed):
    lay = generate_layout(random.Random(seed))
    assert Counter(lay.hex_terrain) == {"wood": 4, "brick": 3, "sheep": 4, "wheat": 4, "ore": 3, DESERT: 1}
    assert lay.hex_terrain[lay.desert_hex] == DESERT
    assert lay.hex_number[lay.desert_hex] is None
    nums = [n for n in lay.hex_number if n is not None]
    assert sorted(nums) == sorted(TOKEN_SEQUENCE)
    assert len(nums) == 18 and 7 not in nums
    assert Counter(lay.port_types) == {"3:1": 4, "wood": 1, "brick": 1, "sheep": 1, "wheat": 1, "ore": 1}
    assert len(lay.port_edges) == 9 and set(lay.port_edges) <= set(T.coast_edges)
    assert len(lay.vertex_ports()) == 18
    assert BoardLayout.from_dict(lay.to_dict()) == lay


def test_tokens_follow_spiral_skipping_desert():
    for seed in range(20):
        lay = generate_layout(random.Random(seed))
        orders = [o for o in T.spiral_orders
                  if [lay.hex_number[h] for h in o if lay.hex_terrain[h] != DESERT] == list(TOKEN_SEQUENCE)]
        assert orders, seed


def test_spiral_orders_cover_all_hexes():
    for o in T.spiral_orders:
        assert sorted(o) == list(range(19))


def test_robber_starts_on_desert():
    for seed in range(10):
        g = Game(seed=seed)
        assert g.robber == g.layout.desert_hex


def test_layout_seeded():
    assert Game(seed=5).layout == Game(seed=5).layout
    assert any(Game(seed=5).layout != Game(seed=s).layout for s in range(6, 10))
