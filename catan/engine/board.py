"""Board topology and layout generation for the standard 19-hex base game.

Topology (hexes, vertices, edges, adjacency) is computed geometrically once and
is identical for every game. Layout (terrain, number tokens, harbor types) is
generated per game from a seeded RNG.

Coordinates: pointy-top hexes, axial (q, r), unit hex radius.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

# Resource indices are used everywhere (hands, bank, costs).
WOOD, BRICK, SHEEP, WHEAT, ORE = range(5)
RESOURCES = ("wood", "brick", "sheep", "wheat", "ore")
DESERT = "desert"
GENERIC_PORT = "3:1"

# Base game terrain distribution (by resource produced).
TERRAIN_COUNTS = {"wood": 4, "brick": 3, "sheep": 4, "wheat": 4, "ore": 3, DESERT: 1}
# Official number tokens A..R, placed in spiral order skipping the desert.
TOKEN_SEQUENCE = (5, 2, 6, 3, 8, 10, 9, 12, 11, 4, 8, 10, 9, 4, 5, 6, 3, 11)
# 9 harbors: 4 generic 3:1 + one 2:1 per resource.
PORT_TYPES = (GENERIC_PORT,) * 4 + RESOURCES
# Harbor positions around the 30 coastal edges: gaps 3,3,4 repeated (fixed frame).
PORT_EDGE_OFFSETS = (0, 3, 6, 10, 13, 16, 20, 23, 26)

AXIAL_DIRS = ((1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1))
SQRT3 = math.sqrt(3)


def _ring(radius: int, start_dir: int) -> list[tuple[int, int]]:
    if radius == 0:
        return [(0, 0)]
    q, r = AXIAL_DIRS[start_dir][0] * radius, AXIAL_DIRS[start_dir][1] * radius
    out = []
    for i in range(6):
        dq, dr = AXIAL_DIRS[(start_dir + 2 + i) % 6]
        for _ in range(radius):
            out.append((q, r))
            q, r = q + dq, r + dr
    return out


def hex_center(q: int, r: int) -> tuple[float, float]:
    return (SQRT3 * (q + r / 2), 1.5 * r)


@dataclass(frozen=True)
class Topology:
    hex_coords: tuple            # hex id -> (q, r)
    hex_xy: tuple                # hex id -> (x, y)
    vertex_xy: tuple             # vertex id -> (x, y)
    edge_vertices: tuple         # edge id -> (v1, v2)
    hex_vertices: tuple          # hex id -> 6 vertex ids (clockwise from top-right)
    vertex_hexes: tuple          # vertex id -> hex ids
    vertex_neighbors: tuple      # vertex id -> adjacent vertex ids
    vertex_edges: tuple          # vertex id -> incident edge ids
    edge_lookup: dict = field(repr=False)  # frozenset(v1,v2) -> edge id
    coast_edges: tuple           # coastal edge ids ordered by angle around center
    spiral_orders: tuple         # 6 spiral hex orders (one per starting corner)

    @property
    def num_hexes(self):
        return len(self.hex_coords)

    @property
    def num_vertices(self):
        return len(self.vertex_xy)

    @property
    def num_edges(self):
        return len(self.edge_vertices)


def build_topology() -> Topology:
    coords = sorted(
        [(q, r) for q in range(-2, 3) for r in range(-2, 3) if abs(q + r) <= 2],
        key=lambda c: (c[1], c[0]),
    )
    hex_xy = [hex_center(q, r) for q, r in coords]

    key_to_v: dict = {}
    raw_hex_corners = []
    for x, y in hex_xy:
        corners = []
        for i in range(6):
            a = math.radians(60 * i - 30)
            k = (round(x + math.cos(a), 3), round(y + math.sin(a), 3))
            corners.append(k)
            key_to_v.setdefault(k, None)
        raw_hex_corners.append(corners)
    vkeys = sorted(key_to_v, key=lambda k: (k[1], k[0]))
    vid = {k: i for i, k in enumerate(vkeys)}
    hex_vertices = tuple(tuple(vid[k] for k in corners) for corners in raw_hex_corners)

    edge_set = set()
    for hv in hex_vertices:
        for i in range(6):
            edge_set.add(frozenset((hv[i], hv[(i + 1) % 6])))

    def mid(e):
        a, b = tuple(e)
        return ((vkeys[a][0] + vkeys[b][0]) / 2, (vkeys[a][1] + vkeys[b][1]) / 2)

    edges = sorted(edge_set, key=lambda e: (round(mid(e)[1], 3), round(mid(e)[0], 3)))
    edge_vertices = tuple(tuple(sorted(e)) for e in edges)
    edge_lookup = {frozenset(e): i for i, e in enumerate(edge_vertices)}

    nv = len(vkeys)
    vertex_hexes = [[] for _ in range(nv)]
    for h, hv in enumerate(hex_vertices):
        for v in hv:
            vertex_hexes[v].append(h)
    vertex_neighbors = [[] for _ in range(nv)]
    vertex_edges = [[] for _ in range(nv)]
    for e, (a, b) in enumerate(edge_vertices):
        vertex_neighbors[a].append(b)
        vertex_neighbors[b].append(a)
        vertex_edges[a].append(e)
        vertex_edges[b].append(e)

    edge_hex_count = {e: 0 for e in range(len(edge_vertices))}
    for hv in hex_vertices:
        for i in range(6):
            edge_hex_count[edge_lookup[frozenset((hv[i], hv[(i + 1) % 6]))]] += 1
    coast = [e for e, c in edge_hex_count.items() if c == 1]
    coast.sort(key=lambda e: math.atan2(mid(edge_vertices[e])[1], mid(edge_vertices[e])[0]))

    cindex = {c: i for i, c in enumerate(coords)}
    spirals = tuple(
        tuple(cindex[c] for c in _ring(2, d) + _ring(1, d) + _ring(0, d)) for d in range(6)
    )

    return Topology(
        hex_coords=tuple(coords),
        hex_xy=tuple(hex_xy),
        vertex_xy=tuple(vkeys),
        edge_vertices=edge_vertices,
        hex_vertices=hex_vertices,
        vertex_hexes=tuple(tuple(x) for x in vertex_hexes),
        vertex_neighbors=tuple(tuple(sorted(x)) for x in vertex_neighbors),
        vertex_edges=tuple(tuple(sorted(x)) for x in vertex_edges),
        edge_lookup=edge_lookup,
        coast_edges=tuple(coast),
        spiral_orders=spirals,
    )


TOPOLOGY = build_topology()


@dataclass(frozen=True)
class BoardLayout:
    hex_terrain: tuple        # hex id -> resource name or "desert"
    hex_number: tuple         # hex id -> int or None
    port_edges: tuple         # 9 coastal edge ids holding harbors
    port_types: tuple         # parallel to port_edges: "3:1" or resource name
    desert_hex: int

    def vertex_ports(self) -> dict:
        """vertex id -> port type for harbor-touching vertices."""
        out = {}
        for e, t in zip(self.port_edges, self.port_types):
            for v in TOPOLOGY.edge_vertices[e]:
                out[v] = t
        return out

    def to_dict(self):
        return {
            "hex_terrain": list(self.hex_terrain),
            "hex_number": list(self.hex_number),
            "port_edges": list(self.port_edges),
            "port_types": list(self.port_types),
        }

    @staticmethod
    def from_dict(d):
        terr = tuple(d["hex_terrain"])
        return BoardLayout(terr, tuple(d["hex_number"]), tuple(d["port_edges"]),
                           tuple(d["port_types"]), terr.index(DESERT))


# Official "Starting Set-up for Beginners" terrain/numbers are not reproduced here;
# the default is the official variable set-up (random terrain, spiral tokens).

def generate_layout(rng: random.Random, shuffle_ports: bool = True) -> BoardLayout:
    """Official variable set-up: random terrain, spiral token placement from a
    random corner skipping the desert, harbor pieces randomly placed on the fixed
    frame harbor positions."""
    terrain = [t for t, n in TERRAIN_COUNTS.items() for _ in range(n)]
    rng.shuffle(terrain)
    order = TOPOLOGY.spiral_orders[rng.randrange(6)]
    numbers = [None] * TOPOLOGY.num_hexes
    tokens = iter(TOKEN_SEQUENCE)
    for h in order:
        if terrain[h] != DESERT:
            numbers[h] = next(tokens)
    ports = list(PORT_TYPES)
    if shuffle_ports:
        rng.shuffle(ports)
    port_edges = tuple(TOPOLOGY.coast_edges[o] for o in PORT_EDGE_OFFSETS)
    return BoardLayout(tuple(terrain), tuple(numbers), port_edges, tuple(ports),
                       terrain.index(DESERT))


def topology_dict() -> dict:
    """JSON-friendly topology for renderers."""
    t = TOPOLOGY
    return {
        "hexes": [{"id": i, "q": q, "r": r, "x": t.hex_xy[i][0], "y": t.hex_xy[i][1],
                   "vertices": list(t.hex_vertices[i])} for i, (q, r) in enumerate(t.hex_coords)],
        "vertices": [{"id": i, "x": x, "y": y, "hexes": list(t.vertex_hexes[i]),
                      "neighbors": list(t.vertex_neighbors[i])} for i, (x, y) in enumerate(t.vertex_xy)],
        "edges": [{"id": i, "v": list(vs)} for i, vs in enumerate(t.edge_vertices)],
    }
