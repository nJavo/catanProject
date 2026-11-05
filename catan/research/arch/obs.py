"""Observation v2: obs v1 (catan.rl.spaces) followed by derived planning features.

v1 is kept as an exact prefix, so ``obs[:V1_DIM]`` is the v1 observation and every
v1 model can be fed from v2 data. Extra blocks (all public information only:
board, buildings, roads, robber; no hands):

- vertex (NV x VX): robber-aware total pips; per relative seat, road distance
  (number of new roads, 0 = already on that seat's network, capped at 3)
- edge (NE x EX): per relative seat, road buildable now (connected, free)
- players (MAXP x PX): robber-aware production per resource and total, pips lost
  to the robber, free legal spots at road distance 0/1/2 and their best pips,
  road distance (capped: 1.0 means 4+ or unreachable) to the best free spot
  on the board and to the nearest free spot,
  longest-road gain from one more road, gap to the longest-road leader, ports
  reachable (owned or a free port spot within 2 roads) per port kind.
"""
from __future__ import annotations

from collections import deque

import numpy as np

from catan.rl.spaces import (OBS_DIM as V1_DIM, MAXP, NV, NE, NH, PIPS, PORT_KINDS,
                             encode_obs as encode_v1, static_features)  # noqa: F401  (re-export)
from catan.engine import TOPOLOGY as T, RESOURCES
from catan.engine.board import DESERT
from catan.engine.game import longest_trail

VE, EV, VN, VH = T.vertex_edges, T.edge_vertices, T.vertex_neighbors, T.vertex_hexes
INF = 99
DCAP = 3
VX = 1 + MAXP
EX = MAXP
PX = 5 + 1 + 1 + 3 + 3 + 2 + 2 + len(PORT_KINDS)
OFFSETS = {"vertex2": (V1_DIM, V1_DIM + NV * VX)}
OFFSETS["edge2"] = (OFFSETS["vertex2"][1], OFFSETS["vertex2"][1] + NE * EX)
OFFSETS["players2"] = (OFFSETS["edge2"][1], OFFSETS["edge2"][1] + MAXP * PX)
OBS_DIM = OFFSETS["players2"][1]


def road_distance(g, p: int) -> list[int]:
    """0-1 BFS: new roads ``p`` needs to reach each vertex. Own roads cost 0, free
    edges 1, opponent roads are impassable, and a path cannot run through an
    opponent's building (same connectivity rule as Game._road_ok)."""
    vo, eo = g.vertex_owner, g.edge_owner
    pl = g.players[p]
    d = [INF] * NV
    dq = deque()
    for v in list(pl.settlements) + list(pl.cities) + [v for e in pl.roads for v in EV[e]]:
        if d[v] and vo[v] in (-1, p):
            d[v] = 0
            dq.append(v)
    while dq:
        v = dq.popleft()
        if vo[v] not in (-1, p):
            continue  # reached, but no continuing through an opponent building
        for e in VE[v]:
            o = eo[e]
            if o not in (-1, p):
                continue
            w = EV[e][0] if EV[e][1] == v else EV[e][1]
            nd = d[v] + (o == -1)
            if nd < d[w] and nd <= DCAP:
                d[w] = nd
                (dq.appendleft if o == p else dq.append)(w)
    return d


_GAIN_CACHE: dict = {}


def _road_gain(g, q, legal) -> int:
    """Longest-road gain from one more road. Cached: roads and buildings change rarely."""
    vo = g.vertex_owner
    roads = g.players[q].roads
    key = (q, tuple(roads), tuple(legal), tuple(vo))
    r = _GAIN_CACHE.get(key)
    if r is None:
        if len(_GAIN_CACHE) > 100_000:
            _GAIN_CACHE.clear()
        base = g.road_length[q]
        r = max((longest_trail(roads + [e], vo, q) - base for e in legal), default=0)
        _GAIN_CACHE[key] = r
    return r


class _Layout:
    """Layout-derived constants (pips per hex/vertex, port spots), built once per board."""

    def __init__(self, lay):
        self.layout = lay
        self.pip_h = [0 if lay.hex_terrain[h] == DESERT else PIPS[lay.hex_number[h]] for h in range(NH)]
        self.res_h = [RESOURCES.index(t) if t != DESERT else -1 for t in lay.hex_terrain]
        self.tot = [sum(self.pip_h[h] for h in VH[v]) for v in range(NV)]
        vp = lay.vertex_ports()
        self.port_spots = [[v for v, t in vp.items() if t == k] for k in PORT_KINDS]
        self.eff = {}  # robber hex -> robber-aware vertex pips / 15
        for rh in range(NH):
            self.eff[rh] = np.asarray([sum(self.pip_h[h] for h in VH[v] if h != rh) for v in range(NV)],
                                      dtype=np.float32) / 15


_LAYOUTS: dict = {}  # id(layout) -> _Layout; many games (one per env) are live at once


# Board-derived part of the v2 block, per absolute seat, memoized on everything it
# reads (layout, robber, buildings, roads, road lengths, ports). It only changes
# when someone builds or moves the robber, while the learner decides many times
# in between (trades, several builds, discards), so most encodes are cache hits.
_BOARD_CACHE: dict = {}


def _layout(lay) -> _Layout:
    L = _LAYOUTS.get(id(lay))
    if L is None or L.layout is not lay:
        if len(_LAYOUTS) > 512:
            _LAYOUTS.clear()
        L = _LAYOUTS[id(lay)] = _Layout(lay)
    return L


def _board_rows(g, L):
    """Per seat q: (road-distance column, legal road edges, players2 row)."""
    pls = g.players
    key = (id(L), g.n, g.robber, tuple(g.vertex_owner), tuple(g.vertex_level), tuple(g.edge_owner),
           tuple(g.road_length),
           tuple((tuple(pl.settlements), tuple(pl.cities), tuple(pl.roads), tuple(sorted(pl.ports))) for pl in pls))
    hit = _BOARD_CACHE.get(key)
    if hit is not None and hit[0] is L:
        return hit[1]
    n = g.n
    vo, lv, robber, pip_h, tot = g.vertex_owner, g.vertex_level, g.robber, L.pip_h, L.tot
    free = [o == -1 for o in vo]
    for v, o in enumerate(vo):
        if o != -1:
            for u in VN[v]:
                free[u] = False
    free_spots = [v for v in range(NV) if free[v]]
    best_spot = max(free_spots, key=tot.__getitem__, default=None)
    lr_max = max(g.road_length)
    rows = []
    for q in range(n):
        pl = pls[q]
        d = road_distance(g, q)
        dcol = np.minimum(d, DCAP) / DCAP
        legal = g.legal_road_edges(q)
        prod = [0.0] * 5
        lost = 0
        for v in pl.settlements + pl.cities:
            for h in VH[v]:
                if pip_h[h]:
                    if h == robber:
                        lost += pip_h[h] * lv[v]
                    else:
                        prod[L.res_h[h]] += pip_h[h] * lv[v]
        cnt, best = [0, 0, 0], [0, 0, 0]
        near = INF
        for v in free_spots:
            k = d[v]
            near = min(near, k)
            if k < 3:
                cnt[k] += 1
                best[k] = max(best[k], tot[v])
        db = d[best_spot] if best_spot is not None else INF
        gain = _road_gain(g, q, legal) if pl.roads and len(pl.roads) < 15 else 0
        ports = [kind in pl.ports or any(free[v] and d[v] <= 2 for v in spots)
                 for kind, spots in zip(PORT_KINDS, L.port_spots)]
        row = ([x / 10 for x in prod] + [sum(prod) / 30, lost / 10] + [c / 5 for c in cnt]
               + [b / 15 for b in best] + [min(db, 4) / 4, min(near, 4) / 4, gain,
                                           (lr_max - g.road_length[q]) / 10] + ports)
        rows.append((dcol, legal, np.asarray(row, dtype=np.float32)))
    if len(_BOARD_CACHE) > 4096:  # ~6 KB per entry
        _BOARD_CACHE.clear()
    _BOARD_CACHE[key] = (L, rows)
    return rows


def encode_obs(g, me: int, static: np.ndarray | None = None) -> np.ndarray:
    out = np.zeros(OBS_DIM, dtype=np.float32)
    out[:V1_DIM] = encode_v1(g, me, static)
    n = g.n
    L = _layout(g.layout)
    a = OFFSETS["vertex2"][0]
    vx = out[a:a + NV * VX].reshape(NV, VX)
    vx[:, 0] = L.eff[g.robber]
    a = OFFSETS["edge2"][0]
    ex = out[a:a + NE * EX].reshape(NE, EX)
    a = OFFSETS["players2"][0]
    px = out[a:a + MAXP * PX].reshape(MAXP, PX)
    for q, (dcol, legal, row) in enumerate(_board_rows(g, L)):
        r = (q - me) % n
        vx[:, 1 + r] = dcol
        ex[legal, r] = 1
        px[r] = row
    return out
