"""Observation and action encoding for RL.

Everything is from the point of view of one seat ("me"). Seats are rotated so
"me" is relative index 0 and the next player clockwise is 1. Only information
visible to that seat is read: board, public counts, and its own hand/devs.
"""
from __future__ import annotations

import numpy as np

from catan.engine import A, Action, Phase, TOPOLOGY, RESOURCES, TRADE_TEMPLATES
from catan.engine.board import GENERIC_PORT, DESERT
from catan.engine.game import PREBUILT_ACTIONS

OBS_VERSION = 1
ACTION_VERSION = 1
MAXP = 4  # observation is always laid out for 4 seats; absent seats are zero

T = TOPOLOGY
NH, NV, NE = T.num_hexes, T.num_vertices, T.num_edges
PIPS = {2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 8: 5, 9: 4, 10: 3, 11: 2, 12: 1}
TERRAINS = RESOURCES + (DESERT,)
PORT_KINDS = (GENERIC_PORT,) + RESOURCES
PHASES = (Phase.SETUP_SETTLEMENT, Phase.SETUP_ROAD, Phase.PRE_ROLL, Phase.DISCARD,
          Phase.MOVE_ROBBER, Phase.STEAL, Phase.MAIN, Phase.ROAD_BUILDING,
          Phase.TRADE_RESPONSE, Phase.TRADE_CONFIRM)
PHASE_INDEX = {p: i for i, p in enumerate(PHASES)}

# ----------------------------------------------------------------- actions
# (type, value) per index. For STEAL / CONFIRM_TRADE the value is the
# relative seat (1..3) of the victim/partner, not the absolute seat.
_ACTIONS: list[tuple[str, object]] = []
_ACTIONS += [(A.BUILD_ROAD, e) for e in range(NE)]
_ACTIONS += [(A.BUILD_SETTLEMENT, v) for v in range(NV)]
_ACTIONS += [(A.BUILD_CITY, v) for v in range(NV)]
_ACTIONS += [(A.MOVE_ROBBER, h) for h in range(NH)]
_ACTIONS += [(A.STEAL, r) for r in range(1, MAXP)]
_ACTIONS += [(A.DISCARD, r) for r in range(5)]
_ACTIONS += [(A.PLAY_YEAR_OF_PLENTY, (a, b)) for a in range(5) for b in range(a, 5)]
_ACTIONS += [(A.PLAY_MONOPOLY, r) for r in range(5)]
_ACTIONS += [(A.MARITIME_TRADE, (g, r)) for g in range(5) for r in range(5) if g != r]
_ACTIONS += [(A.OFFER_TRADE, t) for t in TRADE_TEMPLATES]
_ACTIONS += [(A.COUNTER_OFFER, t) for t in TRADE_TEMPLATES]
_ACTIONS += [(A.ACCEPT_TRADE, None), (A.REJECT_TRADE, None)]
_ACTIONS += [(A.CONFIRM_TRADE, r) for r in range(1, MAXP)]
_ACTIONS += [(A.CANCEL_TRADE, None), (A.ROLL, None), (A.END_TURN, None), (A.BUY_DEV, None),
             (A.PLAY_KNIGHT, None), (A.PLAY_ROAD_BUILDING, None)]
ACTIONS = tuple(_ACTIONS)
ACT_DIM = len(ACTIONS)
_RELATIVE = (A.STEAL, A.CONFIRM_TRADE)
_INDEX = {a: i for i, a in enumerate(ACTIONS) if a[0] not in _RELATIVE}
_REL_BASE = {t: next(i for i, a in enumerate(ACTIONS) if a[0] == t) for t in _RELATIVE}


# OFFER_TRADE and COUNTER_OFFER indices (optional action filter, see EnvConfig.disable_domestic_trade)
ROBBER_BASE = next(i for i, a in enumerate(ACTIONS) if a[0] == A.MOVE_ROBBER)
PROPOSE_MASK = np.array([t in (A.OFFER_TRADE, A.COUNTER_OFFER) for t, _ in ACTIONS])


def encode_action(a: Action, me: int, n: int) -> int:
    if a.type in _REL_BASE:
        return _REL_BASE[a.type] + (a.value - me) % n - 1
    return _INDEX[(a.type, a.value)]


def decode_action(i: int, me: int, n: int) -> Action:
    t, v = ACTIONS[i]
    if t in _REL_BASE:
        v = (me + v) % n
    return Action(me, t, v)


# legal_actions() returns the engine's prebuilt (immortal) Action objects, so
# their ids are stable keys: no (type, value) hashing of nested trade tuples.
# Indexes come from encode_action itself; anything else takes the slow path.
_FAST_INDEX: dict = {}


def _fast_index(me: int, n: int) -> dict:
    t = _FAST_INDEX.get((me, n))
    if t is None:
        t = {}
        for a in PREBUILT_ACTIONS:
            if a.player != me:
                continue
            if a.type in _REL_BASE:
                if type(a.value) is not int or not 0 <= a.value < n or a.value == me:
                    continue  # never legal at this table size
            elif (a.type, a.value) not in _INDEX:
                continue
            t[id(a)] = encode_action(a, me, n)
        _FAST_INDEX[(me, n)] = t
    return t


def action_mask(legal, me: int, n: int) -> np.ndarray:
    m = np.zeros(ACT_DIM, dtype=bool)
    fast = _fast_index(me, n).get
    idx = []
    for a in legal:
        i = fast(id(a))
        idx.append(encode_action(a, me, n) if i is None else i)
    m[idx] = True
    return m


def action_label(i: int) -> str:
    t, v = ACTIONS[i]
    return t if v is None else f"{t}:{v}"


# ------------------------------------------------------------- observation
VN = T.vertex_neighbors
_PORT_INDEX = {k: i for i, k in enumerate(PORT_KINDS)}

HEX_F = len(TERRAINS) + 2                 # terrain one-hot, pips/5, robber
VSTAT_F = len(PORT_KINDS) + 5 + 1         # port one-hot, pips per resource, total pips
VDYN_F = 2 * MAXP + 1                     # settlement by rel seat, city by rel seat, spot free
EDGE_F = MAXP
PUB_F = 20
OWN_F = 16
GLOB_F = 5 + 1 + len(PHASES) + 2 + 11 + 3
TRADE_F = 1 + MAXP + 10 + MAXP * 14

_sizes = [("hex", NH * HEX_F), ("vertex_static", NV * VSTAT_F), ("vertex_dyn", NV * VDYN_F),
          ("edge", NE * EDGE_F), ("players", MAXP * PUB_F), ("own", OWN_F),
          ("global", GLOB_F), ("trade", TRADE_F)]
OFFSETS = {}
_o = 0
for _name, _s in _sizes:
    OFFSETS[_name] = (_o, _o + _s)
    _o += _s
OBS_DIM = _o


def static_features(layout) -> np.ndarray:
    """Layout-only part (hexes without robber + vertex statics). Cache per game."""
    hexf = np.zeros((NH, HEX_F), dtype=np.float32)
    for h in range(NH):
        hexf[h, TERRAINS.index(layout.hex_terrain[h])] = 1
        num = layout.hex_number[h]
        hexf[h, len(TERRAINS)] = PIPS.get(num, 0) / 5
    vs = np.zeros((NV, VSTAT_F), dtype=np.float32)
    for v, kind in layout.vertex_ports().items():
        vs[v, PORT_KINDS.index(kind)] = 1
    for v in range(NV):
        for h in T.vertex_hexes[v]:
            terr = layout.hex_terrain[h]
            if terr != DESERT:
                p = PIPS[layout.hex_number[h]] / 5
                vs[v, len(PORT_KINDS) + RESOURCES.index(terr)] += p
                vs[v, -1] += p / 3
    return np.concatenate([hexf.ravel(), vs.ravel()])


def encode_obs(g, me: int, static: np.ndarray | None = None) -> np.ndarray:
    """Flat float32 observation of game ``g`` from seat ``me``'s point of view."""
    n = g.n
    out = np.zeros(OBS_DIM, dtype=np.float32)
    if static is None:
        static = static_features(g.layout)
    out[:len(static)] = static
    a, _ = OFFSETS["hex"]
    out[a + g.robber * HEX_F + HEX_F - 1] = 1

    rel_of = [(q - me) % n for q in range(n)]
    # Plain loops over the engine's lists: converting them with np.asarray and
    # fancy-indexing cost more than the work itself at this size.
    vo, lv = g.vertex_owner, g.vertex_level
    a, _ = OFFSETS["vertex_dyn"]
    free = [1.0] * NV
    for v, o in enumerate(vo):
        if o != -1:  # owned vertices always have level 1 or 2
            out[a + v * VDYN_F + rel_of[o] + MAXP * (lv[v] - 1)] = 1
            free[v] = 0.0
            for u in VN[v]:
                free[u] = 0.0
    out[a + VDYN_F - 1:a + NV * VDYN_F:VDYN_F] = free

    a, _ = OFFSETS["edge"]
    for e, o in enumerate(g.edge_owner):
        if o != -1:
            out[a + e * EDGE_F + rel_of[o]] = 1

    a, b = OFFSETS["players"]
    pf = out[a:b].reshape(MAXP, PUB_F)
    actor = g.actor if not g.is_over else -1
    for q, pl in enumerate(g.players):
        # one row assignment instead of 20 scalar numpy writes
        f = [1, g.hand_count(q) / 10, g.dev_count(q) / 5, pl.knights / 5,  # public counts
             g.public_victory_points(q) / 10, g.road_length[q] / 15,
             g.longest_road_owner == q, g.largest_army_owner == q,
             (5 - len(pl.settlements)) / 5, (4 - len(pl.cities)) / 4, (15 - len(pl.roads)) / 15,
             0, 0, 0, 0, 0, 0,
             g.discard_remaining[q] / 10, g.current == q, actor == q]
        for k in pl.ports:
            f[11 + _PORT_INDEX[k]] = 1
        pf[rel_of[q]] = f

    a, _ = OFFSETS["own"]
    pl = g.players[me]
    out[a:a + 5] = [x / 10 for x in pl.hand]  # same float32 values for integer counts
    out[a + 5:a + 10] = pl.dev
    out[a + 10:a + 15] = pl.dev_new
    out[a + 15] = g.victory_points(me) / 10

    a, _ = OFFSETS["global"]
    out[a:a + 5] = [x / 19 for x in g.bank]
    out[a + 5] = len(g.dev_deck) / 25
    o = a + 6
    ph = PHASE_INDEX.get(g.phase)
    if ph is not None:
        out[o + ph] = 1
    o += len(PHASES)
    out[o] = g.rolled
    out[o + 1] = g.dev_played
    o += 2
    if g.last_roll:
        out[o + sum(g.last_roll) - 2] = 1
    o += 11
    out[o] = g.free_roads / 2
    out[o + 1] = g.trades_this_turn / 3
    out[o + 2] = g.turn / 200

    tr = g.trade
    if tr is not None:
        a, _ = OFFSETS["trade"]
        out[a] = 1
        out[a + 1 + rel_of[tr["proposer"]]] = 1
        out[a + 1 + MAXP:a + 6 + MAXP] = tr["give"]
        out[a + 6 + MAXP:a + 11 + MAXP] = tr["get"]
        base = a + 11 + MAXP
        for q in range(n):
            f = out[base + rel_of[q] * 14: base + rel_of[q] * 14 + 14]
            if q in tr["responses"]:
                r = tr["responses"][q]
                if r is None:
                    f[1] = 1
                elif r == (tr["give"], tr["get"]):
                    f[2] = 1
                else:
                    f[3] = 1
                    f[4:9] = r[0]
                    f[9:14] = r[1]
            elif q in tr["pending"]:
                f[0] = 1
    return out
