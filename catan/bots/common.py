"""Shared, information-safe helpers for heuristic and search bots.

Everything here reads only what ``game.actor`` may see: the board, public
piece positions, the bank, public counts (cards in hand, dev cards held,
knights played) and the actor's own hand/dev cards. Opponents' per-resource
totals are computed as ``19 - bank - own`` which is public (cards are
conserved), never by reading their hands.
"""
from __future__ import annotations

import random
from functools import lru_cache

from catan.engine.board import TOPOLOGY as T, RESOURCES, DESERT
from catan.engine.game import COST, BANK_START

PIPS = {2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 8: 5, 9: 4, 10: 3, 11: 2, 12: 1}
NV, NE, NH = T.num_vertices, T.num_edges, T.num_hexes
SETTLEMENT, CITY, DEV, ROAD = COST["settlement"], COST["city"], COST["dev"], COST["road"]


class BoardInfo:
    """Static, layout-only statistics (cached per layout)."""

    def __init__(self, layout):
        self.layout = layout
        self.hex_pips = [PIPS.get(n, 0) if n else 0 for n in layout.hex_number]
        self.hex_res = [RESOURCES.index(t) if t != DESERT else -1 for t in layout.hex_terrain]
        self.res_pips = [0] * 5
        for h in range(NH):
            if self.hex_res[h] >= 0:
                self.res_pips[self.hex_res[h]] += self.hex_pips[h]
        mean = sum(self.res_pips) / 5
        # scarcity: rarer resources on this board are worth more (sqrt damped, clipped)
        self.scarcity = [min(1.8, max(0.6, (mean / max(1, x)) ** 0.5)) for x in self.res_pips]
        self.vertex_port = layout.vertex_ports()
        self.vertex_pips = []
        for v in range(NV):
            vec = [0] * 5
            for h in T.vertex_hexes[v]:
                if self.hex_res[h] >= 0:
                    vec[self.hex_res[h]] += self.hex_pips[h]
            self.vertex_pips.append(tuple(vec))


@lru_cache(maxsize=256)
def _board_info(layout) -> BoardInfo:
    return BoardInfo(layout)


_last_bi = (None, None)  # (layout, BoardInfo): skips hashing the frozen layout per call


def board_info(layout) -> BoardInfo:
    global _last_bi
    last = _last_bi  # one read: the pair is swapped atomically, safe across threads
    if last[0] is layout:
        return last[1]
    bi = _board_info(layout)
    _last_bi = (layout, bi)
    return bi


_VN = T.vertex_neighbors
# vertex -> ((edge, other end), ...) in vertex_edges order (BFS order unchanged)
_V_EDGE_NB = tuple(tuple((e, T.edge_vertices[e][1] if T.edge_vertices[e][0] == v else T.edge_vertices[e][0])
                         for e in T.vertex_edges[v]) for v in range(NV))


def spot_free(g, v) -> bool:
    vo = g.vertex_owner
    if vo[v] != -1:
        return False
    for u in _VN[v]:
        if vo[u] != -1:
            return False
    return True


def production(g, p, bi: BoardInfo | None = None, robber: bool = True) -> list[float]:
    """Expected pips per resource for player p (cities count double)."""
    bi = bi or board_info(g.layout)
    out = [0.0] * 5
    for v in g.players[p].settlements + g.players[p].cities:
        lvl = g.vertex_level[v]
        for h in T.vertex_hexes[v]:
            r = bi.hex_res[h]
            if r >= 0 and not (robber and h == g.robber):
                out[r] += bi.hex_pips[h] * lvl
    return out


def opponents_resource_totals(g, me) -> list[int]:
    """Per-resource cards held by all opponents together (public by conservation)."""
    own = g.players[me].hand
    return [BANK_START - g.bank[r] - own[r] for r in range(5)]


def hand_count(g, p) -> int:
    """Public: number of cards in p's hand."""
    return g.hand_count(p)


def dev_count(g, p) -> int:
    """Public: number of unplayed dev cards p holds."""
    return g.dev_count(p)


def road_distances(g, p, max_depth: int = 3) -> dict[int, int]:
    """Min number of new roads player p needs to touch each vertex (BFS over
    free edges, not passing through opponents' buildings). 0 = already touches."""
    eo, vo = g.edge_owner, g.vertex_owner
    dist: dict[int, int] = {}
    frontier = []
    pl = g.players[p]
    for v in pl.settlements + pl.cities:
        dist[v] = 0
        frontier.append(v)
    for e in pl.roads:
        for v in T.edge_vertices[e]:
            if v not in dist:
                dist[v] = 0
                frontier.append(v)
    d = 0
    while frontier and d < max_depth:
        d += 1
        nxt = []
        for v in frontier:
            o = vo[v]
            if o != -1 and o != p:
                continue  # can't extend through an opponent's building
            for e, u in _V_EDGE_NB[v]:
                if eo[e] == -1 and u not in dist:
                    dist[u] = d
                    nxt.append(u)
        frontier = nxt
    return dist


def resource_values(g, me, bi: BoardInfo | None = None, vp: int | None = None) -> list[float]:
    """Value model: scarcity on this board x game-phase
    weight x how little of it I produce. Pass ``vp`` (e.g. public VP) when
    valuing for an opponent: ``victory_points`` includes hidden VP cards."""
    bi = bi or board_info(g.layout)
    return _resource_values(g, me, bi, production(g, me, bi), vp)


def _resource_values(g, me, bi, prod, vp: int | None = None) -> list[float]:
    """resource_values() given ``production(g, me, bi)`` (evaluate needs both)."""
    vp = g.victory_points(me) if vp is None else vp
    late = min(1.0, max(0.0, (vp - 3) / 5))   # 0 early .. 1 late
    phase_w = [1.15 - 0.3 * late, 1.15 - 0.3 * late, 1.0 - 0.2 * late, 1.0 + 0.2 * late, 0.95 + 0.3 * late]
    out = []
    for r in range(5):
        own = 1.0 / (1.0 + prod[r] / 6.0)  # 0.5 at one 6-hex worth of production
        out.append(bi.scarcity[r] * phase_w[r] * (0.6 + 0.6 * own))
    return out


def goals(g, me, bi=None) -> list[tuple[tuple, float, str]]:
    """Builds the actor is working toward, as (cost, weight, name)."""
    return _goals(g, me, free_spots(g, me) if g.supply(me, "settlement") > 0 else ())


_SPOTS_CACHE: dict = {}


def free_spots(g, me) -> list[tuple[int, int]]:
    """[(vertex, roads needed)] for free settlement spots within 2 roads, in
    road_distances order. Shared by goals() and intermediate.evaluate.

    Memoized on everything the answer depends on (board ownership and me's
    piece lists, by value): candidate evaluation and search rollouts keep
    asking about the same road network."""
    pl = g.players[me]
    key = (me, tuple(g.edge_owner), tuple(g.vertex_owner), tuple(pl.settlements),
           tuple(pl.cities), tuple(pl.roads))
    spots = _SPOTS_CACHE.get(key)
    if spots is None:
        if len(_SPOTS_CACHE) >= 50_000:
            _SPOTS_CACHE.clear()  # bounded memory; entries are cheap to rebuild
        spots = _SPOTS_CACHE[key] = tuple(
            (v, d) for v, d in road_distances(g, me, 2).items() if spot_free(g, v))
    return list(spots)


def _goals(g, me, spots) -> list[tuple[tuple, float, str]]:
    """goals() given ``free_spots(g, me)`` (only read when settlements remain)."""
    pl = g.players[me]
    out = []
    if pl.settlements and g.supply(me, "city") > 0:
        out.append((CITY, 1.2, "city"))
    if g.supply(me, "settlement") > 0:
        if any(d == 0 for _, d in spots):
            out.append((SETTLEMENT, 1.1, "settlement"))
        elif spots and g.supply(me, "road") > 0:
            out.append((ROAD, 0.6, "road"))
            out.append((SETTLEMENT, 0.5, "settlement"))
    if g.dev_deck:  # deck size is public
        out.append((DEV, 0.45, "dev"))
    if not out and g.supply(me, "road") > 0:
        out.append((ROAD, 0.2, "road"))
    return out


_HAND_CACHE: dict = {}


def hand_score(hand, gl, rv) -> float:
    """How useful a hand is toward the goals: best goal weight / (1 + weighted
    missing cards), half credit for the second goal, plus a small per-card value."""
    # Pure in the values of its arguments: memoized by value (trade evaluation
    # rescores the same base hand for every template).
    key = (tuple(hand), tuple(rv), tuple(gl))
    s = _HAND_CACHE.get(key)
    if s is None:
        if len(_HAND_CACHE) >= 100_000:
            _HAND_CACHE.clear()
        s = _HAND_CACHE[key] = _hand_score(hand, gl, rv)
    return s


def _hand_score(hand, gl, rv) -> float:
    # Unrolled, but every sum() sees the same values in the same order as the
    # original genexprs: float sums (compensated in 3.12) stay bit-identical.
    h0, h1, h2, h3, h4 = hand
    r0, r1, r2, r3, r4 = rv
    scores = []
    for (c0, c1, c2, c3, c4), w, _ in gl:
        miss = sum(((c0 - h0 if c0 > h0 else 0) * r0, (c1 - h1 if c1 > h1 else 0) * r1,
                    (c2 - h2 if c2 > h2 else 0) * r2, (c3 - h3 if c3 > h3 else 0) * r3,
                    (c4 - h4 if c4 > h4 else 0) * r4))
        scores.append(w / (1.0 + 1.3 * miss))
    scores.sort(reverse=True)
    s = (scores[0] if scores else 0) + (0.4 * scores[1] if len(scores) > 1 else 0)
    s += 0.03 * sum((h0 * r0, h1 * r1, h2 * r2, h3 * r3, h4 * r4))
    over = sum(hand) - 7
    if over > 0:
        s -= 0.06 * over
    return s


def needs(hand, cost) -> list[int]:
    return [max(0, cost[i] - hand[i]) for i in range(5)]


def leader(g, me) -> int:
    """Opponent with the most public VP (ties: more cards, then seat)."""
    # same as max(opponents, key=(public VP, hand count, -seat)); keys are unique
    best, best_key = -1, None
    for q in range(g.n):
        if q != me:
            k = (g.public_victory_points(q), g.hand_count(q), -q)
            if best_key is None or k > best_key:
                best, best_key = q, k
    return best


# ---------------------------------------------------------------- determinization
def unseen_dev_pool(g, me) -> list[int]:
    """Sorted multiset of dev cards not visible to ``me`` (deck + opponents'
    unplayed). Its composition is public; see ``Game.unseen_dev_pool``."""
    return g.unseen_dev_pool(me)


def determinize(g, me, rng: random.Random, trade_safe: bool = True, belief=None):
    """Clone ``g`` and resample everything hidden from ``me``: opponents'
    hands (keeping per-player counts and the public per-resource totals),
    opponents' dev cards (keeping counts), deck order and the RNG.

    With ``trade_safe`` a pending domestic trade stays executable: the
    proposer holds what they offered and counter-offerers what they asked for
    (rejection sampling, 50 tries, then the constraint is dropped).

    ``belief`` (a ``belief.HandPosterior`` for ``me``, synced to ``g``) draws
    the opponents' hands jointly from the posterior given ``me``'s public
    events instead of uniformly; it falls back to uniform when the posterior
    is lost, out of sync, or has no state covering a pending trade.

    Without ``belief`` this is ``Game.determinized`` (works on a full Game and
    on an InfoSetGame); the belief path needs the full Game."""
    if belief is None:
        return g.determinized(me, rng, trade_safe)
    c = g.clone()
    opps = [q for q in range(g.n) if q != me]
    totals = opponents_resource_totals(g, me)
    base_pool = [r for r in range(5) for _ in range(totals[r])]
    counts = [hand_count(g, q) for q in opps]
    req = {}
    if trade_safe and g.trade is not None:
        tr = g.trade
        if tr["proposer"] != me:
            req[tr["proposer"]] = list(tr["give"])
        for q, resp in tr["responses"].items():
            if resp is not None and q != me:
                need = req.get(q, [0] * 5)
                req[q] = [max(need[i], resp[1][i]) for i in range(5)]
    hands = None
    if belief is not None and belief.synced == len(g.action_log):
        hands = belief.sample(rng, req or None)
    for _ in range(0 if hands is not None else 50):
        pool = base_pool[:]
        rng.shuffle(pool)
        hands, i = {}, 0
        for q, k in zip(opps, counts):
            h = [0] * 5
            for r in pool[i:i + k]:
                h[r] += 1
            hands[q] = h
            i += k
        if all(all(hands[q][j] >= need[j] for j in range(5)) for q, need in req.items()):
            break
    for q in opps:
        c.players[q].hand = hands[q]
    dpool = unseen_dev_pool(g, me)
    rng.shuffle(dpool)
    i = 0
    for q in opps:
        k = dev_count(g, q)
        new_k = sum(g.players[q].dev_new)  # cards bought this turn: count is public
        d, dn = [0] * 5, [0] * 5
        for j, t in enumerate(dpool[i:i + k]):
            d[t] += 1
            if j < new_k:
                dn[t] += 1
        c.players[q].dev, c.players[q].dev_new = d, dn
        i += k
    c.dev_deck = dpool[i:]
    c.rng = random.Random(rng.getrandbits(64))
    return c
