"""Extended ``evox`` genome (D29 r3): also evolves the goal weights, hand score and resource values that
``intermediate`` takes from catan.bots.common as fixed constants. ``INTERMEDIATE_ZX`` again reproduces
intermediate exactly. A genome of length ``DIM_X`` selects this class (see ``agent.make``).
"""
from __future__ import annotations

from catan.bots import common as C
from catan.bots.intermediate import IllegalAction
from catan.engine.actions import A
from catan.engine.board import GENERIC_PORT, RESOURCES

from .agent import GENES, EvoxAgent

EXTRA = (
    # goal weights (common._goals)
    ("g_city", 1.0, 1.2), ("g_settle", 1.0, 1.1), ("g_road", 1.0, 0.6), ("g_road_settle", 1.0, 0.5),
    ("g_dev", 1.0, 0.45),
    # hand score (common._hand_score)
    ("hs_miss", 1.0, 1.3), ("hs_second", 1.0, 0.4), ("hs_card", 0.1, 0.03), ("hs_over", 0.1, 0.06),
    # resource value (common._resource_values): base + own * 1 / (1 + prod / 6)
    ("rv_base", 1.0, 0.6), ("rv_own", 1.0, 0.6),
)
GENES_X = GENES + EXTRA
DIM_X = len(GENES_X)
INTERMEDIATE_ZX = tuple(v / s for _, s, v in GENES_X)


class EvoxXAgent(EvoxAgent):
    name = "evox"

    # ---- genome versions of the catan.bots.common value model
    def _rv(self, g, me, bi, prod, vp=None):
        p = self.g
        vp = g.victory_points(me) if vp is None else vp
        late = min(1.0, max(0.0, (vp - 3) / 5))
        phase_w = [1.15 - 0.3 * late, 1.15 - 0.3 * late, 1.0 - 0.2 * late, 1.0 + 0.2 * late, 0.95 + 0.3 * late]
        return [bi.scarcity[r] * phase_w[r] * (p["rv_base"] + p["rv_own"] / (1.0 + prod[r] / 6.0)) for r in range(5)]

    def _goals(self, g, me, spots):
        p, pl, out = self.g, g.players[me], []
        if pl.settlements and g.supply(me, "city") > 0:
            out.append((C.CITY, p["g_city"]))
        if g.supply(me, "settlement") > 0:
            if any(d == 0 for _, d in spots):
                out.append((C.SETTLEMENT, p["g_settle"]))
            elif spots and g.supply(me, "road") > 0:
                out.append((C.ROAD, p["g_road"]))
                out.append((C.SETTLEMENT, p["g_road_settle"]))
        if g.dev_deck:
            out.append((C.DEV, p["g_dev"]))
        if not out and g.supply(me, "road") > 0:
            out.append((C.ROAD, 0.2))
        return out

    def _hs(self, hand, gl, rv):
        p = self.g
        scores = sorted((w / (1.0 + p["hs_miss"] * sum(max(0, c[i] - hand[i]) * rv[i] for i in range(5)))
                         for c, w in gl), reverse=True)
        s = (scores[0] if scores else 0) + (p["hs_second"] * scores[1] if len(scores) > 1 else 0)
        s += p["hs_card"] * sum(hand[i] * rv[i] for i in range(5))
        over = sum(hand) - 7
        return s - p["hs_over"] * over if over > 0 else s

    def _spots(self, g, me):
        return C.free_spots(g, me) if g.supply(me, "settlement") > 0 else ()

    def hand_ctx(self, g, me):
        bi = C.board_info(g.layout)
        return self._rv(g, me, bi, C.production(g, me, bi)), self._goals(g, me, self._spots(g, me))

    def trade_delta(self, g, me, give, get, ctx=None):
        rv, gl = ctx or self.hand_ctx(g, me)
        h = g.players[me].hand
        after = [h[i] - give[i] + get[i] for i in range(5)]
        if min(after) < 0:
            return -9.0
        return self._hs(after, gl, rv) - self._hs(h, gl, rv)

    def discard(self, g, me, legal):
        rv, gl = self.hand_ctx(g, me)
        h = g.players[me].hand
        out = []
        for a in legal:
            hh = h[:]
            hh[a.value] -= 1
            out.append((self._hs(hh, gl, rv), a))
        return out

    # ---- intermediate.evaluate with the genome value model
    def _eval(self, g, me, bi, hand=None):
        if g.is_over:
            return 100.0 if g.winner == me else g.victory_points(me)
        w = self.w
        pl = g.players[me]
        hand = pl.hand if hand is None else hand
        prod = C.production(g, me, bi)
        rv = self._rv(g, me, bi, prod)
        v = float(g.victory_points(me))
        v += w["prod"] * sum(prod[r] * rv[r] for r in range(5))
        v += w["diversity"] * sum(1 for x in prod if x > 0)
        for port in pl.ports:
            v += w["port3"] if port == GENERIC_PORT else w["port2"] * prod[RESOURCES.index(port)]
        spots = self._spots(g, me)
        if spots:
            best = [0.0, 0.0, 0.0]
            for u, d in spots:
                s = sum(bi.vertex_pips[u][r] * rv[r] for r in range(5))
                if s > best[d]:
                    best[d] = s
            v += w["spot0"] * best[0] + w["spot1"] * best[1] + w["spot2"] * best[2]
        v += w["hand"] * self._hs(hand, self._goals(g, me, spots), rv)
        v += w["dev_held"] * (sum(pl.dev) - pl.dev[1])
        v += w["knight_played"] * min(pl.knights, 4)
        v -= w["overflow"] * max(0, sum(hand) - g.config.discard_limit)
        if g.longest_road_owner != me:
            v += w["road_len"] * min(g.road_length[me], 6)
        return v

    def candidate_values(self, g, me, legal, bi=None):
        """IntermediateAgent.candidate_values with self._eval (and the knight_bonus gene)."""
        bi = bi or C.board_info(g.layout)
        w = self.w
        out = []
        base = self._eval(g, me, bi)
        pl = g.players[me]
        for a in legal:
            t = a.type
            if t == A.END_TURN:
                out.append((base, a))
            elif t == A.MARITIME_TRADE:
                if not g.is_legal(a):
                    raise IllegalAction(f"illegal action {a}")
                give, get = a.value
                h = pl.hand[:]
                h[give] -= g.trade_rate(me, give)
                h[get] += 1
                out.append((self._eval(g, me, bi, h), a))
            elif t in (A.BUILD_CITY, A.BUILD_SETTLEMENT, A.BUILD_ROAD, A.PLAY_YEAR_OF_PLENTY):
                c = g.clone()
                c.step(a)
                out.append((self._eval(c, me, bi), a))
            elif t == A.BUY_DEV:
                h = [pl.hand[i] - C.DEV[i] for i in range(5)]
                out.append((self._eval(g, me, bi, h) + w["dev_ev"], a))
            elif t == A.PLAY_KNIGHT:
                if self.knight_worth(g, me):
                    c = g.clone()
                    c.step(a)
                    out.append((self._eval(c, me, bi) + self.g["knight_bonus"], a))
            elif t == A.PLAY_MONOPOLY:
                tot = C.opponents_resource_totals(g, me)
                rv, gl = self.hand_ctx(g, me)
                hh = pl.hand[:]
                hh[a.value] += tot[a.value]
                gain = w["hand"] * (self._hs(hh, gl, rv) - self._hs(pl.hand, gl, rv))
                if tot[a.value] >= 3:
                    out.append((base + gain - w["dev_held"], a))
            elif t == A.PLAY_ROAD_BUILDING:
                rs = self.road_scores(g, me, g.legal_road_edges(me), bi)
                if any(s > 0 for s, _ in rs):
                    out.append((base + 0.25 - w["dev_held"] + 0.01 * max(s for s, _ in rs), a))
        return out
