"""IntermediateAgent: value-model planner.

Setup scores pips x board scarcity, resource diversity, port synergy,
complement to the first settlement and expansion room. In MAIN every
deterministic candidate (builds, bank/port trades, Year of Plenty, Knight) is
applied to a clone and scored with ``evaluate`` (VP, robber-adjusted
production, expansion spots, hand progress toward goals, dev cards, awards);
random-outcome actions (BUY_DEV, Monopoly) are valued analytically from public
information. Domestic trades use the same hand model: a bounded number of
offers per turn built from templates, best counter confirmed, and no trades
that help a player close to winning.
"""
from __future__ import annotations

from catan.engine.actions import A, Action
from catan.engine.board import TOPOLOGY as T, GENERIC_PORT, RESOURCES
from catan.engine.game import TRADE_TEMPLATES, IllegalAction

from . import common as C
from .basic import HeuristicBase

W = {
    "prod": 0.045,       # per value-weighted pip
    "diversity": 0.12,   # per resource type produced
    "spot0": 0.08,      # best currently buildable spot (per pip)
    "spot1": 0.04,      # best spot one road away
    "spot2": 0.015,
    "hand": 0.3,
    "dev_held": 0.3,
    "knight_played": 0.08,
    "road_len": 0.03,
    "port2": 0.012,      # 2:1 port, per pip of that resource
    "port3": 0.12,
    "dev_ev": 0.42,      # expected value of a fresh dev card
    "overflow": 0.0,     # per card above the discard limit
    "setup_exp": 0.15,   # setup: expansion room weight
    "setup_div": 1.2,    # setup: per resource type at the spot
}


# (give, get, overpays, give resource, give count) per template, for offers()
_TEMPLATE_META = tuple((give, get, sum(give) > sum(get), next(i for i in range(5) if give[i]), max(give))
                       for give, get in TRADE_TEMPLATES)
assert all(sum(1 for x in give if x) == 1 for give, _, _, _, _ in _TEMPLATE_META)


def evaluate(g, me, bi=None, w=W) -> float:
    """Heuristic value of the position for ``me`` in VP-ish units. Uses only
    public information plus me's own hand and dev cards."""
    if g.is_over:
        return 100.0 if g.winner == me else g.victory_points(me)
    bi = bi or C.board_info(g.layout)
    pl = g.players[me]
    prod = C.production(g, me, bi)
    rv = C._resource_values(g, me, bi, prod)  # == C.resource_values, without recomputing prod
    v = float(g.victory_points(me))
    p0, p1, p2, p3, p4 = prod
    r0, r1, r2, r3, r4 = rv
    v += w["prod"] * sum((p0 * r0, p1 * r1, p2 * r2, p3 * r3, p4 * r4))  # same terms/order as before
    v += w["diversity"] * ((p0 > 0) + (p1 > 0) + (p2 > 0) + (p3 > 0) + (p4 > 0))
    for port in pl.ports:
        if port == GENERIC_PORT:
            v += w["port3"]
        else:
            v += w["port2"] * prod[RESOURCES.index(port)]
    spots = ()
    if g.supply(me, "settlement") > 0:
        spots = C.free_spots(g, me)  # computed once, reused by goals below
        best = [0.0, 0.0, 0.0]
        for u, d in spots:
            p0, p1, p2, p3, p4 = bi.vertex_pips[u]
            s = sum((p0 * r0, p1 * r1, p2 * r2, p3 * r3, p4 * r4))  # same terms/order as before
            if s > best[d]:
                best[d] = s
        v += w["spot0"] * best[0] + w["spot1"] * best[1] + w["spot2"] * best[2]
    v += w["hand"] * C.hand_score(pl.hand, C._goals(g, me, spots), rv)
    v += w["dev_held"] * (sum(pl.dev) - pl.dev[1])  # VP cards already counted
    v += w["knight_played"] * min(pl.knights, 4)
    v -= w["overflow"] * max(0, sum(pl.hand) - g.config.discard_limit)
    if g.longest_road_owner != me:
        v += w["road_len"] * min(g.road_length[me], 6)
    return v


def _evaluate_with_hand(g, me, bi, w, hand) -> float:
    """evaluate() of a copy of ``g`` where only me's hand (and the bank, which
    evaluate never reads) differ: what candidate_values used to get from
    clone() + a maritime trade / dev payment, without the clone."""
    pl = g.players[me]
    saved = pl.hand
    pl.hand = hand  # swap the list object; the original list is never touched
    try:
        return evaluate(g, me, bi, w)
    finally:
        pl.hand = saved


class IntermediateAgent(HeuristicBase):
    name = "intermediate"

    def __init__(self, seed=None, max_offers: int = 2, weights: dict | None = None, domestic: bool = True):
        super().__init__(seed, domestic=domestic)
        self.max_offers = max_offers if domestic else 0
        self.w = dict(W, **(weights or {}))

    # ------------------------------------------------------------ setup
    def spot_value(self, g, v, me, bi):
        rv = bi.scarcity
        return sum(bi.vertex_pips[v][r] * rv[r] for r in range(5))

    def setup_settlement(self, g, me, legal):
        bi = C.board_info(g.layout)
        mine = g.players[me].settlements
        have = [0] * 5
        for s in mine:
            for r in range(5):
                have[r] += bi.vertex_pips[s][r]
        out = []
        for a in legal:
            v = a.value
            pips = bi.vertex_pips[v]
            s = sum(pips[r] * bi.scarcity[r] for r in range(5))
            s += self.w["setup_div"] * sum(1 for x in pips if x)
            tot = [have[r] + pips[r] for r in range(5)]
            s += 1.0 * sum(1 for x in tot if x)                    # combined diversity
            if tot[0] and tot[1]:
                s += 1.0                                            # wood+brick to expand
            if tot[3] and tot[4]:
                s += 0.8                                            # wheat+ore for cities
            port = bi.vertex_port.get(v)
            if port == GENERIC_PORT:
                s += 1.0
            elif port:
                s += 0.25 * tot[RESOURCES.index(port)]
            # expansion room: best free spot two steps away
            exp = 0.0
            for u in T.vertex_neighbors[v]:
                for x in T.vertex_neighbors[u]:
                    if x != v and C.spot_free(g, x) and x not in T.vertex_neighbors[v]:
                        exp = max(exp, sum(bi.vertex_pips[x]))
            s += self.w["setup_exp"] * exp
            out.append((s, a))
        return out

    # ------------------------------------------------------------ robber
    def hex_block_score(self, g, me, h, bi):
        if any(g.vertex_owner[v] == me for v in T.hex_vertices[h]):
            return -100.0
        r = bi.hex_res[h]
        if r < 0:
            return -50.0
        s = 0.0
        victims = False
        for v in T.hex_vertices[h]:
            o = g.vertex_owner[v]
            if o != -1:
                s += bi.hex_pips[h] * g.vertex_level[v] * bi.scarcity[r] * (1 + 0.3 * g.public_victory_points(o))
                victims = victims or C.hand_count(g, o) > 0
        return s + (2.0 if victims else 0.0)

    def knight_worth(self, g, me):
        pl = g.players[me]
        k = pl.knights + 1
        holder = g.largest_army_owner
        claims = holder != me and k >= 3 and (holder == -1 or k > g.players[holder].knights)
        lead_vp = g.public_victory_points(C.leader(g, me))
        return self.robber_on_me(g, me) or claims or lead_vp >= g.config.vp_to_win - 3

    def pre_roll(self, g, me, legal):
        out = []
        for a in legal:
            if a.type == A.ROLL:
                out.append((1.0, a))
            elif a.type == A.PLAY_KNIGHT and self.knight_worth(g, me):
                out.append((2.0, a))
        return out

    # ------------------------------------------------------------ discard / trade
    def discard(self, g, me, legal):
        bi = C.board_info(g.layout)
        rv = C.resource_values(g, me, bi)
        gl = C.goals(g, me, bi)
        h = g.players[me].hand
        out = []
        for a in legal:
            hh = h[:]
            hh[a.value] -= 1
            out.append((C.hand_score(hh, gl, rv), a))
        return out

    def trade_delta(self, g, me, give, get, ctx=None):
        """Change in my hand score if I give ``give`` and receive ``get``."""
        rv, gl = ctx or self.hand_ctx(g, me)
        h = g.players[me].hand
        after = [h[i] - give[i] + get[i] for i in range(5)]
        if min(after) < 0:
            return -9.0
        return C.hand_score(after, gl, rv) - C.hand_score(h, gl, rv)

    def hand_ctx(self, g, me):
        bi = C.board_info(g.layout)
        return C.resource_values(g, me, bi), C.goals(g, me, bi)

    def partner_ok(self, g, me, q):
        vp = g.public_victory_points(q)
        return vp < g.config.vp_to_win - 2

    def trade_response(self, g, me, legal):
        tr = g.trade
        out = [(0.0, Action(me, A.REJECT_TRADE))]
        acc = Action(me, A.ACCEPT_TRADE)
        # legal[1] is where the engine lists ACCEPT: skips scanning ~50 counters
        has_acc = (len(legal) > 1 and legal[1] == acc) or acc in legal
        if has_acc and self.partner_ok(g, me, tr["proposer"]):
            d = self.trade_delta(g, me, tr["get"], tr["give"])
            # demand more when the proposer is ahead of me
            margin = 0.04 + 0.02 * max(0, g.public_victory_points(tr["proposer"]) - g.victory_points(me))
            if d > margin:
                out.append((d, acc))
        return out

    def trade_confirm(self, g, me, legal):
        out = [(0.0, Action(me, A.CANCEL_TRADE))]
        for a in legal:
            if a.type == A.CONFIRM_TRADE and self.partner_ok(g, me, a.value):
                give, get = g.trade["responses"][a.value]
                out.append((self.trade_delta(g, me, give, get) - 0.01, a))
        return out

    def offers(self, g, me):
        """Candidate domestic offers (template, my gain), best first."""
        h = g.players[me].hand
        ctx = self.hand_ctx(g, me)
        cands = []
        small = sum(h) <= 7
        for give, get, overpay, gi, gn in _TEMPLATE_META:
            if overpay and small:
                continue  # only overpay when we risk discarding
            if h[gi] >= gn:  # templates give one resource: == all(h[i] >= give[i])
                d = self.trade_delta(g, me, give, get, ctx)
                if d > 0.05:
                    cands.append((d, (give, get)))
        cands.sort(key=lambda x: (-x[0], x[1]))
        return cands

    # ------------------------------------------------------------ main
    def candidate_values(self, g, me, legal, bi=None):
        """[(value, action)] for MAIN-phase actions (excluding domestic offers)."""
        bi = bi or C.board_info(g.layout)
        w = self.w
        out = []
        base = evaluate(g, me, bi, w)
        pl = g.players[me]
        for a in legal:
            t = a.type
            if t == A.END_TURN:
                out.append((base, a))
            elif t == A.MARITIME_TRADE:
                if not g.is_legal(a):  # what c.step(a) would have checked
                    raise IllegalAction(f"illegal action {a}")
                give, get = a.value
                h = pl.hand[:]
                h[give] -= g.trade_rate(me, give)
                h[get] += 1
                out.append((_evaluate_with_hand(g, me, bi, w, h), a))
            elif t in (A.BUILD_CITY, A.BUILD_SETTLEMENT, A.BUILD_ROAD, A.PLAY_YEAR_OF_PLENTY):
                c = g.clone()
                c.step(a)
                out.append((evaluate(c, me, bi, w), a))
            elif t == A.BUY_DEV:
                # value the payment; card itself in expectation
                h = [pl.hand[i] - C.DEV[i] for i in range(5)]
                out.append((_evaluate_with_hand(g, me, bi, w, h) + w["dev_ev"], a))
            elif t == A.PLAY_KNIGHT:
                if self.knight_worth(g, me):
                    c = g.clone()
                    c.step(a)
                    out.append((evaluate(c, me, bi, w) + 0.4, a))
            elif t == A.PLAY_MONOPOLY:
                tot = C.opponents_resource_totals(g, me)
                rv = C.resource_values(g, me, bi)
                gl = C.goals(g, me, bi)
                hh = pl.hand[:]
                hh[a.value] += tot[a.value]
                gain = w["hand"] * (C.hand_score(hh, gl, rv) - C.hand_score(pl.hand, gl, rv))
                if tot[a.value] >= 3:
                    out.append((base + gain - w["dev_held"], a))
            elif t == A.PLAY_ROAD_BUILDING:
                rs = self.road_scores(g, me, g.legal_road_edges(me), bi)
                if any(s > 0 for s, _ in rs):
                    out.append((base + 0.25 - w["dev_held"] + 0.01 * max(s for s, _ in rs), a))
        return out

    def main(self, g, me, legal):
        bi = C.board_info(g.layout)
        nonoffer = [a for a in legal if a.type != A.OFFER_TRADE]
        out = self.candidate_values(g, me, nonoffer, bi)
        best_v, best_a = max(out, key=lambda x: x[0])
        base = next(v for v, a in out if a.type == A.END_TURN)
        # try a domestic offer before settling for a bank trade or ending the turn
        if (best_a.type in (A.END_TURN, A.MARITIME_TRADE) and g.trades_this_turn < self.max_offers
                and any(a.type == A.OFFER_TRADE for a in legal)):
            offers = self.offers(g, me)
            if offers:
                # rotate through candidates across the turn's offers (no repeats)
                d, (give, get) = offers[min(g.trades_this_turn, len(offers) - 1)]
                v = base + self.w["hand"] * d * 0.5
                a = Action(me, A.OFFER_TRADE, (give, get))
                if v > best_v - 1e-9 or best_a.type == A.END_TURN:
                    out.append((best_v + 1e-3, a))
        return out

