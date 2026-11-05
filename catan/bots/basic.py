"""BasicHeuristicAgent: simple, readable rules.

- Setup: settle on the highest pip sum; road toward the best nearby free spot.
- Main: city > settlement > dev card > road whenever affordable, then a bank or
  port trade toward the next build, then end turn.
- Robber: block the public leader's best hex, steal from the leader.
- Discard: drop what the next build doesn't need.
- Domestic trades: never offers; accepts only if it completes a build it
  couldn't afford before and the proposer is not close to winning.
"""
from __future__ import annotations

import random

from catan.agents.base import Agent
from catan.engine.actions import A, Action, Phase
from catan.engine.board import TOPOLOGY as T

from . import common as C


_PHASE_METHOD = {
    Phase.SETUP_SETTLEMENT: "setup_settlement",
    Phase.SETUP_ROAD: "setup_road",
    Phase.PRE_ROLL: "pre_roll",
    Phase.DISCARD: "discard",
    Phase.MOVE_ROBBER: "move_robber",
    Phase.STEAL: "steal",
    Phase.ROAD_BUILDING: "road_building",
    Phase.TRADE_RESPONSE: "trade_response",
    Phase.TRADE_CONFIRM: "trade_confirm",
    Phase.MAIN: "main",
}


class HeuristicBase(Agent):
    """Phase dispatch shared by the heuristic bots. Each ``_phase`` method
    returns ``[(score, action), ...]``; the highest score wins (ties: first)."""

    name = "heuristic"
    top_k_info = 5

    def __init__(self, seed: int | None = None, domestic: bool = True):
        self.seed = seed
        self.domestic = domestic  # False: never offer, always reject, cancel
        self.rng = random.Random(seed)
        self.seat = None

    def reset(self, seat, seed=None):
        self.seat = seat
        if seed is not None:
            self.rng = random.Random(seed)

    def act(self, game, legal):
        # Same choice as act_with_info, without building the info payload
        # (sorting + to_json on every decision was ~15% of analysis rollouts).
        scored, best = self._choose(game, legal)
        return scored[best][1] if scored else legal[0]

    def _choose(self, game, legal):
        """(legal scored actions, index of the best: highest score, first on ties)."""
        scored = self.score(game, legal)
        # Membership by list scan when that is cheaper than hashing every legal
        # action into a set (few scored actions); same == semantics either way.
        member = legal if len(scored) * len(legal) <= 64 else set(legal)
        # non-template offers/counters are legal but not enumerated by the engine
        scored = [(s, a) for s, a in scored if a in member
                  or (a.type in (A.OFFER_TRADE, A.COUNTER_OFFER) and game.is_legal(a))]
        if not scored:
            return scored, None
        return scored, max(range(len(scored)), key=lambda i: (scored[i][0], -i))

    def act_with_info(self, game, legal, explain=True):
        scored, best = self._choose(game, legal)
        if not scored:
            a = legal[0]
            return a, {"fallback": True}
        top = sorted(scored, key=lambda x: -x[0])[: self.top_k_info]
        info = {"score_kind": "heuristic_score",
                "top": [{"action": a.to_json(), "score": round(float(s), 4)} for s, a in top]}
        if explain:  # read-only decomposition of the scores above; never changes the choice
            from .explain import explain_decision
            try:
                info["explain"] = explain_decision(self, game, legal, scored, scored[best][1])
            except Exception as e:  # an explanation bug must not cost the agent its move
                info["explain_error"] = f"{type(e).__name__}: {e}"
        return scored[best][1], info

    def score(self, g, legal):
        ph = g.phase
        fn = getattr(self, _PHASE_METHOD[ph])  # KeyError for other phases, as before
        if not self.domestic and ph == Phase.TRADE_RESPONSE:
            return [(0.0, Action(g.actor, A.REJECT_TRADE))]
        if not self.domestic and ph == Phase.TRADE_CONFIRM:
            return [(0.0, Action(g.actor, A.CANCEL_TRADE))]
        return fn(g, g.actor, legal)

    # ----------------------------------------------------------- helpers
    def spot_value(self, g, v, me, bi):
        return sum(bi.vertex_pips[v])

    def road_scores(self, g, me, edges, bi):
        """Score each candidate road by the best free spot it brings closer."""
        base = C.road_distances(g, me, 3)
        out = []
        for e in edges:
            best = 0.0
            # vertices newly reachable through e: BFS from e's far end
            for v0 in T.edge_vertices[e]:
                if g.vertex_owner[v0] not in (-1, me):
                    continue
                for v, d in _bfs_from(g, v0, 2).items():
                    if base.get(v, 9) <= d or not C.spot_free(g, v):
                        continue  # e doesn't bring v closer, or v is not buildable
                    x = self.spot_value(g, v, me, bi) * (0.6 ** d)
                    if x > best:  # == max(best, x)
                        best = x
            out.append((best, e))
        return out

    # ----------------------------------------------------------- phases
    def setup_settlement(self, g, me, legal):
        bi = C.board_info(g.layout)
        return [(sum(bi.vertex_pips[a.value]) + 0.01 * sum(1 for x in bi.vertex_pips[a.value] if x), a)
                for a in legal]

    def setup_road(self, g, me, legal):
        bi = C.board_info(g.layout)
        out = []
        for a in legal:
            a_, b_ = T.edge_vertices[a.value]
            u = b_ if a_ == g.last_settlement else a_
            best = 0
            for w in T.vertex_neighbors[u]:
                if w != g.last_settlement and C.spot_free(g, w):
                    best = max(best, self.spot_value(g, w, me, bi))
            out.append((best, a))
        return out

    def robber_on_me(self, g, me):
        vo = g.vertex_owner
        for v in T.hex_vertices[g.robber]:
            if vo[v] == me:
                return True
        return False

    def pre_roll(self, g, me, legal):
        out = []
        for a in legal:
            if a.type == A.ROLL:
                out.append((1.0, a))
            elif a.type == A.PLAY_KNIGHT and self.robber_on_me(g, me):
                out.append((2.0, a))
        return out

    def target(self, g, me):
        """Next build in priority order that is possible at all."""
        pl = g.players[me]
        if pl.settlements and g.supply(me, "city") > 0:
            return C.CITY
        if g.supply(me, "settlement") > 0 and g.legal_settlement_vertices(me):
            return C.SETTLEMENT
        if g.dev_deck:
            return C.DEV
        return C.ROAD

    def discard(self, g, me, legal):
        h = g.players[me].hand
        need = self.target(g, me)
        return [(h[a.value] - need[a.value] + 0.01 * h[a.value], a) for a in legal]

    def hex_block_score(self, g, me, h, bi):
        vo = g.vertex_owner
        hv = T.hex_vertices[h]
        for v in hv:
            if vo[v] == me:
                return -100.0
        lead = None  # computed only if some building is on the hex
        s = 0.0
        for v in hv:
            o = vo[v]
            if o != -1:
                if lead is None:
                    lead = C.leader(g, me)
                s += bi.hex_pips[h] * g.vertex_level[v] * (10 if o == lead else 1)
        return s

    def move_robber(self, g, me, legal):
        bi = C.board_info(g.layout)
        return [(self.hex_block_score(g, me, a.value, bi), a) for a in legal]

    def steal(self, g, me, legal):
        return [(g.public_victory_points(a.value) * 10 + C.hand_count(g, a.value), a) for a in legal]

    def road_building(self, g, me, legal):
        bi = C.board_info(g.layout)
        sc = dict((e, s) for s, e in self.road_scores(g, me, [a.value for a in legal], bi))
        return [(sc[a.value], a) for a in legal]

    def trade_response(self, g, me, legal):
        tr = g.trade
        h = g.players[me].hand
        out = [(0.0, Action(me, A.REJECT_TRADE))]
        acc = Action(me, A.ACCEPT_TRADE)
        # legal[1] is where the engine lists ACCEPT: skips scanning ~50 counters
        has_acc = (len(legal) > 1 and legal[1] == acc) or acc in legal
        if not has_acc or g.public_victory_points(tr["proposer"]) >= g.config.vp_to_win - 2:
            return out
        # I receive the proposer's "give" and pay their "get"
        after = [h[i] + tr["give"][i] - tr["get"][i] for i in range(5)]
        costs = [C.CITY, C.SETTLEMENT, C.DEV]
        before_ok = any(all(h[i] >= c[i] for i in range(5)) for c in costs)
        after_ok = any(all(after[i] >= c[i] for i in range(5)) for c in costs)
        if after_ok and not before_ok and sum(tr["give"]) >= sum(tr["get"]):
            out.append((1.0, acc))
        return out

    def trade_confirm(self, g, me, legal):
        return [(0.0, Action(me, A.CANCEL_TRADE))]

    def main(self, g, me, legal):
        bi = C.board_info(g.layout)
        pl = g.players[me]
        by = {}
        for a in legal:
            t = a.type
            if t == A.OFFER_TRADE:
                continue  # never offers; up to 60 of these per MAIN decision
            grp = by.get(t)
            if grp is None:
                by[t] = [a]
            else:
                grp.append(a)
        # reuse the legal objects (identity fast path in _choose); equal values
        out = [(0.0, by[A.END_TURN][0] if A.END_TURN in by else Action(me, A.END_TURN))]
        for a in by.get(A.BUILD_CITY, []):
            out.append((400 + sum(bi.vertex_pips[a.value]), a))
        for a in by.get(A.BUILD_SETTLEMENT, []):
            out.append((300 + sum(bi.vertex_pips[a.value]), a))
        for a in by.get(A.BUY_DEV, []):
            out.append((200.0, a))
        roads = by.get(A.BUILD_ROAD, [])
        if roads and not g.legal_settlement_vertices(me):
            for (s, e), a in zip(self.road_scores(g, me, [a.value for a in roads], bi), roads):
                if s > 0:
                    out.append((100 + s, a))  # a == Action(me, BUILD_ROAD, e)
        # dev plays
        if by.get(A.PLAY_KNIGHT) and self.robber_on_me(g, me):
            out.append((150.0, by[A.PLAY_KNIGHT][0]))
        if by.get(A.PLAY_ROAD_BUILDING):
            rs = self.road_scores(g, me, g.legal_road_edges(me), bi)
            if any(s > 0 for s, _ in rs):
                out.append((120.0, by[A.PLAY_ROAD_BUILDING][0]))
        tgt = self.target(g, me)
        miss = C.needs(pl.hand, tgt)
        for a in by.get(A.PLAY_YEAR_OF_PLENTY, []):
            r1, r2 = a.value
            m = miss[:]
            got = 0
            for r in (r1, r2):
                if m[r] > 0:
                    m[r] -= 1
                    got += 1
            out.append((50 + 30 * got - 10 * sum(m), a))
        if by.get(A.PLAY_MONOPOLY):
            tot = C.opponents_resource_totals(g, me)
            for a in by[A.PLAY_MONOPOLY]:
                if tot[a.value] >= 3:
                    out.append((40 + tot[a.value] + 5 * (miss[a.value] > 0), a))
        # bank/port trade toward the target: give surplus, get a missing card
        if any(miss):
            for a in by.get(A.MARITIME_TRADE, []):
                give, get = a.value
                if miss[get] > 0:
                    rate = g.trade_rate(me, give)
                    if pl.hand[give] - tgt[give] >= rate:
                        out.append((10 + pl.hand[give] - tgt[give] - rate, a))
        return out


def _bfs_from(g, v0, depth):
    """Road-steps from v0 over free edges (not crossing opponents' buildings)."""
    vo, eo = g.vertex_owner, g.edge_owner
    dist = {v0: 0}
    frontier = [v0]
    for d in range(1, depth + 1):
        nxt = []
        for v in frontier:
            if d > 1 and vo[v] != -1:
                continue
            for e, u in C._V_EDGE_NB[v]:  # same (edge, other end) order as vertex_edges
                if eo[e] == -1 and u not in dist:
                    dist[u] = d
                    nxt.append(u)
        frontier = nxt
    return dist


class BasicHeuristicAgent(HeuristicBase):
    name = "basic"

