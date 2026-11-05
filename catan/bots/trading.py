"""Trading research layer : belief tracking, a partner
acceptance model, danger checks, counter-offers and bundle (non-template)
offers. Used by ``intermediate``/``search`` when built with ``trader=True``.

Information safety: the tracker consumes only events visible to its seat (the
runner filters by ``visible_to``) and reconciles with public counts (hand
sizes, bank). Opponents are valued with their *public* VP.
"""
from __future__ import annotations

import math

from catan.engine.actions import A, Action

from . import common as C
from .intermediate import IntermediateAgent
from .search import SearchAgent


# ------------------------------------------------------------------ beliefs
class BeliefTracker:
    """Expected hand (5 floats) per player from public events.

    Exact while every card movement is public. Unseen steals move the
    victim's expected distribution proportionally. ``estimate`` then runs
    iterative proportional fitting so rows match public hand sizes and
    columns match the public per-resource totals held by opponents
    (19 - bank - my hand). If any step was missed (agent created mid-game,
    host not calling ``observe``) the tracker is marked lost and ``estimate``
    falls back to the max-entropy fit of those public margins."""

    def __init__(self, n: int, me: int):
        self.n, self.me = n, me
        self.E = [[0.0] * 5 for _ in range(n)]
        self.synced = 0          # number of actions whose events were observed
        self.lost = False
        self.unseen_steals = 0

    def observe(self, game, events):
        if self.lost or self.synced + 1 != len(game.action_log):
            self.lost = True
            return
        self.synced += 1
        E = self.E
        for i, e in enumerate(events):
            t = e["type"]
            if t == "resources_gained":
                for r, k in enumerate(e["resources"]):
                    E[e["player"]][r] += k
            elif t == "discarded":
                E[e["player"]][e["resource"]] -= 1
            elif t == "settlement_built" and not e.get("free"):
                self._pay(e["player"], C.SETTLEMENT)
            elif t == "city_built":
                self._pay(e["player"], C.CITY)
            elif t == "road_built" and not e.get("free"):
                self._pay(e["player"], C.ROAD)
            elif t == "dev_bought":
                self._pay(e["player"], C.DEV)
            elif t == "maritime_trade":
                E[e["player"]][e["give"]] -= e["amount"]
                E[e["player"]][e["get"]] += 1
            elif t == "trade_executed":
                p, q = e["player"], e["partner"]
                for r in range(5):
                    E[p][r] += e["get"][r] - e["give"][r]
                    E[q][r] += e["give"][r] - e["get"][r]
            elif t == "monopoly":
                p, r = e["player"], e["resource"]
                for q in range(self.n):
                    if q != p:
                        E[q][r] = 0.0
                E[p][r] += e["amount"]
            elif t == "stolen":
                th, v = e["thief"], e["victim"]
                det = next((d for d in events[i + 1:] if d["type"] == "stolen_detail"
                            and d["thief"] == th and d["victim"] == v), None)
                if det is not None:
                    E[v][det["resource"]] -= 1
                    E[th][det["resource"]] += 1
                else:
                    self.unseen_steals += 1
                    row = [max(0.0, x) for x in E[v]]
                    s = sum(row)
                    for r in range(5):
                        f = row[r] / s if s > 0 else 0.2
                        E[v][r] -= f
                        E[th][r] += f
        for row in E:
            for r in range(5):
                if row[r] < 0:
                    row[r] = 0.0

    def _pay(self, p, cost):
        for r in range(5):
            self.E[p][r] -= cost[r]

    def estimate(self, g) -> list[list[float]]:
        """Expected hands for every seat (mine exact), consistent with public counts."""
        me = self.me
        opps = [q for q in range(g.n) if q != me]
        rows = {q: float(C.hand_count(g, q)) for q in opps}
        cols = [float(x) for x in C.opponents_resource_totals(g, me)]
        M = {q: [(max(0.0, self.E[q][r]) if not self.lost else 0.0) + 1e-3 for r in range(5)] for q in opps}
        key = (len(g.action_log), self.synced, self.lost)
        if getattr(self, "_cache_key", None) == key:
            return [row[:] for row in self._cache]
        for it in range(400):
            for r in range(5):
                s = sum(M[q][r] for q in opps)
                f = cols[r] / s if s > 0 else 0.0
                for q in opps:
                    M[q][r] *= f
            err = 0.0
            for q in opps:
                s = sum(M[q])
                err = max(err, abs(s - rows[q]))
                f = rows[q] / s if s > 0 else 0.0
                M[q] = [x * f for x in M[q]]
            if err < 1e-6:
                break
        out = [None] * g.n
        for q in opps:
            out[q] = M[q]
        out[me] = [float(x) for x in g.players[me].hand]
        self._cache_key, self._cache = key, out
        return [row[:] for row in out]


def prob_at_least(expected: float, n: int, k: int) -> float:
    """P(holding >= k of a resource) for a hand of n cards with the given
    expected count, modelling the hand as n binomial draws."""
    if k <= 0:
        return 1.0
    if n <= 0 or k > n:
        return 0.0
    p = min(1.0, max(0.0, expected / n))
    return max(0.0, 1.0 - sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k)))


def bundles(hand, max_give: int = 2, max_get: int = 2):
    """(give, get) pairs, distinct resources on each side, give <= hand,
    neither side more than one card larger than the other. Includes the 60
    engine templates plus multi-resource bundles."""
    def vecs(total, cap=None):
        out = []
        def rec(i, left, cur):
            if i == 5:
                if left == 0:
                    out.append(tuple(cur))
                return
            for k in range(left + 1):
                if cap is not None and k > cap[i]:
                    break
                cur.append(k)
                rec(i + 1, left - k, cur)
                cur.pop()
        rec(0, total, [])
        return out
    gives = [v for t in range(1, max_give + 1) for v in vecs(t, hand)]
    gets = [v for t in range(1, max_get + 1) for v in vecs(t)]
    out = []
    for gv in gives:
        sg = sum(gv)
        for gt in gets:
            st = sum(gt)
            if abs(sg - st) > 1 or any(gv[i] and gt[i] for i in range(5)):
                continue
            out.append((gv, gt))
    return out


# ------------------------------------------------------------------ trader mixin
class TraderMixin:
    """Overrides IntermediateAgent's trade methods. Kwargs (all optional):
    counters (respond with counter-offers), bundle_offers (non-template),
    danger_vp (refuse build-completing trades with players at >= vp_to_win - danger_vp)."""

    def _trader_init(self, counters=True, bundle_offers=True, danger_vp=3):
        self.counters = counters
        self.bundle_offers = bundle_offers
        self.danger_vp = danger_vp
        self.tracker = None

    # -- belief plumbing
    def reset(self, seat, seed=None):
        super().reset(seat, seed)
        self.tracker = None

    def observe(self, game, events):
        if self.tracker is None:
            self.tracker = BeliefTracker(game.n, self.seat if self.seat is not None else 0)
        self.tracker.observe(game, events)

    def beliefs(self, g, me):
        if self.tracker is None or self.tracker.me != me:
            t = BeliefTracker(g.n, me)
            t.lost = len(g.action_log) > 0
            return t.estimate(g)
        return self.tracker.estimate(g)

    # -- partner model
    def partner_gain(self, g, q, est, pay, recv):
        """Model of q's hand-score change when q pays ``pay`` and receives ``recv``."""
        # rv/goals of q depend only on public state, fixed within one step
        key = (g.layout, len(g.action_log), g.phase)  # g.seed is hidden (it fixes dice and deck order)
        cache = getattr(self, "_pctx", None)
        if cache is None or cache[0] != key:
            cache = self._pctx = (key, {})
        if q not in cache[1]:
            bi = C.board_info(g.layout)
            cache[1][q] = (C.resource_values(g, q, bi, vp=g.public_victory_points(q)), C.goals(g, q, bi))
        rv, gl = cache[1][q]
        h = est[q]
        after = [max(0.0, h[r] - pay[r]) + recv[r] for r in range(5)]
        return C.hand_score(after, gl, rv) - C.hand_score(h, gl, rv)

    def p_accept(self, g, q, est, pay, recv, lower=None):
        n = C.hand_count(g, q)
        h = est[q] if lower is None else [max(est[q][r], lower[r]) for r in range(5)]
        hold = 1.0
        for r in range(5):
            if pay[r]:
                if lower is not None and lower[r] >= pay[r]:
                    continue
                hold *= prob_at_least(h[r], n, pay[r])
        d = self.partner_gain(g, q, est, pay, recv)
        return hold / (1.0 + math.exp(-(d - 0.03) / 0.03))

    def dangerous(self, g, q, est, recv, pay=(0,) * 5):
        """True if q is close to winning and this trade likely completes a build for it."""
        if g.public_victory_points(q) < g.config.vp_to_win - self.danger_vp:
            return False
        h = est[q]
        after = [max(0.0, h[r] - pay[r]) + recv[r] for r in range(5)]
        for cost, _w, name in C.goals(g, q):
            if name not in ("settlement", "city"):
                continue
            before = sum(max(0.0, cost[r] - h[r]) for r in range(5))
            aft = sum(max(0.0, cost[r] - after[r]) for r in range(5))
            if before >= 0.5 and aft < 0.5:
                return True
        return False

    def _offered_this_turn(self, g, me):
        out = set()
        for a in reversed(g.action_log):  # public log, back to the start of this turn
            if a.type == A.END_TURN:
                break
            if a.type == A.OFFER_TRADE and a.player == me:
                out.add(a.value)
        return out

    def trade_offers(self, g, me):
        """[(expected gain, (give, get))], best first."""
        h = g.players[me].hand
        ctx = self.hand_ctx(g, me)
        est = self.beliefs(g, me)
        done = self._offered_this_turn(g, me)
        big = sum(h) > g.config.discard_limit
        if self.bundle_offers:
            cands = bundles(h, 3 if big else 2, 2)
        else:
            from catan.engine.game import TRADE_TEMPLATES
            cands = [x for x in TRADE_TEMPLATES if all(h[i] >= x[0][i] for i in range(5))]
        out = []
        partners = [q for q in range(g.n) if q != me and self.partner_ok(g, me, q)]
        for give, get in cands:
            if (give, get) in done or (sum(give) > sum(get) and not big):
                continue
            d = self.trade_delta(g, me, give, get, ctx)
            if d <= 0.05:
                continue
            miss = 1.0
            for q in partners:
                if not self.dangerous(g, q, est, recv=give, pay=get):
                    miss *= 1.0 - self.p_accept(g, q, est, pay=get, recv=give)
            ev = d * (1.0 - miss)
            if ev > 0.01:
                out.append((ev, (give, get)))
        out.sort(key=lambda x: (-x[0], x[1]))
        return out

    def trade_response(self, g, me, legal):
        tr = g.trade
        p = tr["proposer"]
        out = [(0.0, Action(me, A.REJECT_TRADE))]
        est = self.beliefs(g, me)
        if not self.partner_ok(g, me, p) or self.dangerous(g, p, est, recv=tr["get"], pay=tr["give"]):
            return out
        margin = 0.04 + 0.02 * max(0, g.public_victory_points(p) - g.victory_points(me))
        acc = Action(me, A.ACCEPT_TRADE)
        best = 0.0
        if acc in legal:
            d = self.trade_delta(g, me, tr["get"], tr["give"])
            if d > margin:
                out.append((d, acc))
                best = d
        if self.counters:
            h = g.players[me].hand
            ctx = self.hand_ctx(g, me)
            lower = list(tr["give"])  # the proposer is known to hold what it offered
            for mine_give, mine_get in bundles(h, 2, 2):
                # proposer perspective: proposer gives mine_get, receives mine_give
                val = (mine_get, mine_give)
                if val == (tr["give"], tr["get"]):
                    continue
                d = self.trade_delta(g, me, mine_give, mine_get, ctx)
                if d <= margin or d <= best:
                    continue
                if self.dangerous(g, p, est, recv=mine_give, pay=mine_get):
                    continue
                pa = 0.7 * self.p_accept(g, p, est, pay=mine_get, recv=mine_give, lower=lower)
                if d * pa > max(0.02, best):
                    out.append((d * pa, Action(me, A.COUNTER_OFFER, val)))
        return out

    def trade_confirm(self, g, me, legal):
        est = self.beliefs(g, me)
        out = [(0.0, Action(me, A.CANCEL_TRADE))]
        for a in legal:
            if a.type == A.CONFIRM_TRADE and self.partner_ok(g, me, a.value):
                give, get = g.trade["responses"][a.value]
                if self.dangerous(g, a.value, est, recv=give, pay=get):
                    continue
                out.append((self.trade_delta(g, me, give, get) - 0.01, a))
        return out

    def main(self, g, me, legal):
        bi = C.board_info(g.layout)
        out = self.candidate_values(g, me, [a for a in legal if a.type != A.OFFER_TRADE], bi)
        best_v, best_a = max(out, key=lambda x: x[0])
        if (self.domestic and best_a.type in (A.END_TURN, A.MARITIME_TRADE)
                and g.trades_this_turn < self.max_offers
                and any(a.type == A.OFFER_TRADE for a in legal)):
            offers = self.trade_offers(g, me)
            if offers:
                out.append((best_v + 1e-3, Action(me, A.OFFER_TRADE, offers[0][1])))
        return out


class TraderIntermediateAgent(TraderMixin, IntermediateAgent):
    name = "intermediate_trader"

    def __init__(self, seed=None, counters=True, bundle_offers=True, danger_vp=3, **kw):
        IntermediateAgent.__init__(self, seed, **kw)
        self._trader_init(counters, bundle_offers, danger_vp)


class TraderSearchAgent(TraderMixin, SearchAgent):
    name = "search_trader"

    def __init__(self, seed=None, counters=True, bundle_offers=True, danger_vp=3, **kw):
        SearchAgent.__init__(self, seed, **kw)
        self._trader_init(counters, bundle_offers, danger_vp)

    def main(self, g, me, legal):
        return self._search(g, me, legal, TraderMixin.main(self, g, me, legal))


# ------------------------------------------------------------------ learned-policy features
def negotiation_feature_names(n: int = 4) -> list[str]:
    names = [f"my_hand_{r}" for r in range(5)] + [f"my_rv_{r}" for r in range(5)]
    names += [f"my_missing_{b}" for b in ("settlement", "city", "dev")]
    for k in range(1, n):  # relative seats, clockwise from me
        names += [f"opp{k}_belief_{r}" for r in range(5)]
        names += [f"opp{k}_hand_count", f"opp{k}_public_vp", f"opp{k}_near_win"]
        names += [f"opp{k}_missing_{b}" for b in ("settlement", "city")]
    names += ["offers_left", "phase_trade_response", "phase_trade_confirm"]
    names += [f"pending_give_{r}" for r in range(5)] + [f"pending_get_{r}" for r in range(5)]
    return names


def negotiation_features(g, me: int, est: list[list[float]]) -> list[float]:
    """Fixed-size, info-safe feature vector for a learned negotiation policy
    ``est`` comes from ``BeliefTracker.estimate``.
    Opponents are in relative seat order; absent seats (n < 4) are omitted,
    so the length is ``len(negotiation_feature_names(g.n))``."""
    bi = C.board_info(g.layout)
    h = g.players[me].hand
    rv = C.resource_values(g, me, bi)
    x = [float(v) for v in h] + rv
    x += [float(sum(max(0, c[r] - h[r]) for r in range(5))) for c in (C.SETTLEMENT, C.CITY, C.DEV)]
    for k in range(1, g.n):
        q = (me + k) % g.n
        vp = g.public_victory_points(q)
        x += est[q]
        x += [float(C.hand_count(g, q)), float(vp), float(vp >= g.config.vp_to_win - 3)]
        x += [sum(max(0.0, c[r] - est[q][r]) for r in range(5)) for c in (C.SETTLEMENT, C.CITY)]
    lim = g.config.max_trade_offers_per_turn
    x += [float(lim - g.trades_this_turn) if lim is not None else 3.0,
          float(g.phase == "TRADE_RESPONSE"), float(g.phase == "TRADE_CONFIRM")]
    tr = g.trade
    x += [float(v) for v in (tr["give"] if tr else (0,) * 5)] + [float(v) for v in (tr["get"] if tr else (0,) * 5)]
    return x
