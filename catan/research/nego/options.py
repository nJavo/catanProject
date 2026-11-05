"""Negotiation decision points as (state, option set) pairs.

Every trade decision becomes a choice among a small set of options. Each
option is an Action plus a feature vector; the trader heuristic's own choice
is always included and marked (``teacher``), so the same code path serves
behaviour cloning, RL sampling and the heuristic fallback.

Decision kinds:
  offer    MAIN, when the base policy would end the turn or bank-trade and
           offers are left: NONE (keep the base action) or one bundle offer
  respond  TRADE_RESPONSE: reject, accept, or a counter bundle
  confirm  TRADE_CONFIRM: cancel or confirm one response
Everything here is info-safe: it reads only the actor's view plus beliefs.
"""
from __future__ import annotations

from catan.bots.trading import TraderMixin, bundles, negotiation_features, negotiation_feature_names
from catan.engine.actions import A, Action

KINDS = ("none", "offer", "accept", "reject", "counter", "confirm", "cancel")
DECISIONS = ("offer", "respond", "confirm")
OPT_F = len(KINDS) + 5 + 5 + 7   # kind, give5, get5, scalar heuristics
MAX_OFFER_OPTS = 24
MAX_COUNTER_OPTS = 16


def state_scale(n: int = 4) -> list[float]:
    out = []
    for nm in negotiation_feature_names(n):
        if "vp" in nm or "hand_count" in nm:
            out.append(0.1)
        elif "rv" in nm or "near_win" in nm or nm.startswith("phase"):
            out.append(1.0)
        else:
            out.append(0.2)
    return out


def state_vec(g, me, est, decision: str) -> list[float]:
    x = negotiation_features(g, me, est)
    sc = state_scale(g.n)
    v = [a * b for a, b in zip(x, sc)]
    v += [float(decision == d) for d in DECISIONS]
    v += [g.victory_points(me) / 10.0, min(g.turn, 150) / 150.0]
    return v


def opt_vec(kind, give, get, gain, p, ev, danger, pvp, heur):
    """give/get from MY perspective (what I hand over / receive)."""
    x = [float(kind == k) for k in KINDS]
    x += [v / 2.0 for v in give] + [v / 2.0 for v in get]
    x += [gain, p, ev, float(danger), pvp / 10.0, heur, (sum(get) - sum(give)) / 2.0]
    return x


Z5 = (0,) * 5


def offer_options(ag, g, me, base_action):
    """Options for the proposal decision. ``ag`` is a TraderMixin agent."""
    est = ag.beliefs(g, me)
    offers = TraderMixin.trade_offers(ag, g, me)  # [(ev, (give, get))] best first, ev > 0.01
    teacher = 1 if offers else 0
    opts = [(base_action, opt_vec("none", Z5, Z5, 0.0, 1.0, 0.0, False, 0, 0.0), Z5, Z5)]
    ctx = ag.hand_ctx(g, me)
    partners = [q for q in range(g.n) if q != me]
    pvp = max(g.public_victory_points(q) for q in partners)
    for ev, (give, get) in offers[:MAX_OFFER_OPTS]:
        d = ag.trade_delta(g, me, give, get, ctx)
        danger = any(ag.dangerous(g, q, est, recv=give, pay=get) for q in partners)
        p = ev / d if d > 1e-9 else 0.0
        opts.append((Action(me, A.OFFER_TRADE, (give, get)),
                     opt_vec("offer", give, get, d, p, ev, danger, pvp, ev), give, get))
    return opts, teacher, est


def respond_options(ag, g, me, legal):
    tr = g.trade
    p = tr["proposer"]
    est = ag.beliefs(g, me)
    pvp = g.public_victory_points(p)
    scored = TraderMixin.trade_response(ag, g, me, legal)  # heuristic: reject, maybe accept/counters
    best = max(range(len(scored)), key=lambda i: (scored[i][0], -i))
    teacher_action = scored[best][1]
    heur = {a: s for s, a in scored}
    opts = [(Action(me, A.REJECT_TRADE), opt_vec("reject", Z5, Z5, 0.0, 1.0, 0.0, False, pvp, 0.0), Z5, Z5)]
    acc = Action(me, A.ACCEPT_TRADE)
    if acc in legal:
        d = ag.trade_delta(g, me, tr["get"], tr["give"])
        danger = ag.dangerous(g, p, est, recv=tr["get"], pay=tr["give"])
        opts.append((acc, opt_vec("accept", tr["get"], tr["give"], d, 1.0, d, danger, pvp, heur.get(acc, 0.0)),
                     tuple(tr["get"]), tuple(tr["give"])))
    if ag.counters:
        h = g.players[me].hand
        ctx = ag.hand_ctx(g, me)
        lower = list(tr["give"])
        cands = []
        for mine_give, mine_get in bundles(h, 2, 2):
            val = (mine_get, mine_give)
            if val == (tr["give"], tr["get"]):
                continue
            d = ag.trade_delta(g, me, mine_give, mine_get, ctx)
            if d <= 0.0:
                continue
            pa = 0.7 * ag.p_accept(g, p, est, pay=mine_get, recv=mine_give, lower=lower)
            cands.append((d * pa, d, pa, mine_give, mine_get, val))
        cands.sort(key=lambda x: (-x[0], x[5]))
        top = cands[:MAX_COUNTER_OPTS]
        if teacher_action.type == A.COUNTER_OFFER and all(c[5] != teacher_action.value for c in top):
            top += [c for c in cands if c[5] == teacher_action.value]
        for ev, d, pa, mg, mt, val in top:
            a = Action(me, A.COUNTER_OFFER, val)
            danger = ag.dangerous(g, p, est, recv=mg, pay=mt)
            opts.append((a, opt_vec("counter", mg, mt, d, pa, ev, danger, pvp, heur.get(a, 0.0)), mg, mt))
    teacher = next(i for i, o in enumerate(opts) if o[0] == teacher_action)
    return opts, teacher, est


def confirm_options(ag, g, me, legal):
    est = ag.beliefs(g, me)
    scored = TraderMixin.trade_confirm(ag, g, me, legal)
    best = max(range(len(scored)), key=lambda i: (scored[i][0], -i))
    teacher_action = scored[best][1]
    heur = {a: s for s, a in scored}
    opts = [(Action(me, A.CANCEL_TRADE), opt_vec("cancel", Z5, Z5, 0.0, 1.0, 0.0, False, 0, 0.0), Z5, Z5)]
    for a in legal:
        if a.type != A.CONFIRM_TRADE:
            continue
        q = a.value
        give, get = g.trade["responses"][q]
        d = ag.trade_delta(g, me, give, get)
        danger = ag.dangerous(g, q, est, recv=give, pay=get)
        opts.append((a, opt_vec("confirm", give, get, d, 1.0, d, danger, g.public_victory_points(q),
                                heur.get(a, 0.0)), tuple(give), tuple(get)))
    teacher = next(i for i, o in enumerate(opts) if o[0] == teacher_action)
    return opts, teacher, est


STATE_F = len(negotiation_feature_names(4)) + len(DECISIONS) + 2
