"""Decision explanations for the heuristic and search bots (AI lab inspector).

``explain_decision`` decomposes the scores a bot already computed into the
terms that produced them. It is read-only: it works on clones, never touches
the agent's RNG, and the bot's choice is made before it runs. Every number is
recomputed with the bots' own formulas; when a candidate's terms do not add up
to the score the bot reported, the gap is shown as ``unattributed`` instead of
being hidden (tests assert it never appears).

``evaluate_terms`` and ``resource_value_parts`` mirror ``intermediate.evaluate``
and ``common.resource_values`` term by term; tests check the sums match.
The summary sentence is filled in from these numbers by fixed templates.
"""
from __future__ import annotations

import math

from catan.engine.actions import A, Action, Phase
from catan.engine.board import TOPOLOGY as T, GENERIC_PORT, RESOURCES

from . import common as C
from . import provenance as P
from .intermediate import IntermediateAgent

K = 5
EPS = 1e-6
RES = ["wood", "brick", "sheep", "wheat", "ore"]
COSTS = {A.BUILD_ROAD: C.ROAD, A.BUILD_SETTLEMENT: C.SETTLEMENT, A.BUILD_CITY: C.CITY, A.BUY_DEV: C.DEV}
STEP_TYPES = (A.BUILD_CITY, A.BUILD_SETTLEMENT, A.BUILD_ROAD, A.MARITIME_TRADE, A.PLAY_YEAR_OF_PLENTY)
LABEL = {
    "vp": "VP", "production": "production", "diversity": "resource diversity", "ports": "ports",
    "spot0": "buildable spot", "spot1": "spot one road away", "spot2": "spot two roads away",
    "hand": "hand progress", "dev_held": "dev cards held", "knights": "knights played",
    "road_len": "road length", "overflow": "overflow penalty", "dev_ev": "expected card value",
    "knight_bonus": "knight play bonus", "monopoly_gain": "monopoly gain", "dev_spent": "card used up",
    "road_building": "road building bonus", "road_reach": "best road reach", "offer_rule": "offer-first rule",
    "pips_x_scarcity": "pips × scarcity", "spot_diversity": "resource types here",
    "combined_diversity": "types with first settlement", "wood_brick": "wood+brick bonus",
    "wheat_ore": "wheat+ore bonus", "port": "port synergy", "expansion": "expansion room",
    "pips": "pips", "tiebreak": "tie-break", "next_spot": "best next spot", "rule": "rule score",
    "blocked": "blocked production", "victims_bonus": "has a victim with cards", "own_hex": "own building here",
    "desert": "desert", "victim_vp": "victim VP ×10", "victim_cards": "victim cards",
    "hand_after": "hand score after", "surplus": "surplus over next build", "hand_delta": "hand gain",
    "confirm_cost": "confirm cost", "reach": "road reach", "tier": "priority tier", "detail": "within-tier score",
    "game_over": "final result", "unattributed": "unattributed",
}


def _r(x):
    return round(float(x), 4)


def _nz(terms):
    """Rounded terms, zeros dropped to keep the payload small (missing = 0)."""
    return {k: _r(v) for k, v in terms.items() if abs(v) >= 5e-5}


def _fmt(x):
    return f"{x:+.2f}".replace("-", "−")


def action_label(a) -> str:
    t, v = a.type, a.value
    if t == A.MARITIME_TRADE:
        return f"Bank trade {RES[v[0]]} for {RES[v[1]]}"
    if t in (A.OFFER_TRADE, A.COUNTER_OFFER):
        side = lambda x: "+".join(f"{n} {RES[i]}" for i, n in enumerate(x) if n) or "nothing"
        return f"Offer {side(v[0])} for {side(v[1])}"
    if t == A.PLAY_YEAR_OF_PLENTY:
        return f"Year of Plenty {RES[v[0]]}+{RES[v[1]]}"
    if t in (A.PLAY_MONOPOLY, A.DISCARD):
        return f"{'Monopoly on' if t == A.PLAY_MONOPOLY else 'Discard'} {RES[v]}"
    where = {A.BUILD_ROAD: "on edge", A.BUILD_SETTLEMENT: "at", A.BUILD_CITY: "at", A.MOVE_ROBBER: "to hex",
             A.STEAL: "from seat", A.CONFIRM_TRADE: "with seat"}
    name = {A.BUY_DEV: "Buy dev card", A.END_TURN: "End turn", A.MOVE_ROBBER: "Move robber",
            A.CONFIRM_TRADE: "Trade", A.ACCEPT_TRADE: "Accept trade", A.REJECT_TRADE: "Decline trade",
            A.CANCEL_TRADE: "Cancel trade"}.get(t, t.replace("_", " ").capitalize())
    return f"{name} {where[t]} {v}" if t in where else name


# ------------------------------------------------------------------ value model
def evaluate_terms(g, me, bi, w) -> dict:
    """``intermediate.evaluate`` split into its terms (same formulas, same order)."""
    if g.is_over:
        return {"game_over": 100.0 if g.winner == me else float(g.victory_points(me))}
    pl = g.players[me]
    rv = C.resource_values(g, me, bi)
    prod = C.production(g, me, bi)
    t = {"vp": float(g.victory_points(me)),
         "production": w["prod"] * sum(prod[r] * rv[r] for r in range(5)),
         "diversity": w["diversity"] * sum(1 for x in prod if x > 0),
         "ports": sum(w["port3"] if p == GENERIC_PORT else w["port2"] * prod[RESOURCES.index(p)] for p in pl.ports)}
    best = [0.0, 0.0, 0.0]
    if g.supply(me, "settlement") > 0:
        for u, d in C.road_distances(g, me, 2).items():
            if C.spot_free(g, u):
                best[d] = max(best[d], sum(bi.vertex_pips[u][r] * rv[r] for r in range(5)))
    for d in range(3):
        t[f"spot{d}"] = w[f"spot{d}"] * best[d]
    t["hand"] = w["hand"] * C.hand_score(pl.hand, C.goals(g, me, bi), rv)
    t["dev_held"] = w["dev_held"] * (sum(pl.dev) - pl.dev[1])
    t["knights"] = w["knight_played"] * min(pl.knights, 4)
    t["road_len"] = w["road_len"] * min(g.road_length[me], 6) if g.longest_road_owner != me else 0.0
    t["overflow"] = -w["overflow"] * max(0, sum(pl.hand) - g.config.discard_limit)
    return t


def resource_value_parts(g, me, bi) -> list[dict]:
    """``common.resource_values`` split into scarcity x game phase x own-production factor."""
    late = min(1.0, max(0.0, (g.victory_points(me) - 3) / 5))
    phase_w = [1.15 - 0.3 * late, 1.15 - 0.3 * late, 1.0 - 0.2 * late, 1.0 + 0.2 * late, 0.95 + 0.3 * late]
    prod = C.production(g, me, bi)
    rv = C.resource_values(g, me, bi)
    return [{"res": RES[r], "value": _r(rv[r]), "scarcity": _r(bi.scarcity[r]), "phase": _r(phase_w[r]),
             "own_prod": _r(prod[r]), "own_factor": _r(0.6 + 0.6 / (1.0 + prod[r] / 6.0))} for r in range(5)]


def _goals(g, me, bi, hand, rv):
    return [{"name": n, "weight": wt, "cost": list(cost),
             "missing": [max(0, cost[i] - hand[i]) for i in range(5)],
             "missing_value": _r(sum(max(0, cost[i] - hand[i]) * rv[i] for i in range(5)))}
            for cost, wt, n in C.goals(g, me, bi)]


def _affordable(gl, hand):
    return [n for cost, _, n in gl if all(hand[i] >= cost[i] for i in range(5))]


def _main_terms(agent, g, me, a, bi, base, score):
    """(terms, hand after, position evaluated) for one MAIN candidate, following ``candidate_values``."""
    w, pl, t = agent.w, g.players[me], a.type
    if t in STEP_TYPES or t == A.PLAY_KNIGHT or t == A.BUY_DEV:
        c = g.clone()
        if t == A.BUY_DEV:
            c._pay(me, C.DEV)  # same as candidate_values: pay now, card in expectation
        else:
            c.step(a)
        terms = evaluate_terms(c, me, bi, w)
        if t == A.BUY_DEV:
            terms["dev_ev"] = w["dev_ev"]
        elif t == A.PLAY_KNIGHT:
            terms["knight_bonus"] = 0.4
        return terms, c.players[me].hand, c
    terms, hand = dict(base), pl.hand
    if t == A.PLAY_MONOPOLY:
        rv, gl = C.resource_values(g, me, bi), C.goals(g, me, bi)
        hand = pl.hand[:]
        hand[a.value] += C.opponents_resource_totals(g, me)[a.value]
        terms["monopoly_gain"] = w["hand"] * (C.hand_score(hand, gl, rv) - C.hand_score(pl.hand, gl, rv))
        terms["dev_spent"] = -w["dev_held"]
    elif t == A.PLAY_ROAD_BUILDING:
        rs = agent.road_scores(g, me, g.legal_road_edges(me), bi)
        terms.update(road_building=0.25, dev_spent=-w["dev_held"], road_reach=0.01 * max(s for s, _ in rs))
    elif t == A.OFFER_TRADE:
        # main() ranks the offer 0.001 above the best non-offer action; it is a rule, not a valuation
        terms["offer_rule"] = score - sum(base.values())
    return terms, hand, g


def _rank(scored, chosen=None):
    """Unique actions, best score first; the chosen action leads (it wins ties by list order)."""
    best = {}
    for s, a in scored:
        if a not in best or s > best[a]:
            best[a] = s
    return sorted(best.items(), key=lambda x: (x[0] != chosen, -x[1]))


def _search_stats(agent, chosen, heur_ranked):
    vals = getattr(agent, "last_search", None)
    out = {"ran": vals is not None, "rollouts": agent.rollouts, "candidates": agent.candidates,
           "margin": agent.margin, "horizon": agent.horizon, "opp_weight": agent.opp_weight,
           "leaf": f"evaluate(me) - {agent.opp_weight} * max(evaluate(opponent)); evaluate(me) if the game ends",
           "policy": "candidate, then intermediate for the rest of my turn, basic for every other seat; "
                     f"horizon {agent.horizon if agent.horizon is not None else 'one round'} turn(s); "
                     "hidden cards resampled; same determinizations for every candidate"}
    if vals is None:
        if len(heur_ranked) <= 1:
            out["skip_reason"] = "only one candidate"
        elif agent.rollouts <= 0:
            out["skip_reason"] = "rollouts disabled"
        else:
            m = heur_ranked[0][1] - heur_ranked[1][1]
            out["heuristic_margin"] = _r(m)
            out["skip_reason"] = f"heuristic margin {m:.2f} > {agent.margin}"
        return out, {}
    per = {}
    ref = vals.get(chosen, [])
    for a, xs in vals.items():
        n = len(xs)
        mean = sum(xs) / n if n else 0.0
        sd = math.sqrt(sum((x - mean) ** 2 for x in xs) / (n - 1)) if n > 1 else 0.0
        d = [x - y for x, y in zip(xs, ref)]
        dm = sum(d) / n if n else 0.0
        dsd = math.sqrt(sum((x - dm) ** 2 for x in d) / (n - 1)) if n > 1 else 0.0
        per[a] = {"mean": _r(mean), "sd": _r(sd), "n": n, "ci95": _r(1.96 * sd / math.sqrt(n)) if n else 0.0,
                  "diff_vs_chosen": _r(dm), "diff_sd": _r(dsd)}
    return out, per


def _explain_main(agent, g, me, legal, scored, chosen):
    bi = C.board_info(g.layout)
    w = agent.w
    from .search import SearchAgent
    is_search = isinstance(agent, SearchAgent)
    legal_set = set(legal)
    heur = [(s, a) for s, a in IntermediateAgent.main(agent, g, me, legal) if a in legal_set] if is_search else scored
    heur_score = dict((a, s) for a, s in _rank(heur))
    base = evaluate_terms(g, me, bi, w)
    baseline = sum(base.values())
    rv = C.resource_values(g, me, bi)
    gl = C.goals(g, me, bi)
    search, stats = _search_stats(agent, chosen, _rank(heur)) if is_search else (None, {})
    cands, states = [], []
    for a, s in _rank(scored, chosen)[:K]:
        h = heur_score.get(a, s)
        terms, hand, after = _main_terms(agent, g, me, a, bi, base, h)
        states.append(after)
        gap = h - sum(terms.values())
        if abs(gap) > EPS:
            terms["unattributed"] = gap
        c = {"action": a.to_json(), "label": action_label(a), "score": _r(s), "heuristic": _r(h),
             "delta": _r(h - baseline),
             "terms": _nz(terms),
             "deltas": {k: _r(v - base.get(k, 0.0)) for k, v in terms.items() if abs(v - base.get(k, 0.0)) > 5e-5},
             "affordable_after": _affordable(gl, hand)}
        if a.type in COSTS:
            c["cost"] = list(COSTS[a.type])
        if a.type == A.OFFER_TRADE:
            d = next((d for d, t in agent.offers(g, me) if t == a.value), None)
            if d is not None:
                c["detail"] = {"hand_gain_if_accepted": _r(d), "best_other": _r(h - 1e-3)}
        if a in stats:
            c["rollout"] = stats[a]
        cands.append(c)
    out = {"kind": "value_model", "phase": g.phase, "baseline": _r(baseline),
           "baseline_terms": _nz(base), "candidates": cands,
           "goals": _goals(g, me, bi, g.players[me].hand, rv),
           "affordable_now": _affordable(gl, g.players[me].hand),
           "resource_values": resource_value_parts(g, me, bi)}
    if search is not None:
        out["search"] = search
    _add_provenance(agent, g, me, bi, out, states, heur)
    out["summary"] = _summary_main(out)
    out["summary_source"] = "generated from heuristic terms"
    return out


def _knight_ctx(agent, g, me):
    pl = g.players[me]
    k, holder = pl.knights + 1, g.largest_army_owner
    return {"robber_on_me": agent.robber_on_me(g, me),
            "claims_army": holder != me and k >= 3 and (holder == -1 or k > g.players[holder].knights),
            "leader_near_win": g.public_victory_points(C.leader(g, me)) >= g.config.vp_to_win - 3}


def heuristic_choice(agent, g, me):
    """The intermediate heuristic's pick in ``g`` (same scoring and tie-break as act_with_info, no rollouts)."""
    legal = g.legal_actions()
    ok = set(legal)
    sc = [(s, a) for s, a in IntermediateAgent.main(agent, g, me, legal) if a in ok]
    return sc[max(range(len(sc)), key=lambda i: (sc[i][0], -i))][1]


def add_card(g, me, r):
    """Counterfactual copy: one more card of resource r in my hand, taken from the bank."""
    c = g.clone()
    c.bank[r] -= 1
    c.players[me].hand[r] += 1
    return c


def take_vertex(g, me, v):
    """Counterfactual copy: the leading opponent has a settlement on v (placed directly; road
    lengths are not recomputed)."""
    c = g.clone()
    c._place_settlement(C.leader(g, me), v)
    return c


def _counterfactuals(agent, g, me, out):
    h0 = heuristic_choice(agent, g, me)
    res, flips = [], []
    for r in range(5):
        if g.bank[r] > 0:
            a = heuristic_choice(agent, add_card(g, me, r), me)
            if a != h0:
                flips.append({"kind": "add_card", "res": r, "choice": a.to_json(),
                              "text": f"With 1 more {P.RES[r]}, the heuristic would choose {action_label(a)} instead."})
            res.append(r)
    if not flips and res:
        flips.append({"kind": "add_card_none", "checked": res,
                      "text": "One more card of any single resource (" + ", ".join(P.RES[r] for r in res)
                              + ") would not change the heuristic's choice."})
    spot = next((c["why"][k]["inputs"]["vertex"] for c in out["candidates"][:1] for k in ("spot0", "spot1")
                 if k in c.get("why", {}) and c["why"][k]["inputs"].get("vertex") is not None), None)
    if spot is not None and C.spot_free(g, spot):
        a = heuristic_choice(agent, take_vertex(g, me, spot), me)
        q = C.leader(g, me)
        flips.append({"kind": "vertex_taken", "vertex": spot, "by": q, "choice": a.to_json(), "changes": a != h0,
                      "text": f"If seat {q} (the leader) had a settlement on vertex {spot}, the heuristic would "
                              + (f"choose {action_label(a)} instead." if a != h0 else f"still choose {action_label(a)}.")})
    return {"label": "counterfactual: re-scored on a modified copy (heuristic ranking, no rollouts)",
            "baseline_choice": h0.to_json(), "items": flips}


def _why(agent, g, me, a, bi, cand, fa, fb, after):
    """Provenance for the terms of one candidate that differ from the current position."""
    out = {}
    for k, d in cand["deltas"].items():
        p = P.describe(k, fa, fb, after, bi, agent.w, cand["label"])
        if p is None:
            ctx = _knight_ctx(agent, g, me) if k == "knight_bonus" else None
            p = P.describe_analytic(k, agent, g, me, a, bi, cand["terms"].get(k, 0.0), ctx)
        if p is not None:
            out[k] = p
    return out


def _add_provenance(agent, g, me, bi, out, states, heur):
    w = agent.w
    fb = P.facts(g, me, bi, w)
    out["baseline_why"] = {k: P.describe(k, fb, None, g, bi, w) for k in out["baseline_terms"]
                           if P.describe(k, fb, None, g, bi, w) is not None}
    cs = out["candidates"]
    for c, st in list(zip(cs, states))[:2]:  # chosen and runner-up only: keeps the payload small
        a = Action.from_json(c["action"]) if isinstance(c["action"], dict) else c["action"]
        fa = P.facts(st, me, bi, w)
        c["why"] = _why(agent, g, me, a, bi, c, fa, fb, st)
        c["rv_after"] = [_r(x) for x in fa["rv"]]
    for part in out["resource_values"]:
        part["text"] = P.resource_value_text(g, me, bi, part)
    if len(cs) > 1:
        a, b = cs[0], cs[1]
        keys = set(a["deltas"]) | set(b["deltas"])
        diffs = sorted(((k, a["deltas"].get(k, 0.0), b["deltas"].get(k, 0.0)) for k in keys), key=lambda x: -abs(x[1] - x[2]))
        # "side": whose provenance sentence explains the difference (a = chosen, b = runner-up)
        out["decisive"] = [{"term": k, "a": _r(da), "b": _r(db), "diff": _r(da - db),
                            "side": "a" if k in a["why"] else "b" if k in b["why"] else None}
                           for k, da, db in diffs[:3] if abs(da - db) > 5e-4]
    if g.phase == Phase.MAIN and cs:
        out["counterfactuals"] = _counterfactuals(agent, g, me, out)


def _summary_main(e):
    cs = e["candidates"]
    c = cs[0]
    parts = []
    if "cost" in c:
        parts.append("costs " + ", ".join(f"{n} {RES[i]}" for i, n in enumerate(c["cost"]) if n))  # "2 grain, 3 ore", not "grain+grain+ore+ore+ore"
    top = sorted(c["deltas"].items(), key=lambda kv: -abs(kv[1]))[:3]
    parts += [f"{LABEL.get(k, k)} {_fmt(v)}" for k, v in top]
    if c["action"]["type"] == A.END_TURN:
        s = "End turn: no candidate scored above the current position"
    elif c["action"]["type"] == A.OFFER_TRADE and "detail" in c:
        s = (f"{c['label']}: domestic offers are tried before ending the turn or trading with the bank (rule); "
             f"hand gain if accepted {_fmt(c['detail']['hand_gain_if_accepted'])}")
    else:
        s = f"{c['label']}: " + "; ".join(parts) + f"; net {_fmt(c['delta'])} vs ending turn"
    if len(cs) > 1:
        s += f". Runner-up: {cs[1]['label']} ({_fmt(cs[1]['delta'])})"
    s += "."
    sr = e.get("search")
    if sr is not None:
        if not sr["ran"]:
            s += f" Search skipped: {sr['skip_reason']}."
        else:
            r = c["rollout"]
            s += f" Rollouts (n={r['n']}): mean {r['mean']:.2f} ± {r['ci95']:.2f}"
            if len(cs) > 1 and "rollout" in cs[1]:
                r2 = cs[1]["rollout"]
                s += f" vs runner-up {r2['mean']:.2f} ± {r2['ci95']:.2f} (paired gap {_fmt(-r2['diff_vs_chosen'])})"
            hr = sorted(cs, key=lambda x: -x["heuristic"]).index(c) + 1
            if hr > 1:
                s += f"; the heuristic alone ranked it #{hr}"
            s += "."
    return s


# ------------------------------------------------------------------ other phases
def _setup_settlement(agent, g, me, a, bi):
    pips = bi.vertex_pips[a.value]
    if not isinstance(agent, IntermediateAgent):
        return {"pips": float(sum(pips)), "tiebreak": 0.01 * sum(1 for x in pips if x)}, None
    w = agent.w
    have = [sum(bi.vertex_pips[s][r] for s in g.players[me].settlements) for r in range(5)]
    tot = [have[r] + pips[r] for r in range(5)]
    port = bi.vertex_port.get(a.value)
    exp = 0.0
    v = a.value
    for u in T.vertex_neighbors[v]:
        for x in T.vertex_neighbors[u]:
            if x != v and C.spot_free(g, x) and x not in T.vertex_neighbors[v]:
                exp = max(exp, sum(bi.vertex_pips[x]))
    t = {"pips_x_scarcity": sum(pips[r] * bi.scarcity[r] for r in range(5)),
         "spot_diversity": w["setup_div"] * sum(1 for x in pips if x),
         "combined_diversity": 1.0 * sum(1 for x in tot if x),
         "wood_brick": 1.0 if tot[0] and tot[1] else 0.0,
         "wheat_ore": 0.8 if tot[3] and tot[4] else 0.0,
         "port": 1.0 if port == GENERIC_PORT else (0.25 * tot[RESOURCES.index(port)] if port else 0.0),
         "expansion": w["setup_exp"] * exp}
    return t, {"pips": list(pips), "port": port}


def _hex(agent, g, me, a, bi):
    h = a.value
    if any(g.vertex_owner[v] == me for v in T.hex_vertices[h]):
        return {"own_hex": -100.0}, None
    inter = isinstance(agent, IntermediateAgent)
    r = bi.hex_res[h]
    if inter and r < 0:
        return {"desert": -50.0}, None
    lead = C.leader(g, me)
    blocked, victims, rows = 0.0, False, []
    for v in T.hex_vertices[h]:
        o = g.vertex_owner[v]
        if o == -1:
            continue
        if inter:
            f = 1 + 0.3 * g.public_victory_points(o)
            x = bi.hex_pips[h] * g.vertex_level[v] * bi.scarcity[r] * f
            victims = victims or C.hand_count(g, o) > 0
        else:
            f = 10 if o == lead else 1
            x = bi.hex_pips[h] * g.vertex_level[v] * f
        blocked += x
        rows.append({"seat": o, "vp": g.public_victory_points(o), "pips": bi.hex_pips[h], "level": g.vertex_level[v],
                     "leader_factor": _r(f), "value": _r(x)})
    t = {"blocked": blocked}
    if inter:
        t["victims_bonus"] = 2.0 if victims else 0.0
    return t, {"victims": rows, "scarcity": _r(bi.scarcity[r]) if inter and r >= 0 else None}


def _discard(agent, g, me, a, bi):
    h = g.players[me].hand
    if isinstance(agent, IntermediateAgent):
        hh = h[:]
        hh[a.value] -= 1
        return {"hand_after": C.hand_score(hh, C.goals(g, me, bi), C.resource_values(g, me, bi))}, {"hand_after": hh}
    need = agent.target(g, me)
    return {"surplus": float(h[a.value] - need[a.value]), "tiebreak": 0.01 * h[a.value]}, None


def _road(agent, g, me, a, bi):
    if g.phase == Phase.SETUP_ROAD:
        a_, b_ = T.edge_vertices[a.value]
        u = b_ if a_ == g.last_settlement else a_
        best, at = 0.0, None
        for x in T.vertex_neighbors[u]:
            if x != g.last_settlement and C.spot_free(g, x) and agent.spot_value(g, x, me, bi) > best:
                best, at = agent.spot_value(g, x, me, bi), x
        return {"next_spot": float(best)}, {"toward_vertex": at}
    return {"reach": dict((e, s) for s, e in agent.road_scores(g, me, [a.value], bi))[a.value]}, None


def _steal(agent, g, me, a, bi):
    return {"victim_vp": 10.0 * g.public_victory_points(a.value), "victim_cards": float(C.hand_count(g, a.value))}, None


def _trade_confirm(agent, g, me, a, bi):
    if a.type != A.CONFIRM_TRADE:
        return {"rule": 0.0}, None
    give, get = g.trade["responses"][a.value]
    return {"hand_delta": agent.trade_delta(g, me, give, get), "confirm_cost": -0.01}, None


def _basic_main(agent, g, me, a, bi, score):
    tier = {A.BUILD_CITY: 400, A.BUILD_SETTLEMENT: 300, A.BUY_DEV: 200, A.PLAY_KNIGHT: 150, A.PLAY_ROAD_BUILDING: 120,
            A.BUILD_ROAD: 100, A.PLAY_YEAR_OF_PLENTY: 50, A.PLAY_MONOPOLY: 40, A.MARITIME_TRADE: 10}.get(a.type, 0)
    return {"tier": float(tier), "detail": score - tier}, None


PHASE_TERMS = {Phase.SETUP_SETTLEMENT: _setup_settlement, Phase.SETUP_ROAD: _road, Phase.ROAD_BUILDING: _road,
               Phase.MOVE_ROBBER: _hex, Phase.DISCARD: _discard, Phase.STEAL: _steal}


def _context(agent, g, me):
    """Phase-level facts behind rule-based choices (trades, knights)."""
    inter = isinstance(agent, IntermediateAgent)
    if g.phase == Phase.TRADE_RESPONSE:
        tr = g.trade
        if inter:
            d = agent.trade_delta(g, me, tr["get"], tr["give"])
            margin = 0.04 + 0.02 * max(0, g.public_victory_points(tr["proposer"]) - g.victory_points(me))
            rv, gl = agent.hand_ctx(g, me)
            h = g.players[me].hand
            after = [h[i] - tr["get"][i] + tr["give"][i] for i in range(5)]
            pv, mv = g.public_victory_points(tr["proposer"]), g.victory_points(me)
            ok = agent.partner_ok(g, me, tr["proposer"])
            text = (f"Seat {tr['proposer']} offers {P.cards(tr['give'])} for your {P.cards(tr['get'])}. "
                    + (f"Hand score {P.num(C.hand_score(h, gl, rv))} \u2192 {P.num(C.hand_score(after, gl, rv))} (gain {P.sg(d)}) "
                       if min(after) >= 0 else "You can't pay it, ")
                    + f"vs required margin {P.num(margin)} = 0.04 + 0.02 \u00d7 max(0, their public VP {pv} \u2212 your VP {mv}). "
                    + ("Accepted only if the gain is larger." if ok else
                       f"Seat {tr['proposer']} has {pv} public VP, within 2 of winning, so any trade is refused."))
            return {"rule": "accept only if hand gain > margin and the proposer is not within 2 VP of winning",
                    # DS31-03: -9 is trade_delta's can't-pay sentinel, not a hand score; don't show it as one
                    "hand_delta": _r(d) if min(after) >= 0 else "can't pay", "margin": _r(margin), "partner_ok": ok, "text": text}
        h = g.players[me].hand
        after = [h[i] + tr["give"][i] - tr["get"][i] for i in range(5)]
        ok = lambda hh: any(all(hh[i] >= c[i] for i in range(5)) for c in (C.CITY, C.SETTLEMENT, C.DEV))
        return {"rule": "accept only if it completes a city, settlement or dev card, is not a loss in card count, "
                        "and the proposer is not within 2 VP of winning",
                "completes_build": ok(after) and not ok(h), "fair_count": sum(tr["give"]) >= sum(tr["get"]),
                "partner_ok": g.public_victory_points(tr["proposer"]) < g.config.vp_to_win - 2}
    if g.phase == Phase.PRE_ROLL:
        ctx = {"robber_on_me": agent.robber_on_me(g, me)}
        if inter:
            pl = g.players[me]
            k, holder = pl.knights + 1, g.largest_army_owner
            ctx["claims_army"] = holder != me and k >= 3 and (holder == -1 or k > g.players[holder].knights)
            ctx["leader_near_win"] = g.public_victory_points(C.leader(g, me)) >= g.config.vp_to_win - 3
        ctx["rule"] = "play a knight before rolling if any of these hold, else roll"
        return ctx
    return None


def _rules_text(agent, g, me, a, bi, terms, detail) -> dict:
    """Provenance sentence and board refs for a non-MAIN candidate."""
    inter = isinstance(agent, IntermediateAgent)
    if g.phase == Phase.MOVE_ROBBER:
        vs = [v for v in T.hex_vertices[a.value] if g.vertex_owner[v] != -1]
        return {"text": P.robber_text(g, me, bi, a, terms, detail), "refs": {"hexes": [a.value], "vertices": vs}}
    if g.phase == Phase.STEAL:
        q = a.value
        return {"text": f"Seat {q}: {g.public_victory_points(q)} public VP \u00d7 10 + {C.hand_count(g, q)} cards in hand."}
    if g.phase == Phase.DISCARD and inter:
        gs = P.goal_scores(detail["hand_after"], C.goals(g, me, bi), C.resource_values(g, me, bi))
        return {"text": f"Discarding {P.RES[a.value]} leaves {P.cards(detail['hand_after'])}: closest goal "
                        + (P.goal_txt(gs[0]) if gs else "none") + f"; hand score {P.num(terms['hand_after'])}."}
    if g.phase == Phase.SETUP_SETTLEMENT and inter:
        v = a.value
        hx = P.vertex_hexes(g, bi, v)
        text = (f"Vertex {v}: " + (", ".join(P.hex_txt(x) for x in hx) or "no resource hexes")
                + f". Pips \u00d7 scarcity {P.num(terms['pips_x_scarcity'])}; {sum(1 for x in detail['pips'] if x)} resource "
                f"type(s) \u00d7 {agent.w['setup_div']} = {P.num(terms['spot_diversity'])}; with your first settlement "
                f"{P.num(terms['combined_diversity'])} type(s)")
        if terms["wood_brick"]:
            text += "; wood and brick together +1.0"
        if terms["wheat_ore"]:
            text += "; wheat and ore together +0.8"
        if terms["port"]:
            text += f"; port {detail['port']} {P.sg(terms['port'])}"
        text += f"; best free spot two steps away \u00d7 {agent.w['setup_exp']} = {P.num(terms['expansion'])}."
        return {"text": text, "refs": {"vertices": [v], "hexes": [x["hex"] for x in hx]}}
    if g.phase in (Phase.SETUP_ROAD, Phase.ROAD_BUILDING):
        tv = detail.get("toward_vertex")
        return {"refs": {"edges": [a.value], "vertices": [tv] if tv is not None else []}}
    return {}


def _explain_rules(agent, g, me, scored, chosen):
    bi = C.board_info(g.layout)
    fn = PHASE_TERMS.get(g.phase)
    if g.phase == Phase.TRADE_CONFIRM and isinstance(agent, IntermediateAgent):
        fn = _trade_confirm
    cands = []
    for a, s in _rank(scored, chosen)[:K]:
        if g.phase == Phase.MAIN:
            terms, detail = _basic_main(agent, g, me, a, bi, s)
        elif fn is not None:
            terms, detail = fn(agent, g, me, a, bi)
        elif a.type == A.ACCEPT_TRADE and isinstance(agent, IntermediateAgent):
            tr = g.trade
            terms, detail = {"hand_delta": agent.trade_delta(g, me, tr["get"], tr["give"])}, None
        else:
            terms, detail = {"rule": float(s)}, None
        gap = s - sum(terms.values())
        if abs(gap) > EPS:
            terms["unattributed"] = gap
        c = {"action": a.to_json(), "label": action_label(a), "score": _r(s), "terms": _nz(terms)}
        if detail:
            c["detail"] = detail
        c.update(_rules_text(agent, g, me, a, bi, terms, detail or {}))
        cands.append(c)
    out = {"kind": "value_model" if g.phase in PHASE_TERMS and isinstance(agent, IntermediateAgent) else "rules",
           "phase": g.phase, "candidates": cands}
    ctx = _context(agent, g, me)
    if ctx:
        out["context"] = ctx
    c = cands[0]
    top = sorted(c["terms"].items(), key=lambda kv: -abs(kv[1]))[:3]
    s = f"{c['label']}: score {c['score']:.2f}"
    if top:  # a zero-score default (e.g. declining a trade) has no terms to list (DS2-10)
        s += " (" + ", ".join(f"{LABEL.get(k, k)} {v:.2f}" for k, v in top) + ")"
    if len(cands) > 1:
        s += f". Runner-up: {cands[1]['label']} ({cands[1]['score']:.2f})"
    if ctx and "hand_delta" in ctx:
        s += (". Can't pay it" if isinstance(ctx["hand_delta"], str) else f". Hand gain {_fmt(ctx['hand_delta'])} vs required {_fmt(ctx['margin'])}") + ("" if ctx["partner_ok"] else "; proposer is close to winning")
    elif ctx and "completes_build" in ctx:
        s += f". Completes a build: {'yes' if ctx['completes_build'] else 'no'}" + ("" if ctx["partner_ok"] else "; proposer is close to winning")
    out["summary"] = s + "."
    out["summary_source"] = "generated from heuristic terms"
    return out


def explain_decision(agent, g, legal, scored, chosen) -> dict:
    """JSON-safe explanation of a decision whose candidate scores are ``scored``."""
    me = g.actor
    if g.phase == Phase.MAIN and isinstance(agent, IntermediateAgent):
        return _explain_main(agent, g, me, legal, scored, chosen)
    return _explain_rules(agent, g, me, scored, chosen)
