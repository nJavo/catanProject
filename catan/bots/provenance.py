"""Plain-language provenance for the value model's terms (used by explain.py).

``facts`` collects the inputs behind every ``intermediate.evaluate`` term in
one position: which buildings and hexes produce, which free spot is the best
at each road distance, which goal the hand is closest to, and so on.
``describe`` turns the facts before and after a candidate action into a
sentence, the board references it mentions, and the exact inputs. Tests
recompute each term from ``inputs``, so the sentences cannot drift from the
numbers. Nothing here is a guess: every value comes from the game state and
the bots' own formulas.
"""
from __future__ import annotations

from catan.engine.board import TOPOLOGY as T, GENERIC_PORT, RESOURCES
from catan.engine.game import BANK_START

from . import common as C

RES = ["wood", "brick", "sheep", "wheat", "ore"]
DIST = ["buildable now", "one road away", "two roads away"]


def _r(x):
    return round(float(x), 4)


def num(x, d=2):
    return f"{x:.{d}f}".replace("-", "−")


def sg(x, d=2):
    return f"{x:+.{d}f}".replace("-", "−")


def cards(v) -> str:
    return " + ".join(f"{n} {RES[i]}" if n > 1 else RES[i] for i, n in enumerate(v) if n) or "nothing"


def hex_info(g, bi, h) -> dict:
    r = bi.hex_res[h]
    return {"hex": h, "res": r, "num": g.layout.hex_number[h], "pips": bi.hex_pips[h], "robbed": h == g.robber}


def hex_txt(x) -> str:
    return f"{RES[x['res']]} {x['num']} ({x['pips']} pip{'s' if x['pips'] != 1 else ''})"


def vertex_hexes(g, bi, v) -> list[dict]:
    return [hex_info(g, bi, h) for h in T.vertex_hexes[v] if bi.hex_res[h] >= 0]


def best_spots(g, me, bi, rv):
    """Best free spot per road distance, iterated exactly like ``evaluate`` (first strict maximum)."""
    best, at = [0.0, 0.0, 0.0], [None, None, None]
    if g.supply(me, "settlement") > 0:
        for u, d in C.road_distances(g, me, 2).items():
            if C.spot_free(g, u):
                s = sum(bi.vertex_pips[u][r] * rv[r] for r in range(5))
                if s > best[d]:
                    best[d], at[d] = s, u
    return [{"vertex": at[d], "value": best[d], "pips": list(bi.vertex_pips[at[d]]) if at[d] is not None else [0] * 5}
            for d in range(3)]


def goal_scores(hand, gl, rv):
    out = []
    for cost, wt, name in gl:
        miss = [max(0, cost[i] - hand[i]) for i in range(5)]
        m = sum(miss[i] * rv[i] for i in range(5))
        out.append({"name": name, "weight": wt, "missing": miss, "missing_value": m, "score": wt / (1.0 + 1.3 * m)})
    return sorted(out, key=lambda x: -x["score"])


def facts(g, me, bi, w) -> dict:
    pl = g.players[me]
    rv = C.resource_values(g, me, bi)
    prod = C.production(g, me, bi)
    hand = list(pl.hand)
    gs = goal_scores(hand, C.goals(g, me, bi), rv)
    return {
        "me": me, "rv": rv, "prod": prod, "hand": hand, "goals": gs,
        "buildings": [{"v": v, "level": g.vertex_level[v], "hexes": vertex_hexes(g, bi, v)}
                      for v in pl.settlements + pl.cities],
        "spots": best_spots(g, me, bi, rv),
        "ports": sorted(pl.ports),
        "vp": {"settlements": len(pl.settlements), "cities": len(pl.cities), "longest_road": g.longest_road_owner == me,
               "largest_army": g.largest_army_owner == me, "vp_cards": pl.dev[1]},
        "dev_held": sum(pl.dev) - pl.dev[1], "knights": pl.knights, "road": g.road_length[me],
        "holds_lr": g.longest_road_owner == me,
    }


def _hand_parts(f):
    gs, hand, rv = f["goals"], f["hand"], f["rv"]
    best = gs[0]["score"] if gs else 0.0
    second = gs[1]["score"] if len(gs) > 1 else 0.0
    cv = sum(hand[i] * rv[i] for i in range(5))
    over = max(0, sum(hand) - 7)
    return best, second, cv, over, best + 0.4 * second + 0.03 * cv - 0.06 * over


def goal_txt(x):
    miss = "nothing" if not any(x["missing"]) else cards(x["missing"])
    return (f"{x['name']} (weight {x['weight']}) missing {miss} (value {num(x['missing_value'])}) "
            f"→ {x['weight']}/(1+1.3·{num(x['missing_value'])}) = {num(x['score'])}")


def _spot_txt(s, d, w, g, bi):
    if s["vertex"] is None:
        return "none (0)"
    hx = " + ".join(hex_txt(x) for x in vertex_hexes(g, bi, s["vertex"]))
    return f"vertex {s['vertex']}: {hx} \u2192 value-weighted {num(s['value'])} \u00d7 {w} = {sg(w * s['value'])}"


def _spot_hex_refs(g, bi, v):
    return [h for h in T.vertex_hexes[v] if bi.hex_res[h] >= 0] if v is not None else []


def describe(term, fa, fb, g_after, bi, w, label=None) -> dict | None:
    """Sentence, board refs and exact inputs for one evaluate() term. ``fb`` is the position before
    the action (None when describing the current position itself)."""
    if term == "game_over":
        won = g_after.winner == fa["me"]
        return {"text": "This move ends the game with you as the winner: a won position is worth a flat 100."
                if won else "This move ends the game; the position is worth your final VP.", "refs": {}, "inputs": {}}
    rv = fa["rv"]
    refs: dict = {}
    if term.startswith("spot"):
        d = int(term[-1])
        a, b = fa["spots"][d], fb["spots"][d] if fb else None
        wt = w[term]
        inputs = {"vertex": a["vertex"], "pips": a["pips"], "weight": wt}
        if a["vertex"] is not None:
            refs = {"vertices": [a["vertex"]], "hexes": _spot_hex_refs(g_after, bi, a["vertex"])}
        if b is None:
            text = f"Best free spot {DIST[d]}: {_spot_txt(a, d, wt, g_after, bi)}."
        elif a["vertex"] != b["vertex"] and a["vertex"] is not None:
            text = f"{label} makes vertex {a['vertex']} the best spot {DIST[d]}: {_spot_txt(a, d, wt, g_after, bi)}. Before: {_spot_txt(b, d, wt, g_after, bi)}."
        elif a["vertex"] is None and b["vertex"] is not None:
            gone = "is no longer free" if not C.spot_free(g_after, b["vertex"]) else "is no longer at this distance"
            text = f"After {label.lower()}, vertex {b['vertex']} {gone}, and no free spot is {DIST[d]} (0). Before: {_spot_txt(b, d, wt, g_after, bi)}."
            refs = {"vertices": [b["vertex"]]}
        else:
            text = f"Best spot {DIST[d]} stays {_spot_txt(a, d, wt, g_after, bi)}" + (
                f" (was {num(b['value'])} with the old resource values)." if abs(a["value"] - b["value"]) > 5e-5 else ".")
        return {"text": text, "refs": refs, "inputs": inputs}
    if term == "hand":
        best, second, cv, over, hs = _hand_parts(fa)
        inputs = {"hand": fa["hand"], "weight": w["hand"],
                  "goals": [{"name": x["name"], "weight": x["weight"], "missing": x["missing"]} for x in fa["goals"]]}
        now = f"closest goal {goal_txt(fa['goals'][0])}" if fa["goals"] else "no goals"
        tail = (f"Hand score = {num(best)} + 0.4×{num(second)} (second goal) + 0.03×{num(cv)} (card value)"
                + (f" − 0.06×{over} (over 7 cards)" if over else "") + f" = {num(hs)}; × {w['hand']} = {sg(w['hand'] * hs)}.")
        if fb is None:
            text = f"Hand {cards(fa['hand'])}: {now}. {tail}"
        else:
            was = f"closest goal was {goal_txt(fb['goals'][0])}" if fb["goals"] else "no goals"
            text = f"Hand after: {cards(fa['hand'])}, {now}. Before ({cards(fb['hand'])}): {was}. {tail}"
        return {"text": text, "refs": refs, "inputs": inputs}
    if term == "production":
        hexes, robbed = [], []
        for b in fa["buildings"]:
            for x in b["hexes"]:
                hexes.append(x["hex"])
                if x["robbed"] and x not in robbed:
                    robbed.append(x)
        parts = [f"{RES[r]} {fa['prod'][r]:g} pips × {num(rv[r])}" for r in range(5) if fa["prod"][r]]
        s = sum(fa["prod"][r] * rv[r] for r in range(5))
        text = ("Production: " + (" + ".join(parts) or "nothing") +
                f" = {num(s)} value-weighted pips × {w['prod']} = {sg(w['prod'] * s)}.")
        if fb is not None:
            old = {(b["v"], b["level"]) for b in fb["buildings"]}
            for b in fa["buildings"]:
                if (b["v"], b["level"]) not in old:
                    kind = "city" if b["level"] == 2 else "settlement"
                    text += f" New {kind} at vertex {b['v']} adds " + ", ".join(
                        hex_txt(x) + (" ×2" if b["level"] == 2 else "") for x in b["hexes"]) + "."
                    refs = {"vertices": [b["v"]], "hexes": [x["hex"] for x in b["hexes"]]}
        for x in robbed:
            text += f" Hex {x['hex']} ({RES[x['res']]} {x['num']}) has the robber, so its {x['pips']} pips don't count."
        refs.setdefault("hexes", sorted(set(hexes)))
        inputs = {"weight": w["prod"],
                  "buildings": [{"level": b["level"], "hexes": [[x["res"], x["pips"], x["robbed"]] for x in b["hexes"]]}
                                for b in fa["buildings"]]}
        return {"text": text, "refs": refs, "inputs": inputs}
    if term == "diversity":
        have = [RES[r] for r in range(5) if fa["prod"][r] > 0]
        miss = [RES[r] for r in range(5) if fa["prod"][r] <= 0]
        text = f"You produce {len(have)} of 5 resources ({', '.join(have) or 'none'}" + (
            f"; not {', '.join(miss)}" if miss else "") + f") × {w['diversity']} = {sg(w['diversity'] * len(have))}."
        if fb is not None:
            new = [RES[r] for r in range(5) if fa["prod"][r] > 0 >= fb["prod"][r]]
            if new:
                text += f" New: {', '.join(new)}."
        return {"text": text, "refs": refs, "inputs": {"prod": [_r(x) for x in fa["prod"]], "weight": w["diversity"]}}
    if term == "ports":
        parts = []
        for p in fa["ports"]:
            if p == GENERIC_PORT:
                parts.append(f"3:1 port {sg(w['port3'])}")
            else:
                r = RESOURCES.index(p)
                parts.append(f"2:1 {RES[r]} port × {fa['prod'][r]:g} {RES[r]} pips × {w['port2']} = {sg(w['port2'] * fa['prod'][r])}")
        return {"text": "Ports: " + ("; ".join(parts) or "none") + ".", "refs": refs, "inputs": {"ports": fa["ports"]}}
    if term == "vp":
        v = fa["vp"]
        parts = [f"{v['settlements']} settlement(s)", f"{v['cities']} city ×2" if v["cities"] else None,
                 "Longest Road +2" if v["longest_road"] else None, "Largest Army +2" if v["largest_army"] else None,
                 f"{v['vp_cards']} VP card(s)" if v["vp_cards"] else None]
        tot = v["settlements"] + 2 * v["cities"] + 2 * v["longest_road"] + 2 * v["largest_army"] + v["vp_cards"]
        return {"text": f"VP {tot}: " + " + ".join(p for p in parts if p) + ".", "refs": refs, "inputs": v}
    if term == "dev_held":
        return {"text": f"{fa['dev_held']} unplayed non-VP dev card(s) × {w['dev_held']} = {sg(w['dev_held'] * fa['dev_held'])}.",
                "refs": refs, "inputs": {"count": fa["dev_held"], "weight": w["dev_held"]}}
    if term == "knights":
        k = min(fa["knights"], 4)
        return {"text": f"{fa['knights']} knight(s) played (counted up to 4) × {w['knight_played']} = {sg(w['knight_played'] * k)}.",
                "refs": refs, "inputs": {"count": fa["knights"], "weight": w["knight_played"]}}
    if term == "road_len":
        if fa["holds_lr"]:
            text = "You hold Longest Road, so road length adds nothing here (its 2 VP are in VP)."
        else:
            text = f"Longest road {fa['road']} (counted up to 6) × {w['road_len']} = {sg(w['road_len'] * min(fa['road'], 6))}."
        return {"text": text, "refs": refs, "inputs": {"length": fa["road"], "holds_lr": fa["holds_lr"], "weight": w["road_len"]}}
    return None


def describe_analytic(term, agent, g, me, a, bi, value, ctx) -> dict | None:
    """Sentences for terms candidate_values adds on top of evaluate()."""
    w = agent.w
    if term == "dev_ev":
        return {"text": f"An unknown dev card is valued at a flat expected {w['dev_ev']}. "
                        "Its cost (sheep + wheat + ore) shows up in hand progress.", "refs": {}, "inputs": {}}
    if term == "knight_bonus":
        k = ctx or {}
        why = [s for s, on in (("the robber is on your hex", k.get("robber_on_me")),
                                ("it claims Largest Army", k.get("claims_army")),
                                ("the leader is within 3 VP of winning", k.get("leader_near_win"))) if on]
        return {"text": f"Playing a knight adds a flat 0.4 because {' and '.join(why) or 'the rule fired'}.", "refs": {}, "inputs": k}
    if term == "monopoly_gain":
        r = a.value
        tot = C.opponents_resource_totals(g, me)[r]
        h0 = list(g.players[me].hand)
        h1 = h0[:]
        h1[r] += tot
        rv, gl = C.resource_values(g, me, bi), C.goals(g, me, bi)
        s0, s1 = C.hand_score(h0, gl, rv), C.hand_score(h1, gl, rv)
        return {"text": f"Opponents hold {tot} {RES[r]} in total (public: {BANK_START} − bank {g.bank[r]} − yours {h0[r]}). "
                        f"Taking them changes hand score {num(s0)} → {num(s1)}; × {w['hand']} = {sg(value)}.",
                "refs": {}, "inputs": {"taken": tot}}
    if term == "dev_spent":
        return {"text": f"The card is used up, so it no longer counts as a held dev card ({sg(-w['dev_held'])}).", "refs": {}, "inputs": {}}
    if term == "road_building":
        return {"text": "Road Building adds a flat 0.25 (two free roads).", "refs": {}, "inputs": {}}
    if term == "road_reach":
        return {"text": f"Plus 0.01 × the best road's reach score ({num(value / 0.01)}).", "refs": {}, "inputs": {}}
    if term == "offer_rule":
        return {"text": "Domestic offers are tried before ending the turn or trading with the bank: the offer is ranked "
                        "0.001 above the best other action. This is a rule, not a valuation.", "refs": {}, "inputs": {}}
    return None


def resource_value_text(g, me, bi, part) -> str:
    r = RES.index(part["res"])
    mean = sum(bi.res_pips) / 5
    raw = (mean / max(1, bi.res_pips[r])) ** 0.5
    clip = " (clipped to 0.6–1.8)" if abs(raw - bi.scarcity[r]) > 1e-9 else ""
    vp = g.victory_points(me)
    late = min(1.0, max(0.0, (vp - 3) / 5))
    stage = "early game" if late == 0 else "late game" if late == 1 else f"{late:.0%} of the way to late game"
    return (f"{part['res']} {num(part['value'])} = board scarcity {num(part['scarcity'])} ({part['res']} has "
            f"{bi.res_pips[r]} pips on this board vs an average of {mean:.1f}: √({mean:.1f}/{bi.res_pips[r]}){clip}) "
            f"× game phase {num(part['phase'])} ({stage}, VP {vp}) × own production {num(part['own_factor'])} "
            f"(you produce {part['own_prod']:g} {part['res']} pips: 0.6 + 0.6/(1 + {part['own_prod']:g}/6)).")


def robber_text(g, me, bi, a, terms, detail) -> str:
    h = a.value
    if "own_hex" in terms:
        return f"Hex {h} touches your own building: −100, never chosen while another hex is legal."
    if "desert" in terms:
        return f"Hex {h} is the desert: −50."
    x = hex_info(g, bi, h)
    parts = [f"seat {v['seat']} {'city' if v['level'] == 2 else 'settlement'} ({x['pips']} pips × level {v['level']}"
             + (f" × scarcity {detail['scarcity']}" if detail.get("scarcity") is not None else "")
             + (f" × (1 + 0.3×{v['vp']} VP)" if detail.get("scarcity") is not None else
                f" × {v['leader_factor']:g} ({'the leader' if v['leader_factor'] == 10 else 'not the leader'})")
             + f" = {num(v['value'])})" for v in detail["victims"]]
    text = f"Robber on hex {h} ({hex_txt(x)}) blocks " + ("; ".join(parts) or "no one") + f" = {num(terms['blocked'])}."
    if terms.get("victims_bonus"):
        text += " +2 because a victim there has cards to steal."
    return text
