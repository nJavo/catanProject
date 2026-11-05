"""Per-game, per-seat behavioral metrics from a replay (seed, config, actions).

The game is re-simulated with ``Game`` (deterministic), so every metric can
read the exact state before and after each action. Metrics use the true state
(including hidden hands): this is offline analysis, not an agent.

``analyze_game(seed, config, actions)`` returns one flat-ish dict per seat.
Counts are per game; ratios are left as numerator/denominator pairs
(``*_num`` / ``*_den``) so aggregation can pool them correctly.
"""
from __future__ import annotations

from collections import Counter, defaultdict

from catan.bots.common import board_info
from catan.engine import A, Action, Game, GameConfig, Phase
from catan.engine.board import TOPOLOGY as T, RESOURCES
from catan.engine.game import COST, DEV_TYPES, SUPPLY

VP_MARKS = (5, 8, 10)


def _vertex_pips(bi, v):
    return sum(bi.vertex_pips[v])


def _seat_pips(g, bi, q):
    """Expected pips (cities double) for seat q, ignoring the robber."""
    tot = 0
    for v in g.players[q].settlements + g.players[q].cities:
        tot += _vertex_pips(bi, v) * g.vertex_level[v]
    return tot


def _hex_block(g, bi, h, q):
    """Pips seat q loses if the robber sits on hex h."""
    pips = bi.hex_pips[h]
    return sum(pips * g.vertex_level[v] for v in T.hex_vertices[h] if g.vertex_owner[v] == q)


def dangling_roads(g, q) -> int:
    """Roads left on dead-end branches that end at no own building.

    Repeatedly prunes road edges with a free end (degree 1 in q's road graph)
    that holds no building of q. What survives connects buildings or forms
    loops. shortcut: ignores that a dangling branch may still be q's longest
    road; the report shows the Longest Road holder separately."""
    roads = set(g.players[q].roads)
    own = set(g.players[q].settlements) | set(g.players[q].cities)
    changed = True
    while changed:
        changed = False
        deg = Counter(v for e in roads for v in T.edge_vertices[e])
        for e in list(roads):
            if any(deg[v] == 1 and v not in own for v in T.edge_vertices[e]):
                roads.discard(e)
                changed = True
                break
    return len(g.players[q].roads) - len(roads)


def _new_seat():
    return {
        "actions": Counter(),          # "PHASE:TYPE" -> count
        "builds": Counter(),           # settlement/city/road/dev (paid, not setup)
        "first_build_round": {},
        "vp_round": {},
        "dev_bought_round": [],
        "dev_played": Counter(),
        "knight_preroll": 0, "knight_postroll": 0,
        "dev_hold_rounds": [],
        "robber": Counter(),
        "robber_eff": [],
        "maritime_by_rate": Counter(),
        "maritime_cards_given": 0, "maritime_cards_got": 0,
        "offers": 0, "offers_executed": 0,
        "asked": 0, "accepted": 0, "countered": 0, "accepted_executed": 0,
        "domestic_cards_given": 0, "domestic_cards_got": 0,
        "discarded": 0, "sevens_discarding": 0,
        "hand_at_7": [], "hand_at_end_turn": [],
        "production_cards": 0, "stolen_got": 0, "stolen_lost": 0, "monopoly_got": 0,
        "end_turns": 0, "missed": Counter(),
    }


def analyze_game(seed: int, config, actions, layout=None) -> list[dict]:
    """Behavioral metrics per seat. ``config`` is a GameConfig or its dict;
    ``actions`` are Action objects or their JSON form."""
    cfg = config if isinstance(config, GameConfig) else GameConfig(**config)
    g = Game(config=cfg, seed=seed, layout=layout)
    n = g.n
    bi = board_info(g.layout)
    S = [_new_seat() for _ in range(n)]
    opening = [[] for _ in range(n)]
    dev_queue = [defaultdict(list) for _ in range(n)]  # card -> [buy round] FIFO
    vp_now = [0] * n

    def rnd():
        return g.turn // n

    for raw in actions:
        a = raw if isinstance(raw, Action) else Action.from_json(raw)
        p, t, v = a
        ph = g.phase
        s = S[p]
        s["actions"][f"{ph}:{t}"] += 1

        # ---- pre-action state captures
        if ph == Phase.SETUP_SETTLEMENT:
            free = [u for u in range(T.num_vertices) if g._settlement_spot_free(u)]
            best = max(_vertex_pips(bi, u) for u in free)
            better = sum(1 for u in free if _vertex_pips(bi, u) > _vertex_pips(bi, v))
            opening[p].append({"vertex": v, "pips": _vertex_pips(bi, v), "best_free": best,
                               "rank_frac": better / len(free),
                               "res": [r for r in range(5) if bi.vertex_pips[v][r] > 0],
                               "port": bi.vertex_port.get(v)})
        elif ph == Phase.MOVE_ROBBER:
            opp = [q for q in range(n) if q != p]
            pub = {q: g.public_victory_points(q) for q in opp}
            top = max(pub.values())
            leaders = {q for q in opp if pub[q] == top}
            on_hex = {g.vertex_owner[u] for u in T.hex_vertices[v]} - {-1}
            blocked = sum(_hex_block(g, bi, v, q) for q in opp)
            best_block = max(sum(_hex_block(g, bi, h, q) for q in opp)
                             for h in range(T.num_hexes) if h != g.robber
                             and _hex_block(g, bi, h, p) == 0)
            r = s["robber"]
            r["moves"] += 1
            r["knight"] += g.action_log[-1].type == A.PLAY_KNIGHT
            r["hits_leader"] += bool(on_hex & leaders)
            r["self_block"] += p in on_hex
            r["no_opponent"] += not (on_hex - {p})
            r["desert"] += g.layout.hex_terrain[v] == "desert"
            s["robber_eff"].append(blocked / best_block if best_block else 1.0)
        elif ph == Phase.MAIN and t == A.END_TURN:
            s["end_turns"] += 1
            pl = g.players[p]
            m = s["missed"]
            spot = bool(g.legal_settlement_vertices(p))
            m["has_spot"] += spot
            m["afford_settlement"] += g.can_afford(p, COST["settlement"])
            m["afford_city"] += g.can_afford(p, COST["city"])
            miss_s = g.can_afford(p, COST["settlement"]) and spot
            miss_c = (g.can_afford(p, COST["city"]) and pl.settlements
                      and len(pl.cities) < SUPPLY["city"])
            miss_d = g.can_afford(p, COST["dev"]) and bool(g.dev_deck)
            m["settlement"] += bool(miss_s)
            m["city"] += bool(miss_c)
            m["dev"] += bool(miss_d)
            m["any_vp_build"] += bool(miss_s or miss_c)
            m["any"] += bool(miss_s or miss_c or miss_d)
            s["hand_at_end_turn"].append(sum(pl.hand))
        elif t == A.PLAY_KNIGHT:
            s["knight_preroll" if ph == Phase.PRE_ROLL else "knight_postroll"] += 1
        elif t == A.ACCEPT_TRADE or t == A.REJECT_TRADE or t == A.COUNTER_OFFER:
            s["asked"] += 1
            s["accepted"] += t == A.ACCEPT_TRADE
            s["countered"] += t == A.COUNTER_OFFER

        events = g.step(a)

        # ---- events
        vp_dirty = False
        for ev in events:
            et = ev["type"]
            if et == "dice_rolled" and ev["total"] == 7:
                for q in range(n):
                    S[q]["hand_at_7"].append(sum(g.players[q].hand))
                    S[q]["sevens_discarding"] += g.discard_remaining[q] > 0
            elif et == "discarded":
                S[ev["player"]]["discarded"] += 1
            elif et == "resources_gained" and ev["reason"] == "production":
                S[ev["player"]]["production_cards"] += sum(ev["resources"])
            elif et == "stolen":
                S[ev["thief"]]["stolen_got"] += 1
                S[ev["victim"]]["stolen_lost"] += 1
            elif et == "monopoly":
                S[ev["player"]]["monopoly_got"] += ev["amount"]
            elif et in ("settlement_built", "city_built", "road_built"):
                vp_dirty = True
                if not ev.get("free"):
                    kind = et.split("_")[0]
                    S[ev["player"]]["builds"][kind] += 1
                    S[ev["player"]]["first_build_round"].setdefault(kind, rnd())
                elif et == "road_built" and ph == Phase.ROAD_BUILDING:
                    S[ev["player"]]["builds"]["free_road"] += 1
            elif et == "dev_bought_detail":
                q = ev["player"]
                S[q]["builds"]["dev"] += 1
                S[q]["dev_bought_round"].append(rnd())
                dev_queue[q][ev["card"]].append(rnd())
                vp_dirty = True
            elif et == "dev_played":
                q, c = ev["player"], ev["card"]
                S[q]["dev_played"][c] += 1
                if dev_queue[q][c]:
                    S[q]["dev_hold_rounds"].append(rnd() - dev_queue[q][c].pop(0))
            elif et == "award_changed":
                vp_dirty = True
            elif et == "maritime_trade":
                q = ev["player"]
                S[q]["maritime_by_rate"][str(ev["amount"])] += 1
                S[q]["maritime_cards_given"] += ev["amount"]
                S[q]["maritime_cards_got"] += 1
            elif et == "trade_offered":
                S[ev["player"]]["offers"] += 1
            elif et == "trade_executed":
                q, w = ev["player"], ev["partner"]
                gv, gt = sum(ev["give"]), sum(ev["get"])
                S[q]["offers_executed"] += 1
                S[q]["domestic_cards_given"] += gv
                S[q]["domestic_cards_got"] += gt
                S[w]["accepted_executed"] += 1
                S[w]["domestic_cards_given"] += gt
                S[w]["domestic_cards_got"] += gv
        if vp_dirty:
            for q in range(n):
                vp_now[q] = g.victory_points(q)
                for mark in VP_MARKS:
                    if vp_now[q] >= mark:
                        S[q]["vp_round"].setdefault(str(mark), rnd())

    # ---- end of game
    out = []
    for q in range(n):
        s, pl = S[q], g.players[q]
        op = opening[q]
        res_cov = set(r for o in op for r in o["res"])
        rv = s["robber"]
        row = {
            "seat": q,
            "won": int(g.winner == q),
            "truncated": int(g.truncated),
            "rounds": rnd(),
            "vp": g.victory_points(q),
            # opening
            "open_pips_1": op[0]["pips"] if op else 0,
            "open_pips_2": op[1]["pips"] if len(op) > 1 else 0,
            "open_pips_total": sum(o["pips"] for o in op),
            "open_pick_vs_best": (sum(o["pips"] for o in op) / max(1, sum(o["best_free"] for o in op))),
            "open_rank_frac": sum(o["rank_frac"] for o in op) / max(1, len(op)),
            "open_resource_count": len(res_cov),
            "open_missing": [RESOURCES[r] for r in range(5) if r not in res_cov],
            "open_port": int(any(o["port"] for o in op)),
            # expansion
            # paid builds only: the 2 free setup settlements are excluded (final_settlements includes them)
            "settlements_built": s["builds"]["settlement"],
            "cities_built": s["builds"]["city"],
            "roads_built": s["builds"]["road"],
            "free_roads": s["builds"]["free_road"],
            "roads_per_settlement": (s["builds"]["road"] + s["builds"]["free_road"])
            / max(1, s["builds"]["settlement"]),
            "dangling_roads": dangling_roads(g, q),
            "first_settlement_round": s["first_build_round"].get("settlement"),
            "first_city_round": s["first_build_round"].get("city"),
            "final_settlements": len(pl.settlements),
            "final_cities": len(pl.cities),
            "final_roads": len(pl.roads),
            "final_pips": _seat_pips(g, bi, q),
            # VP sources
            "vp_settlements": len(pl.settlements),
            "vp_cities": 2 * len(pl.cities),
            "vp_longest_road": 2 * (g.longest_road_owner == q),
            "vp_largest_army": 2 * (g.largest_army_owner == q),
            "vp_cards": pl.dev[1],
            "round_vp5": s["vp_round"].get("5"),
            "round_vp8": s["vp_round"].get("8"),
            "round_vp10": s["vp_round"].get("10"),
            # dev cards
            "dev_bought": s["builds"]["dev"],
            "dev_first_buy_round": s["dev_bought_round"][0] if s["dev_bought_round"] else None,
            "dev_played": {c: s["dev_played"][c] for c in DEV_TYPES if c != "victory_point"},
            "knight_preroll": s["knight_preroll"],
            "knight_postroll": s["knight_postroll"],
            "dev_hold_rounds_mean": (sum(s["dev_hold_rounds"]) / len(s["dev_hold_rounds"])
                                     if s["dev_hold_rounds"] else None),
            "dev_unplayed_end": sum(pl.dev) - pl.dev[1],
            "knights_unplayed_end": pl.dev[0],
            # robber
            "robber_moves": rv["moves"],
            "robber_by_knight": rv["knight"],
            "robber_hits_leader": rv["hits_leader"],
            "robber_self_block": rv["self_block"],
            "robber_no_opponent": rv["no_opponent"],
            "robber_desert": rv["desert"],
            "robber_block_eff_mean": (sum(s["robber_eff"]) / len(s["robber_eff"])
                                      if s["robber_eff"] else None),
            # trading
            "maritime": sum(s["maritime_by_rate"].values()),
            "maritime_4to1": s["maritime_by_rate"]["4"],
            "maritime_3to1": s["maritime_by_rate"]["3"],
            "maritime_2to1": s["maritime_by_rate"]["2"],
            "ports_owned": len(pl.ports),
            "offers": s["offers"],
            "offers_executed": s["offers_executed"],
            "trade_asked": s["asked"],
            "trade_accepted": s["accepted"],
            "trade_countered": s["countered"],
            "trade_accepted_executed": s["accepted_executed"],
            "domestic_net_cards": s["domestic_cards_got"] - s["domestic_cards_given"],
            "maritime_net_cards": s["maritime_cards_got"] - s["maritime_cards_given"],
            # hoarding / income
            "production_cards": s["production_cards"],
            "stolen_got": s["stolen_got"],
            "stolen_lost": s["stolen_lost"],
            "monopoly_got": s["monopoly_got"],
            "discarded_cards": s["discarded"],
            "sevens_discarding": s["sevens_discarding"],
            "sevens_rolled": len(s["hand_at_7"]),
            "hand_at_7_mean": sum(s["hand_at_7"]) / len(s["hand_at_7"]) if s["hand_at_7"] else None,
            "hand_at_end_turn_mean": (sum(s["hand_at_end_turn"]) / len(s["hand_at_end_turn"])
                                      if s["hand_at_end_turn"] else None),
            "hand_final": sum(pl.hand),
            # missed builds
            "end_turns": s["end_turns"],
            "missed_settlement": s["missed"]["settlement"],
            "missed_city": s["missed"]["city"],
            "missed_dev": s["missed"]["dev"],
            "missed_vp_build": s["missed"]["any_vp_build"],
            "missed_any": s["missed"]["any"],
            "end_turn_has_spot": s["missed"]["has_spot"],
            "end_turn_afford_settlement": s["missed"]["afford_settlement"],
            "end_turn_afford_city": s["missed"]["afford_city"],
            "actions": dict(s["actions"]),
        }
        out.append(row)
    return out
