"""PositionSpec <-> engine Game: build, validate and determinize positions.

A PositionSpec holds what one seat (``viewer``) can see.
``determinize`` turns it into full engine states by sampling everything
hidden from the viewer. See ``ASSUMPTIONS`` for the sampling model.
"""
from __future__ import annotations

import copy
import random
from collections import Counter

from catan.engine import TOPOLOGY as T, RESOURCES, BoardLayout, Game, GameConfig, Phase
from catan.engine.board import DESERT, PORT_TYPES, TERRAIN_COUNTS, TOKEN_SEQUENCE
from catan.engine.game import BANK_START, DEV_DECK, SUPPLY, VICTORY_POINT, KNIGHT

NV, NE, NH = T.num_vertices, T.num_edges, T.num_hexes
TOTAL_RES = BANK_START * 5
TOTAL_DEV = sum(DEV_DECK)
SUPPORTED_PHASES = (Phase.PRE_ROLL, Phase.MAIN, Phase.MOVE_ROBBER, Phase.STEAL, Phase.DISCARD,
                    Phase.ROAD_BUILDING, Phase.SETUP_SETTLEMENT, Phase.SETUP_ROAD)
TRADE_PHASES = (Phase.TRADE_RESPONSE, Phase.TRADE_CONFIRM)
SETUP_PHASES = (Phase.SETUP_SETTLEMENT, Phase.SETUP_ROAD)
PROGRESS = (2, 3, 4)  # road_building, year_of_plenty, monopoly

ASSUMPTIONS = [
    "opponent hands: drawn uniformly (without replacement) from the unseen resource pool "
    "(19 per type minus the viewer's hand minus the bank when given) after first placing "
    "each opponent's known_hand_min cards; without a bank, the bank absorbs the rest",
    "opponent dev cards and deck order: drawn uniformly from the unseen dev pool "
    "(25-card composition minus the viewer's cards minus publicly played cards)",
    "when played_dev_cards is absent, non-knight played cards implied by the counts are "
    "removed uniformly at random from the unseen road_building/year_of_plenty/monopoly cards",
    "samples where the current (non-viewer) player would already hold >= vp_to_win are "
    "rejected (up to 50 tries per sample)",
    "dice, steals and dev draws use a fresh RNG per sample; no opponent modelling "
    "(hands are not conditioned on past behaviour)",
]


def _vec(x, n=5):
    return isinstance(x, list) and len(x) == n and all(type(v) is int and v >= 0 for v in x)


# ----------------------------------------------------------- from a game
def position_from_game(game: Game, viewer: int) -> dict:
    """PositionSpec of ``game`` as seen by ``viewer``. Reads only
    ``game.player_view(viewer)`` plus public quantities (played dev cards per
    type follow from 25 = deck + held + played; dev cards bought this turn are
    seen being bought)."""
    v = game.player_view(viewer)
    players = []
    for q, info in enumerate(v["players"]):
        if q == viewer:
            players.append({"hand": game.players[q].hand[:], "dev_cards": game.players[q].dev[:],
                            "dev_new": game.players[q].dev_new[:], "knights_played": info["knights_played"]})
        else:
            players.append({"resource_count": info["resource_count"], "dev_count": info["dev_count"],
                            "knights_played": info["knights_played"],
                            "dev_new_count": sum(game.players[q].dev_new)})
    # Pending trade: what offerers/acceptors/counterers committed to is public.
    tr = v["trade"]
    if tr is not None:
        def need(q, vec):
            if q != viewer:
                cur = players[q].setdefault("known_hand_min", [0] * 5)
                for i in range(5):
                    cur[i] = max(cur[i], vec[i])
        need(tr["proposer"], tr["give"])
        for k, r in tr["responses"].items():
            if r is not None:
                need(int(k), r["get"])
    held = [sum(pl.dev[t] for pl in game.players) for t in range(5)]
    played = [DEV_DECK[t] - held[t] - game.dev_deck.count(t) for t in range(5)]  # public by conservation
    spec = {
        "version": 1, "num_players": game.n, "viewer": viewer, "actor": v["actor"], "layout": v["layout"],
        "vertex_owner": v["vertex_owner"], "vertex_level": v["vertex_level"], "edge_owner": v["edge_owner"],
        "robber": v["robber"], "current": v["current"], "phase": v["phase"], "rolled": v["rolled"],
        "dev_played": v["dev_played"], "turn": v["turn"], "players": players,
        "longest_road_owner": v["longest_road_owner"], "largest_army_owner": v["largest_army_owner"],
        "bank": v["bank"], "dev_deck_count": v["dev_deck_count"], "played_dev_cards": played,
        "discard_remaining": v["discard_remaining"], "free_roads": v["free_roads"],
        "trades_this_turn": v["trades_this_turn"], "trade": tr,
        "last_roll": list(game.last_roll) if game.last_roll else None,  # public; policies observe it
        "config": {"vp_to_win": game.config.vp_to_win, "discard_limit": game.config.discard_limit,
                   "max_trade_offers_per_turn": game.config.max_trade_offers_per_turn},
    }
    return spec


# ----------------------------------------------------------- helpers
def _config(spec) -> GameConfig:
    c = spec.get("config") or {}
    return GameConfig(num_players=spec["num_players"], vp_to_win=c.get("vp_to_win", 10),
                      discard_limit=c.get("discard_limit", 7),
                      max_trade_offers_per_turn=c.get("max_trade_offers_per_turn", 3))


def _pieces(spec):
    """Per seat: (settlements, cities, roads) from the board arrays."""
    n = spec["num_players"]
    out = [([], [], []) for _ in range(n)]
    for vtx, (o, lvl) in enumerate(zip(spec["vertex_owner"], spec["vertex_level"])):
        if o != -1:
            out[o][0 if lvl == 1 else 1].append(vtx)
    for e, o in enumerate(spec["edge_owner"]):
        if o != -1:
            out[o][2].append(e)
    return out


def _setup_index(spec):
    k = sum(1 for o in spec["vertex_owner"] if o != -1)
    return k if spec["phase"] == Phase.SETUP_SETTLEMENT else k - 1


def _setup_order(n):
    return list(range(n)) + list(reversed(range(n)))


def _actor(spec):
    ph, n = spec["phase"], spec["num_players"]
    if ph in SETUP_PHASES:
        i = _setup_index(spec)
        return _setup_order(n)[i] if 0 <= i < 2 * n else None
    if ph == Phase.DISCARD:
        dr = spec.get("discard_remaining") or [0] * n
        for i in range(n):
            p = (spec["current"] + i) % n
            if dr[p] > 0:
                return p
        return None
    if ph == Phase.TRADE_RESPONSE:
        tr = spec.get("trade") or {}
        return (tr.get("pending") or [None])[0]
    return spec["current"]


def _road_lengths(spec, pieces):
    g = _skeleton(spec, pieces)
    return [g.longest_road(q) for q in range(spec["num_players"])]


def _skeleton(spec, pieces):
    """Board-only Game (no cards) used for derived quantities."""
    g = Game(config=_config(spec), seed=0, layout=BoardLayout.from_dict(spec["layout"]))
    g.vertex_owner, g.vertex_level, g.edge_owner = list(spec["vertex_owner"]), list(spec["vertex_level"]), list(spec["edge_owner"])
    for q, (s, c, r) in enumerate(pieces):
        pl = g.players[q]
        pl.settlements, pl.cities, pl.roads = s[:], c[:], r[:]
    return g


def _awards(spec, lengths):
    n = spec["num_players"]
    lr = spec.get("longest_road_owner")
    if lr is None:
        best = max(lengths)
        lr = lengths.index(best) if best >= 5 and lengths.count(best) == 1 else -1
    la = spec.get("largest_army_owner")
    if la is None:
        k = [spec["players"][q]["knights_played"] for q in range(n)]
        best = max(k)
        la = k.index(best) if best >= 3 and k.count(best) == 1 else -1
    return lr, la


def _played(spec):
    """(played per type or None, number of non-knight played cards)."""
    n = spec["num_players"]
    knights = sum(spec["players"][q]["knights_played"] for q in range(n))
    if spec.get("played_dev_cards") is not None:
        p = list(spec["played_dev_cards"])
        return p, 0
    held = _held_dev_total(spec)
    deck = spec.get("dev_deck_count")
    extra = 0 if deck is None else TOTAL_DEV - deck - held - knights
    return None, extra


def _held_dev_total(spec):
    v = spec["viewer"]
    return sum(sum(pl["dev_cards"]) if q == v else pl["dev_count"] for q, pl in enumerate(spec["players"]))


# ----------------------------------------------------------- validation
def _norm(spec):
    """Deep copy with defaults filled, so callers' dicts are never mutated."""
    spec = copy.deepcopy(spec)
    if isinstance(spec, dict) and isinstance(spec.get("players"), list):
        for pl in spec["players"]:
            if isinstance(pl, dict):
                pl.setdefault("knights_played", 0)
    return spec


def _err(message, vertices=(), edges=(), hexes=()):
    return {"message": message, "vertices": list(vertices), "edges": list(edges), "hexes": list(hexes)}


def validate_position_detailed(spec) -> list[dict]:
    """Problems as ``{message, vertices, edges, hexes}`` (ids of offending pieces)."""
    spec = _norm(spec)
    errs = _validate_shape(spec) or _validate_content(spec)
    return [x if isinstance(x, dict) else _err(x) for x in errs]


def validate_position(spec) -> list[str]:
    """Return a list of human-readable problems (empty == OK)."""
    return [x["message"] for x in validate_position_detailed(spec)]


def _validate_shape(spec):
    e = []
    if not isinstance(spec, dict):
        return ["position must be an object"]
    if spec.get("version", 1) != 1:
        e.append("version must be 1")
    n = spec.get("num_players")
    if type(n) is not int or not 2 <= n <= 4:
        return ["num_players must be 2..4"]
    for k in ("viewer", "current", "turn", "robber"):
        if type(spec.get(k)) is not int:
            e.append(f"{k} must be an integer")
    if e:
        return e
    if not 0 <= spec["viewer"] < n:
        e.append("viewer out of range")
    if not 0 <= spec["current"] < n:
        e.append("current out of range")
    if spec.get("actor") is not None and (type(spec["actor"]) is not int or not 0 <= spec["actor"] < n):
        e.append("actor must be a seat number or null")
    if not 0 <= spec["robber"] < NH:
        e.append(f"robber must be a hex id 0..{NH - 1}")
    if spec["turn"] < 0:
        e.append("turn must be >= 0")
    lay = spec.get("layout")
    if not isinstance(lay, dict):
        e.append("layout missing")
    else:
        e += _validate_layout(lay)
    for k, size, lo, hi in (("vertex_owner", NV, -1, n - 1), ("vertex_level", NV, 0, 2), ("edge_owner", NE, -1, n - 1)):
        a = spec.get(k)
        if not (isinstance(a, list) and len(a) == size and all(type(x) is int for x in a)):
            e.append(f"{k} must be a list of {size} integers")
        elif any(not lo <= x <= hi for x in a):
            e.append(f"{k} values must be in {lo}..{hi}")
    ph = spec.get("phase")
    if ph in TRADE_PHASES:
        if not isinstance(spec.get("trade"), dict):
            e.append(f"phase {ph} needs the trade state (only real game positions support trade phases)")
    elif spec.get("trade") is not None:
        e.append("trade state is only valid in TRADE_RESPONSE / TRADE_CONFIRM")
    if ph not in SUPPORTED_PHASES + TRADE_PHASES:
        e.append(f"unsupported phase {ph!r}; supported: {', '.join(SUPPORTED_PHASES)}")
    for k in ("rolled", "dev_played"):
        if type(spec.get(k, False)) is not bool:
            e.append(f"{k} must be a boolean")
    pls = spec.get("players")
    if not (isinstance(pls, list) and len(pls) == n and all(isinstance(p, dict) for p in pls)):
        return e + [f"players must be a list of {n} objects"]
    for q, pl in enumerate(pls):
        who = "viewer" if q == spec["viewer"] else f"player {q}"
        if type(pl.get("knights_played", 0)) is not int or pl.get("knights_played", 0) < 0:
            e.append(f"{who}: knights_played must be a non-negative integer")
        if q == spec["viewer"]:
            for k in ("hand", "dev_cards"):
                if not _vec(pl.get(k)):
                    e.append(f"viewer: {k} must be 5 non-negative integers")
            if "dev_new" in pl and not _vec(pl["dev_new"]):
                e.append("viewer: dev_new must be 5 non-negative integers")
        else:
            for k in ("resource_count", "dev_count"):
                if type(pl.get(k)) is not int or pl[k] < 0:
                    e.append(f"{who}: {k} must be a non-negative integer")
            if "known_hand_min" in pl and pl["known_hand_min"] is not None and not _vec(pl["known_hand_min"]):
                e.append(f"{who}: known_hand_min must be 5 non-negative integers")
            if type(pl.get("dev_new_count", 0)) is not int or pl.get("dev_new_count", 0) < 0:
                e.append(f"{who}: dev_new_count must be a non-negative integer")
    for k in ("bank", "played_dev_cards"):
        if spec.get(k) is not None and not _vec(spec[k]):
            e.append(f"{k} must be 5 non-negative integers")
    if spec.get("dev_deck_count") is not None and (type(spec["dev_deck_count"]) is not int or not 0 <= spec["dev_deck_count"] <= TOTAL_DEV):
        e.append("dev_deck_count must be 0..25")
    if spec.get("discard_remaining") is not None and not _vec(spec["discard_remaining"], n):
        e.append(f"discard_remaining must be {n} non-negative integers")
    for k in ("longest_road_owner", "largest_army_owner"):
        x = spec.get(k)
        if x is not None and (type(x) is not int or not -1 <= x < n):
            e.append(f"{k} must be -1..{n - 1}")
    c = spec.get("config")
    if c is not None:
        if not isinstance(c, dict):
            e.append("config must be an object")
        elif type(c.get("vp_to_win", 10)) is not int or not 3 <= c.get("vp_to_win", 10) <= 20:
            e.append("config.vp_to_win must be 3..20")
    return e


def _validate_layout(lay):
    e = []
    terr, nums = lay.get("hex_terrain"), lay.get("hex_number")
    if not (isinstance(terr, list) and len(terr) == NH):
        return [f"layout.hex_terrain must list {NH} terrains"]
    if Counter(terr) != Counter(TERRAIN_COUNTS):
        e.append("layout.hex_terrain must be the base set: 4 wood, 3 brick, 4 sheep, 4 wheat, 3 ore, 1 desert")
    if not (isinstance(nums, list) and len(nums) == NH):
        return e + [f"layout.hex_number must list {NH} entries"]
    for h, (t, x) in enumerate(zip(terr, nums)):
        if (t == DESERT) != (x is None):
            e.append(_err(f"hex {h}: the desert (and only the desert) has no number", hexes=[h]))
    if sorted(x for x in nums if x is not None) != sorted(TOKEN_SEQUENCE):
        e.append("layout.hex_number must use the 18 base number tokens (2,3,3,4,4,5,5,6,6,8,8,9,9,10,10,11,11,12)")
    pe, pt = lay.get("port_edges"), lay.get("port_types")
    if not (isinstance(pe, list) and isinstance(pt, list) and len(pe) == len(pt) == 9):
        return e + ["layout needs 9 port_edges and 9 port_types"]
    if any(x not in T.coast_edges for x in pe) or len(set(pe)) != 9:
        e.append("layout.port_edges must be 9 distinct coastal edge ids")
    if Counter(pt) != Counter(PORT_TYPES):
        e.append("layout.port_types must be 4 x '3:1' plus one 2:1 port per resource")
    return e


def _validate_content(spec):
    e = []
    n, viewer, ph = spec["num_players"], spec["viewer"], spec["phase"]
    vo, vl, eo = spec["vertex_owner"], spec["vertex_level"], spec["edge_owner"]
    pls = spec["players"]
    cfg = _config(spec)
    for vtx in range(NV):
        if (vo[vtx] == -1) != (vl[vtx] == 0):
            e.append(_err(f"vertex {vtx}: owner and level disagree (owned vertices need level 1 or 2)", vertices=[vtx]))
    for vtx in range(NV):
        bad = [u for u in T.vertex_neighbors[vtx] if u > vtx and vo[u] != -1]
        if vo[vtx] != -1 and bad:
            e.append(_err(f"vertex {vtx}: distance rule violated (adjacent building)", vertices=[vtx] + bad))
    pieces = _pieces(spec)
    setup = ph in SETUP_PHASES
    for q, (s, c, r) in enumerate(pieces):
        for name, lst in (("settlement", s), ("city", c), ("road", r)):
            if len(lst) > SUPPLY[name]:
                e.append(f"player {q}: {len(lst)} {name}s exceeds the supply of {SUPPLY[name]}")
        # each road component must touch one of the player's buildings
        own_v = set(s) | set(c)
        comp_seen = set()
        for e0 in r:
            if e0 in comp_seen:
                continue
            stack, comp = [e0], set()
            while stack:
                x = stack.pop()
                if x in comp:
                    continue
                comp.add(x)
                for vv in T.edge_vertices[x]:
                    stack.extend(y for y in T.vertex_edges[vv] if eo[y] == q and y not in comp)
            comp_seen |= comp
            if not any(vv in own_v for x in comp for vv in T.edge_vertices[x]):
                e.append(_err(f"player {q}: road(s) {sorted(comp)} not connected to any of their buildings", edges=sorted(comp)))
        if not setup:
            for vtx in s + c:
                if not any(eo[x] == q for x in T.vertex_edges[vtx]):
                    e.append(_err(f"player {q}: building at vertex {vtx} has no adjacent own road", vertices=[vtx]))
            if len(s) + len(c) < 2 or len(r) < 2:
                e.append(f"player {q}: after setup every player has at least 2 buildings and 2 roads")
    if setup:
        k = _setup_index(spec)
        order = _setup_order(n)
        if not 0 <= k < 2 * n:
            e.append(f"setup phase {ph} with {k + (ph == Phase.SETUP_ROAD)} settlements on the board is impossible")
        else:
            placed = Counter(order[:k + (ph == Phase.SETUP_ROAD)])
            roads_done = Counter(order[:k])
            for q, (s, c, r) in enumerate(pieces):
                if c:
                    e.append(f"player {q}: no cities during setup")
                if len(s) != placed[q] or len(r) != roads_done[q]:
                    e.append(f"player {q}: setup expects {placed[q]} settlement(s) and {roads_done[q]} road(s), "
                             f"found {len(s)} and {len(r)}")
            if ph == Phase.SETUP_ROAD and not _pending_setup_settlement(spec, pieces):
                e.append("SETUP_ROAD: the current player has no settlement waiting for its road")
    else:
        if ph == Phase.PRE_ROLL and spec.get("rolled"):
            e.append("PRE_ROLL requires rolled=false")
        if ph in (Phase.MAIN, Phase.DISCARD) and not spec.get("rolled"):
            e.append(f"{ph} requires rolled=true")
    # resources
    hand = pls[viewer]["hand"]
    bank = spec.get("bank")
    opp = [q for q in range(n) if q != viewer]
    opp_count = sum(pls[q]["resource_count"] for q in opp)
    mins = [sum((pls[q].get("known_hand_min") or [0] * 5)[r] for q in opp) for r in range(5)]
    for q in opp:
        km = pls[q].get("known_hand_min") or [0] * 5
        if sum(km) > pls[q]["resource_count"]:
            e.append(f"player {q}: known_hand_min ({sum(km)} cards) exceeds resource_count {pls[q]['resource_count']}")
    for r, name in enumerate(RESOURCES):
        used = hand[r] + (bank[r] if bank else 0) + mins[r]
        if used > BANK_START:
            e.append(f"{name}: viewer hand + {'bank + ' if bank else ''}known opponent cards = {used} > 19")
    if bank is not None:
        tot = sum(bank) + sum(hand) + opp_count
        if tot != TOTAL_RES:
            e.append(f"resource conservation: bank {sum(bank)} + viewer {sum(hand)} + opponents {opp_count} = {tot}, must be 95")
    elif sum(hand) + opp_count > TOTAL_RES:
        e.append("more resource cards in hands than exist (95)")
    # dev cards
    me = pls[viewer]
    dn = me.get("dev_new") or [0] * 5
    if any(dn[t] > me["dev_cards"][t] for t in range(5)):
        e.append("viewer: dev_new cannot exceed dev_cards")
    knights = sum(p["knights_played"] for p in pls)
    played, extra = _played(spec)
    held = _held_dev_total(spec)
    if played is not None:
        if played[KNIGHT] != knights:
            e.append(f"played_dev_cards knights ({played[KNIGHT]}) != sum of knights_played ({knights})")
        if played[VICTORY_POINT]:
            e.append("victory point cards are never played")
        base = played
    else:
        base = [knights, 0, 0, 0, 0]
    for t in range(5):
        if base[t] + me["dev_cards"][t] > DEV_DECK[t]:
            e.append(f"dev card type {t}: viewer holds + played exceeds the {DEV_DECK[t]} in the game")
    deck = spec.get("dev_deck_count")
    tot = held + sum(base) + extra + (deck if deck is not None else 0)
    if deck is not None and tot != TOTAL_DEV:
        e.append(f"dev conservation: deck {deck} + held {held} + played {sum(base)} = {held + sum(base) + deck}, must be 25")
    if deck is None and held + sum(base) > TOTAL_DEV:
        e.append("more dev cards held/played than exist (25)")
    if extra < 0:
        e.append("dev conservation: counts imply a negative number of played cards")
    unseen_prog = sum(DEV_DECK[t] - me["dev_cards"][t] for t in PROGRESS)
    if extra > unseen_prog:
        e.append(f"dev counts imply {extra} played progress cards but only {unseen_prog} are unaccounted for")
    # phase specifics
    dr = spec.get("discard_remaining") or [0] * n
    if ph == Phase.DISCARD:
        if not any(dr):
            e.append("DISCARD needs some discard_remaining > 0")
        for q in range(n):
            cnt = sum(hand) if q == viewer else pls[q]["resource_count"]
            if dr[q] > cnt:
                e.append(f"player {q}: must discard {dr[q]} but holds {cnt} cards")
    elif any(dr):
        e.append("discard_remaining must be all zero outside DISCARD")
    if ph == Phase.STEAL:
        victims = {vo[v] for v in T.hex_vertices[spec["robber"]] if vo[v] not in (-1, spec["current"])}
        victims = [q for q in victims if (sum(hand) if q == viewer else pls[q]["resource_count"]) > 0]
        if len(victims) < 2:
            e.append(_err("STEAL needs at least 2 victims with cards on the robber hex (otherwise the steal is automatic)", hexes=[spec["robber"]]))
    if ph == Phase.ROAD_BUILDING:
        fr = spec.get("free_roads", 0)
        if type(fr) is not int or not 1 <= fr <= 2:
            e.append("ROAD_BUILDING needs free_roads 1..2")
        if not spec.get("dev_played"):
            e.append("ROAD_BUILDING requires dev_played=true")
    if spec["robber"] != BoardLayout.from_dict(spec["layout"]).desert_hex and setup:
        e.append(_err("robber must be on the desert during setup", hexes=[spec["robber"]]))
    # awards and victory
    if not e:
        lengths = _road_lengths(spec, pieces)
        lr, la = spec.get("longest_road_owner"), spec.get("largest_army_owner")
        if lr is not None and lr != -1 and (lengths[lr] < 5 or lengths[lr] < max(lengths)):
            e.append(f"longest_road_owner {lr} has road length {lengths[lr]} (needs >= 5 and the longest)")
        k = [p["knights_played"] for p in pls]
        if la is not None and la != -1 and (k[la] < 3 or k[la] < max(k)):
            e.append(f"largest_army_owner {la} has {k[la]} knights (needs >= 3 and the most)")
        if not setup and not e:
            lr, la = _awards(spec, lengths)
            cur = spec["current"]
            vp = len(pieces[cur][0]) + 2 * len(pieces[cur][1]) + 2 * (lr == cur) + 2 * (la == cur)
            if cur == viewer:
                vp += me["dev_cards"][VICTORY_POINT]
            if vp >= cfg.vp_to_win:
                e.append(f"current player {cur} already has {vp} VP >= {cfg.vp_to_win}: the game is over")
    actor = _actor(spec) if not e else None
    if not e and actor is None:
        e.append("no player is due to act in this position")
    if not e and spec.get("actor") is not None and spec["actor"] != actor:
        e.append(f"actor {spec['actor']} disagrees with the phase rules, which give seat {actor}")
    return e


def _pending_setup_settlement(spec, pieces):
    cur = _setup_order(spec["num_players"])[_setup_index(spec)]  # setup: the actor, not `current`
    eo = spec["edge_owner"]
    for vtx in pieces[cur][0]:
        if not any(eo[x] == cur for x in T.vertex_edges[vtx]):
            return vtx
    return None


def derived(spec) -> dict:
    """Quantities derived from a valid spec (for the UI and for building games)."""
    spec = _norm(spec)
    pieces = _pieces(spec)
    lengths = _road_lengths(spec, pieces)
    lr, la = _awards(spec, lengths)
    viewer = spec["viewer"]
    pls = spec["players"]
    hand = pls[viewer]["hand"]
    bank = spec.get("bank")
    played, extra = _played(spec)
    knights = sum(p["knights_played"] for p in pls)
    held = _held_dev_total(spec)
    base = played or [knights, 0, 0, 0, 0]
    deck = spec.get("dev_deck_count")
    if deck is None:
        deck = TOTAL_DEV - held - sum(base)
    pub = [len(s) + 2 * len(c) + 2 * (lr == q) + 2 * (la == q) for q, (s, c, _) in enumerate(pieces)]
    out = {
        "actor": _actor(spec),
        "road_length": lengths, "longest_road_owner": lr, "largest_army_owner": la,
        "public_vp": pub, "viewer_vp": pub[viewer] + pls[viewer]["dev_cards"][VICTORY_POINT],
        "dev_deck_count": deck, "played_non_knight_unknown": extra,
        "unseen_resources": [BANK_START - hand[r] - (bank[r] if bank else 0) for r in range(5)],
        "unseen_dev": [DEV_DECK[t] - pls[viewer]["dev_cards"][t] - base[t] for t in range(5)],
        "bank": bank,
    }
    if spec["phase"] in SETUP_PHASES:
        out["setup_index"] = _setup_index(spec)
    return out


# ----------------------------------------------------------- determinization
BELIEF_ASSUMPTION = (
    "opponent hands: drawn jointly from the exact posterior given the viewer's public events "
    "(catan/bots/belief.py: every public card movement applied, unseen steals branched by the "
    "victim's cards, states contradicting a build, trade, offer or discard dropped)")


def assumptions(belief=None) -> list[str]:
    return [BELIEF_ASSUMPTION] + ASSUMPTIONS[1:] if belief is not None else ASSUMPTIONS


def determinize(spec: dict, n: int, seed: int | None = 0, max_turns: int | None = None,
                belief=None) -> list[Game]:
    """``n`` full engine states consistent with ``spec`` (validated first).
    ``max_turns`` (absolute turn number) caps automated continuations. ``belief``
    (a ``HandPosterior`` for the viewer at this position) replaces uniform hand sampling."""
    errs = validate_position(spec)
    if errs:
        raise ValueError("invalid position: " + "; ".join(errs))
    spec = _norm(spec)
    rng = random.Random(seed)
    d = derived(spec)
    return [_sample(spec, d, random.Random(rng.getrandbits(64)), max_turns, belief) for _ in range(n)]


def _belief_hands(spec, belief, rng):
    """Hands from the posterior, or None when it doesn't fit the spec (then sampling is uniform)."""
    pls = spec["players"]
    req = {q: pls[q]["known_hand_min"] for q in belief.opps if pls[q].get("known_hand_min")}
    hands = belief.sample(rng, req or None)
    if hands is None or any(sum(hands[q]) != pls[q]["resource_count"] for q in belief.opps):
        return None
    return hands


def _sample(spec, d, rng, max_turns, belief=None):
    n, viewer = spec["num_players"], spec["viewer"]
    pls = spec["players"]
    opp = [q for q in range(n) if q != viewer]
    me = pls[viewer]
    bank = spec.get("bank")
    played, extra = _played(spec)
    knights = sum(p["knights_played"] for p in pls)
    base = played or [knights, 0, 0, 0, 0]
    cfg = _config(spec)
    cfg.max_turns = max_turns
    pieces = _pieces(spec)
    lr, la = d["longest_road_owner"], d["largest_army_owner"]
    cur = spec["current"]
    for _ in range(50):
        # resources
        pool = Counter({r: BANK_START - me["hand"][r] - (bank[r] if bank else 0) for r in range(5)})
        hands = {}
        for q in opp:
            km = pls[q].get("known_hand_min") or [0] * 5
            hands[q] = list(km)
            pool.subtract({r: km[r] for r in range(5)})
        flat = [r for r in range(5) for _ in range(pool[r])]
        rng.shuffle(flat)
        i = 0
        for q in opp:
            k = pls[q]["resource_count"] - sum(hands[q])
            for r in flat[i:i + k]:
                hands[q][r] += 1
            i += k
        if belief is not None:
            hands = _belief_hands(spec, belief, rng) or hands
        # dev cards
        dpool = [t for t in range(5) for _ in range(DEV_DECK[t] - me["dev_cards"][t] - base[t])]
        if extra:
            prog = [j for j, t in enumerate(dpool) if t in PROGRESS]
            drop = set(rng.sample(prog, extra))
            dpool = [t for j, t in enumerate(dpool) if j not in drop]
        rng.shuffle(dpool)
        devs, i = {}, 0
        for q in opp:
            k = pls[q]["dev_count"]
            devs[q] = dpool[i:i + k]
            i += k
        deck = dpool[i:]
        if cur == viewer or spec["phase"] in SETUP_PHASES:
            break
        vp_cur = d["public_vp"][cur] + devs[cur].count(VICTORY_POINT)
        if vp_cur < cfg.vp_to_win:
            break
    players = []
    vports = BoardLayout.from_dict(spec["layout"]).vertex_ports()
    for q in range(n):
        s, c, r = pieces[q]
        ports = sorted({vports[v] for v in s + c if v in vports})
        if q == viewer:
            hand, dev = list(me["hand"]), list(me["dev_cards"])
            dev_new = list(me.get("dev_new") or [0] * 5)
        else:
            hand = hands[q]
            dev = [devs[q].count(t) for t in range(5)]
            dev_new = [0] * 5
            for t in devs[q][:pls[q].get("dev_new_count", 0)]:
                dev_new[t] += 1
        players.append({"hand": hand, "dev": dev, "dev_new": dev_new, "knights": pls[q]["knights_played"],
                        "settlements": s, "cities": c, "roads": r, "ports": ports})
    if bank is None:
        bank_v = [BANK_START - sum(p["hand"][r] for p in players) for r in range(5)]
    else:
        bank_v = list(bank)
    gseed = rng.getrandbits(63)
    ph = spec["phase"]
    tr = spec.get("trade")
    rolled = bool(spec.get("rolled", False))
    g_dict = {
        "version": 1, "seed": gseed, "config": cfg.to_dict(), "layout": spec["layout"],
        "rng_state": _rng_state(gseed), "players": players, "bank": bank_v, "dev_deck": deck,
        "vertex_owner": list(spec["vertex_owner"]), "vertex_level": list(spec["vertex_level"]),
        "edge_owner": list(spec["edge_owner"]), "robber": spec["robber"], "phase": ph,
        "setup_index": _setup_index(spec) if ph in SETUP_PHASES else 2 * n,
        "current": 0 if ph in SETUP_PHASES else cur, "turn": spec["turn"], "rolled": rolled, "dev_played": bool(spec.get("dev_played", False)),
        "last_roll": _last_roll(spec), "discard_remaining": list(spec.get("discard_remaining") or [0] * n),
        "free_roads": spec.get("free_roads", 0) if ph == Phase.ROAD_BUILDING else 0,
        "last_settlement": (_pending_setup_settlement(spec, pieces) if ph == Phase.SETUP_ROAD else -1),
        "resume_phase": Phase.MAIN if rolled else Phase.PRE_ROLL,
        "trade": None if tr is None else {
            "proposer": tr["proposer"], "give": tr["give"], "get": tr["get"], "pending": tr["pending"],
            "responses": {k: (None if r is None else [r["give"], r["get"]]) for k, r in tr["responses"].items()}},
        "trades_this_turn": spec.get("trades_this_turn", 1 if tr else 0),
        "longest_road_owner": lr, "largest_army_owner": la, "road_length": d["road_length"],
        "winner": -1, "truncated": False, "action_log": [],
    }
    return Game.from_dict(g_dict)


def _last_roll(spec):
    r = spec.get("last_roll")
    ok = isinstance(r, list) and len(r) == 2 and all(type(x) is int and 1 <= x <= 6 for x in r)
    return list(r) if ok else None


def _rng_state(seed):
    st = random.Random(seed).getstate()
    return [st[0], list(st[1]), st[2]]

