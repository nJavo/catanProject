"""Per-seat opponent features from public information only (D21).

``seat_features(layout, config, log)`` reads the board layout, the game config and the
public action log (every action is announced, catan.engine.infoset), nothing else, so it is
safe on an ``InfoSetGame``. Two kinds of evidence:

- **setup policy match**: setup has no hidden state, so the setup is replayed on a fresh
  ``Game`` with the same layout and, at each of the seat's 4 setup decisions, we check whether
  basic's and intermediate's (also trader's and search's) setup rules pick the action played;
- **behaviour rates** over the seat's own turns: domestic offers (count, bundle and overpay
  shape), accept / reject / counter responses, confirms, bank trades, builds, dev buys and
  plays, knight timing, robber targets (own hex, leader on hex, pips).
"""
from __future__ import annotations

import math

import numpy as np

from catan.bots import common as C
from catan.bots.basic import BasicHeuristicAgent
from catan.bots.intermediate import IntermediateAgent
from catan.engine.actions import A, Action
from catan.engine.board import TOPOLOGY as T
from catan.engine.game import Game

CLASSES = ("basic", "intermediate", "intermediate_trader", "search", "learned")

_SETUP_BOTS = (BasicHeuristicAgent(), IntermediateAgent())
_RESP = (A.ACCEPT_TRADE, A.REJECT_TRADE, A.COUNTER_OFFER)

FEATURE_NAMES = (
    [f"setup_{k}_{b}" for k in ("s1", "s2", "r1", "r2") for b in ("basic", "int")]
    + ["setup_seen"]
    + ["offers", "offer_bundle", "offer_overpay", "offer_any",
       "resp_n", "accept", "reject", "counter", "confirm", "cancel",
       "maritime", "road", "settlement", "city", "buy_dev",
       "knight", "knight_preroll", "knight_robbed", "monopoly", "yop", "road_building",
       "robber_n", "robber_own", "robber_leader", "robber_pips", "robber_desert", "steal_leader",
       "log_turns", "log_game_turn"])
NF = len(FEATURE_NAMES)


def _best(bot, g, me, legal, method):
    """The action the bot's setup rule plays (highest score, first on ties, as HeuristicBase)."""
    sc = getattr(bot, method)(g, me, legal)
    if not sc:
        return None
    return sc[max(range(len(sc)), key=lambda i: (sc[i][0], -i))][1]


def setup_matches(layout, config, log) -> np.ndarray:
    """[n, 9]: per setup step (s1, s2, r1, r2) x (basic, intermediate) match flags, then the
    fraction of the seat's 4 setup actions seen so far. Replays only the public setup prefix."""
    n = config.num_players
    out = np.zeros((n, 9), np.float32)
    g = Game(config=config, layout=layout, seed=0)  # setup uses no RNG; the seed is irrelevant
    k_set, k_road = [0] * n, [0] * n
    for a in log:
        if g.phase not in ("SETUP_SETTLEMENT", "SETUP_ROAD"):
            break
        p = g.actor
        legal = g.legal_actions()
        if a.type == A.BUILD_SETTLEMENT:
            col, k_set[p] = k_set[p], k_set[p] + 1
            method = "setup_settlement"
        else:
            col, k_road[p] = 2 + k_road[p], k_road[p] + 1
            method = "setup_road"
        if col < 4 and len(legal) > 1:
            for b, bot in enumerate(_SETUP_BOTS):
                out[p, 2 * col + b] = 1.0 if _best(bot, g, p, legal, method) == a else -1.0
        out[p, 8] += 0.25
        g.step(Action(p, a.type, a.value))
    return out


def seat_features(layout, config, log) -> np.ndarray:
    """[n, NF] feature matrix for every seat (row q describes seat q)."""
    n = config.num_players
    bi = C.board_info(layout)
    f = np.zeros((n, NF), np.float32)
    f[:, :9] = setup_matches(layout, config, log)
    turns = [0] * n
    owner = [-1] * T.num_vertices
    level = [0] * T.num_vertices
    bscore = [0] * n            # public building VP (settlements + 2 cities), longest road/army ignored
    robber = layout.desert_hex
    c = {k: np.zeros(n) for k in FEATURE_NAMES[9:]}
    rolled = [False] * n
    game_turn = 0
    setup_left = 4 * n
    for a in log:
        p, t, v = a
        if setup_left > 0:
            if t == A.BUILD_SETTLEMENT:
                owner[v], level[v] = p, 1
                bscore[p] += 1
            setup_left -= 1
            continue
        if t == A.ROLL:
            rolled[p] = True
        elif t == A.END_TURN:
            turns[p] += 1
            game_turn += 1
            rolled[p] = False
        elif t == A.OFFER_TRADE:
            give, get = v
            c["offers"][p] += 1
            c["offer_bundle"][p] += sum(give) > 1 or sum(get) > 1
            c["offer_overpay"][p] += sum(give) > sum(get)
        elif t in _RESP:
            c["resp_n"][p] += 1
            c[{A.ACCEPT_TRADE: "accept", A.REJECT_TRADE: "reject", A.COUNTER_OFFER: "counter"}[t]][p] += 1
        elif t == A.CONFIRM_TRADE:
            c["confirm"][p] += 1
        elif t == A.CANCEL_TRADE:
            c["cancel"][p] += 1
        elif t == A.MARITIME_TRADE:
            c["maritime"][p] += 1
        elif t == A.BUILD_ROAD:
            c["road"][p] += 1
        elif t == A.BUILD_SETTLEMENT:
            owner[v], level[v] = p, 1
            bscore[p] += 1
            c["settlement"][p] += 1
        elif t == A.BUILD_CITY:
            level[v] = 2
            bscore[p] += 1
            c["city"][p] += 1
        elif t == A.BUY_DEV:
            c["buy_dev"][p] += 1
        elif t == A.PLAY_KNIGHT:
            c["knight"][p] += 1
            c["knight_preroll"][p] += not rolled[p]
            c["knight_robbed"][p] += any(owner[x] == p for x in T.hex_vertices[robber])
        elif t == A.PLAY_MONOPOLY:
            c["monopoly"][p] += 1
        elif t == A.PLAY_YEAR_OF_PLENTY:
            c["yop"][p] += 1
        elif t == A.PLAY_ROAD_BUILDING:
            c["road_building"][p] += 1
        elif t == A.MOVE_ROBBER:
            robber = v
            others = [q for q in range(n) if q != p]
            lead = max(others, key=lambda q: (bscore[q], -q))
            on = [owner[x] for x in T.hex_vertices[v]]
            c["robber_n"][p] += 1
            c["robber_own"][p] += p in on
            c["robber_leader"][p] += lead in on
            c["robber_pips"][p] += bi.hex_pips[v] / 5.0
            c["robber_desert"][p] += bi.hex_res[v] < 0
        elif t == A.STEAL:
            others = [q for q in range(n) if q != p]
            c["steal_leader"][p] += v == max(others, key=lambda q: (bscore[q], -q))
    for q in range(n):
        tq = turns[q] + 1.0
        row = f[q]
        for j, k in enumerate(FEATURE_NAMES[9:], start=9):
            x = c[k][q]
            if k in ("offer_bundle", "offer_overpay"):
                x = x / max(1.0, c["offers"][q])
            elif k in ("accept", "reject", "counter"):
                x = x / max(1.0, c["resp_n"][q])
            elif k in ("knight_preroll", "knight_robbed"):
                x = x / max(1.0, c["knight"][q])
            elif k in ("robber_own", "robber_leader", "robber_pips", "robber_desert"):
                x = x / max(1.0, c["robber_n"][q])
            elif k == "offer_any":
                x = float(c["offers"][q] > 0)
            elif k == "log_turns":
                x = math.log1p(turns[q])
            elif k == "log_game_turn":
                x = math.log1p(game_turn)
            elif k == "resp_n":
                x = math.log1p(x)
            else:
                x = x / tq
            row[j] = x
    return f
