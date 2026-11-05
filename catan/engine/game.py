"""Authoritative Catan base-game rules engine (5th edition rules).

Single source of truth for UI, bots, RL and evaluation. All randomness flows
through one seeded ``random.Random`` so (seed, config, action list) fully
determines a game.

Usage:
    g = Game(num_players=4, seed=7)
    while not g.is_over:
        acts = g.legal_actions()          # only for g.actor
        events = g.step(acts[0])
"""
from __future__ import annotations

import random
from functools import lru_cache
from dataclasses import dataclass, asdict

from .actions import A, Action, Phase, ALL_ACTION_TYPES
from .board import (TOPOLOGY, RESOURCES, DESERT, GENERIC_PORT, BoardLayout,
                    generate_layout)

KNIGHT, VICTORY_POINT, ROAD_BUILDING, YEAR_OF_PLENTY, MONOPOLY = range(5)
DEV_TYPES = ("knight", "victory_point", "road_building", "year_of_plenty", "monopoly")
DEV_DECK = (14, 5, 2, 2, 2)
BANK_START = 19

COST = {
    "road": (1, 1, 0, 0, 0),
    "settlement": (1, 1, 1, 1, 0),
    "city": (0, 0, 0, 2, 3),
    "dev": (0, 0, 1, 1, 1),
}
SUPPLY = {"road": 15, "settlement": 5, "city": 4}

T = TOPOLOGY
NV, NE, NH = T.num_vertices, T.num_edges, T.num_hexes


class IllegalAction(Exception):
    pass


@dataclass
class GameConfig:
    num_players: int = 4
    vp_to_win: int = 10
    discard_limit: int = 7          # more than this many cards -> discard half
    # Platform protocol limit (not an official rule): bounds negotiation so
    # turns always terminate. Set None for unlimited (human tables).
    max_trade_offers_per_turn: int | None = 3
    shuffle_ports: bool = True
    # Hard cap on turns for automated play; game ends with no winner (truncated).
    max_turns: int | None = None

    def to_dict(self):
        return asdict(self)


def trade_templates():
    """Bounded menu of domestic offers enumerated by legal_actions:
    1:1, 2:1 and 1:2 swaps between distinct resources (60 total)."""
    out = []
    for g in range(5):
        for r in range(5):
            if g == r:
                continue
            for gn, rn in ((1, 1), (2, 1), (1, 2)):
                give = [0] * 5
                get = [0] * 5
                give[g] = gn
                get[r] = rn
                out.append((tuple(give), tuple(get)))
    return tuple(out)


TRADE_TEMPLATES = trade_templates()
# (give_res, give_n, get_res, get_n, value) per template, same order: avoids
# 5-wide vector scans in the hot legal_actions loop.
_TPL = tuple((next(i for i in range(5) if g[i]), max(g), next(i for i in range(5) if r[i]), max(r), tt)
             for tt in TRADE_TEMPLATES for g, r in (tt,))
_TPL_INDEX = {t[4]: t for t in _TPL}
# Prebuilt actions (and catan.rl.spaces) reuse the TRADE_TEMPLATES objects as
# values, so is_legal can recognize them by identity: no _well_formed walk and
# no nested-tuple hash. Ids are unique while TRADE_TEMPLATES keeps them alive.
_TPL_BY_ID = {id(t[4]): t for t in _TPL}
VE, EV, VN = T.vertex_edges, T.edge_vertices, T.vertex_neighbors
# Prebuilt (immutable) template actions per seat (0..3), grouped so whole
# groups are skipped with one check. Groups keep TRADE_TEMPLATES order.
# _OFFER_ACTS[q][give_res] = (actions giving 1 card, all 12 actions)
_OFFER_ACTS = tuple(
    tuple((tuple(Action(q, A.OFFER_TRADE, t[4]) for t in _TPL if t[0] == gi and t[1] == 1),
           tuple(Action(q, A.OFFER_TRADE, t[4]) for t in _TPL if t[0] == gi)) for gi in range(5))
    for q in range(4))
# _COUNTER_ACTS[q] = ((give_res, get_res, ((give_n, get_n, value, action), ...)), ...)
_COUNTER_ACTS = tuple(
    tuple((gi, ri, tuple((t[1], t[3], t[4], Action(q, A.COUNTER_OFFER, t[4]))
                         for t in _TPL if t[0] == gi and t[2] == ri))
          for gi in range(5) for ri in range(5) if gi != ri)
    for q in range(4))
# _COUNTERS[q][hand pattern] = (counter actions in _COUNTER_ACTS order, {value: index}).
# Counter legality only depends on min(hand[r], 2) per resource (3**5 patterns).
_CLAMP2 = (0, 1) + (2,) * 62  # min(count, 2); counts are <= 19 by conservation, slack for edited states
_PATTERN_W = (1, 3, 9, 27, 81)


def _counter_table(q):
    out = []
    for idx in range(3 ** 5):
        h = [idx // w % 3 for w in _PATTERN_W]
        acts = tuple(act for gi, ri, group in _COUNTER_ACTS[q] if h[ri]
                     for gn, rn, val, act in group if h[ri] >= rn)
        out.append((acts, {a.value: i for i, a in enumerate(acts)}))
    return tuple(out)


_COUNTERS = tuple(_counter_table(q) for q in range(4))


# Prebuilt (immutable) actions per seat: building them in legal_actions was a
# large share of the per-step cost.
def _per_seat(t, values):
    return tuple(tuple(Action(q, t, v) for v in values) for q in range(4))


_ROAD_ACTS = _per_seat(A.BUILD_ROAD, range(NE))
_SETTLE_ACTS = _per_seat(A.BUILD_SETTLEMENT, range(NV))
_CITY_ACTS = _per_seat(A.BUILD_CITY, range(NV))
_ROBBER_ACTS = _per_seat(A.MOVE_ROBBER, range(NH))
_DISCARD_ACTS = _per_seat(A.DISCARD, range(5))
_STEAL_ACTS = _per_seat(A.STEAL, range(4))
_CONFIRM_ACTS = _per_seat(A.CONFIRM_TRADE, range(4))
_MONO_ACTS = _per_seat(A.PLAY_MONOPOLY, range(5))
_YOP_PAIRS = tuple((a, b) for a in range(5) for b in range(a, 5))
_YOP_ACTS = tuple(tuple((a, b, Action(q, A.PLAY_YEAR_OF_PLENTY, (a, b))) for a, b in _YOP_PAIRS)
                  for q in range(4))
# _MARITIME_ACTS[q][g] = ((r, action), ...) for r != g
_MARITIME_ACTS = tuple(tuple(tuple((r, Action(q, A.MARITIME_TRADE, (g, r))) for r in range(5) if r != g)
                             for g in range(5)) for q in range(4))
_SIMPLE_ACTS = tuple({t: Action(q, t) for t in ALL_ACTION_TYPES} for q in range(4))
# Every Action object legal_actions() can return (it never builds new ones).
# They live for the whole process, so id(action) is a stable cache key for
# callers such as catan.rl.spaces.action_mask.
PREBUILT_ACTIONS = tuple(
    [a for tbl in (_ROAD_ACTS, _SETTLE_ACTS, _CITY_ACTS, _ROBBER_ACTS, _DISCARD_ACTS, _STEAL_ACTS,
                   _CONFIRM_ACTS, _MONO_ACTS) for seat in tbl for a in seat]
    + [a for seat in _YOP_ACTS for _, _, a in seat]
    + [a for seat in _MARITIME_ACTS for row in seat for _, a in row]
    + [a for seat in _SIMPLE_ACTS for a in seat.values()]
    + [a for seat in _OFFER_ACTS for ones, every in seat for a in ones + every]
    + [act for seat in _COUNTER_ACTS for _, _, group in seat for _, _, _, act in group])


def longest_trail(edges, vertex_owner, p: int) -> int:
    """Public: longest road ``p`` would have with road set ``edges`` (any
    iterable of edge ids) on a board with ``vertex_owner``; opponents'
    buildings block continuation, as in Game.longest_road. Memoized."""
    edges = frozenset(edges)
    if not edges:
        return 0
    blocked = frozenset(v for e in edges for v in EV[e] if vertex_owner[v] != -1 and vertex_owner[v] != p)
    return _longest_trail(edges, blocked)


@lru_cache(maxsize=1 << 14)
def _longest_trail(edges: frozenset, blocked: frozenset) -> int:
    """Longest trail over ``edges`` that does not pass through ``blocked``
    vertices. A pure function of the two sets, so it is memoized: bots and
    search re-evaluate the same networks (candidate roads, rollouts) a lot."""
    adj = {}  # vertex -> [(own edge, other end)]
    for e in edges:
        a, b = EV[e]
        adj.setdefault(a, []).append((e, b))
        adj.setdefault(b, []).append((e, a))
    best = 0

    def dfs(v, used, length):
        nonlocal best
        if length > best:
            best = length
        if used and v in blocked:
            return  # opponent building blocks continuation
        for e, w in adj[v]:
            if e not in used:
                used.add(e)
                dfs(w, used, length + 1)
                used.discard(e)

    # A maximum trail can always start at a dead end, branch point or
    # blocked vertex; only a pure cycle component has none of those.
    starts = [v for v, nb in adj.items() if len(nb) != 2 or v in blocked]
    covered = set()
    stack = starts[:]
    while stack:
        v = stack.pop()
        if v not in covered:
            covered.add(v)
            stack.extend(w for _, w in adj[v])
    for v in adj:
        if v not in covered:  # first vertex of an uncovered cycle component
            starts.append(v)
            stack = [v]
            while stack:
                u = stack.pop()
                if u not in covered:
                    covered.add(u)
                    stack.extend(w for _, w in adj[u])
    n_edges = len(edges)
    for v in starts:
        dfs(v, set(), 0)
        if best == n_edges:
            break
    return best


def _affords(h, c) -> bool:
    return h[0] >= c[0] and h[1] >= c[1] and h[2] >= c[2] and h[3] >= c[3] and h[4] >= c[4]


class Player:
    __slots__ = ("hand", "dev", "dev_new", "knights", "settlements", "cities",
                 "roads", "ports")

    def __init__(self):
        self.hand = [0] * 5
        self.dev = [0] * 5        # held (unplayed) dev cards by type
        self.dev_new = [0] * 5    # of which bought this turn
        self.knights = 0          # knights played
        self.settlements: list[int] = []
        self.cities: list[int] = []
        self.roads: list[int] = []
        self.ports: set = set()   # port types owned ("3:1" or resource name)

    def copy(self):
        p = Player.__new__(Player)
        p.hand = self.hand[:]
        p.dev = self.dev[:]
        p.dev_new = self.dev_new[:]
        p.knights = self.knights
        p.settlements = self.settlements[:]
        p.cities = self.cities[:]
        p.roads = self.roads[:]
        p.ports = set(self.ports)
        return p

    def to_dict(self):
        return {"hand": self.hand[:], "dev": self.dev[:], "dev_new": self.dev_new[:],
                "knights": self.knights, "settlements": self.settlements[:],
                "cities": self.cities[:], "roads": self.roads[:],
                "ports": sorted(self.ports)}

    @staticmethod
    def from_dict(d):
        p = Player()
        p.hand, p.dev, p.dev_new = list(d["hand"]), list(d["dev"]), list(d["dev_new"])
        p.knights = d["knights"]
        p.settlements, p.cities, p.roads = list(d["settlements"]), list(d["cities"]), list(d["roads"])
        p.ports = set(d["ports"])
        return p


def _valid_vec(v):
    return (isinstance(v, tuple) and len(v) == 5
            and all(isinstance(x, int) and not isinstance(x, bool) and 0 <= x <= 19 for x in v))


def _well_formed(v):
    """None, a plain int, or a tuple of plain ints / tuples of plain ints.
    bool/float are rejected: True == 1 and 1.0 == 1 would otherwise match
    legal actions. No action value nests deeper than two levels."""
    if v is None or type(v) is int:
        return True
    if type(v) is not tuple:
        return False
    for x in v:
        tx = type(x)
        if tx is int:
            continue
        if tx is not tuple:
            return False
        for y in x:
            if type(y) is not int:
                return False
    return True


class Game:
    def __init__(self, num_players: int = 4, seed: int | None = None,
                 config: GameConfig | None = None, layout: BoardLayout | None = None):
        self.config = config or GameConfig(num_players=num_players)
        n = self.config.num_players
        if not 2 <= n <= 4:
            raise ValueError("num_players must be 2..4 (2 is an unofficial variant)")
        self.seed = seed if seed is not None else random.randrange(2**31)
        self.rng = random.Random(self.seed)
        # Always consume the layout draws so a given seed yields the same dice and
        # deck whether or not an explicit layout is passed (replay with layout=).
        generated = generate_layout(self.rng, self.config.shuffle_ports)
        self.layout = layout or generated
        self._vertex_port = self.layout.vertex_ports()
        lay = self.layout
        self._hex_res = tuple(-1 if t == DESERT else RESOURCES.index(t) for t in lay.hex_terrain)
        self._hexes_by_number = {}
        for h, num in enumerate(lay.hex_number):
            if num is not None:
                self._hexes_by_number.setdefault(num, []).append(h)
        self.n = n
        self.players = [Player() for _ in range(n)]
        self.bank = [BANK_START] * 5
        deck = [t for t, c in enumerate(DEV_DECK) for _ in range(c)]
        self.rng.shuffle(deck)
        self.dev_deck = deck
        self.vertex_owner = [-1] * NV
        self.vertex_level = [0] * NV       # 1 settlement, 2 city
        self.edge_owner = [-1] * NE
        self.robber = self.layout.desert_hex
        self.phase = Phase.SETUP_SETTLEMENT
        self.setup_order = list(range(n)) + list(reversed(range(n)))
        self.setup_index = 0
        self.current = 0
        self.turn = 0                  # increments on every END_TURN after setup
        self.rolled = False
        self.dev_played = False
        self.last_roll: tuple | None = None
        self.discard_remaining = [0] * n
        self.free_roads = 0
        self.last_settlement = -1      # setup: road must touch this vertex
        self.resume_phase = Phase.MAIN  # where knight/robber flow returns
        self.trade: dict | None = None
        self.trades_this_turn = 0
        self.longest_road_owner = -1
        self.largest_army_owner = -1
        self.road_length = [0] * n
        self.winner = -1
        self.truncated = False
        self.action_log: list[Action] = []

    # ------------------------------------------------- copy-on-write members
    # clone() shares the RNG and the action log with its source instead of
    # copying them (together ~70% of clone time; search clones replace the
    # RNG right away and rarely read the log). Each game still behaves as if
    # it owned a private copy taken at clone time:
    # - rng: both games are flagged shared; the first access through
    #   ``.rng`` on either side copies the state. Do not keep a reference
    #   to ``game.rng`` across clone() and draw from it afterwards.
    # - action_log: the clone keeps (source list, length) plus its own tail;
    #   the source only ever appends, so the prefix is stable. Reading
    #   ``.action_log`` on the clone materializes a private list.
    @property
    def rng(self) -> random.Random:
        if self._rng_shared:
            r = random.Random.__new__(random.Random)  # setstate overwrites everything
            r.setstate(self._rng.getstate())
            self._rng = r
            self._rng_shared = False
        return self._rng

    @rng.setter
    def rng(self, r: random.Random):
        self._rng = r
        self._rng_shared = False

    @property
    def action_log(self) -> list[Action]:
        base = self._log_base
        if base is not None:
            self._log = base[:self._log_n] + self._log
            self._log_base = None
        return self._log

    @action_log.setter
    def action_log(self, log: list[Action]):
        self._log = log
        self._log_base = None

    # ------------------------------------------------------------------ queries
    @property
    def is_over(self) -> bool:
        return self.phase == Phase.GAME_OVER

    @property
    def actor(self) -> int:
        """Player who must act next."""
        ph = self.phase
        if ph == Phase.MAIN or ph == Phase.PRE_ROLL:  # hot path first; branches are disjoint
            return self.current
        if ph == Phase.TRADE_RESPONSE:
            return self.trade["pending"][0]
        if ph == Phase.SETUP_SETTLEMENT or ph == Phase.SETUP_ROAD:
            return self.setup_order[self.setup_index]
        if ph == Phase.DISCARD:
            for i in range(self.n):
                p = (self.current + i) % self.n
                if self.discard_remaining[p] > 0:
                    return p
        return self.current

    def victory_points(self, p: int) -> int:
        return self.public_victory_points(p) + self.players[p].dev[VICTORY_POINT]

    def public_victory_points(self, p: int) -> int:
        pl = self.players[p]
        vp = len(pl.settlements) + 2 * len(pl.cities)
        if self.longest_road_owner == p:
            vp += 2
        if self.largest_army_owner == p:
            vp += 2
        return vp

    def supply(self, p: int, piece: str) -> int:
        pl = self.players[p]
        if piece == "road":
            return SUPPLY["road"] - len(pl.roads)
        if piece == "settlement":
            return SUPPLY["settlement"] - len(pl.settlements)
        return SUPPLY[piece] - len(pl.cities)  # KeyError for unknown pieces, as before

    def can_afford(self, p: int, cost) -> bool:
        h = self.players[p].hand
        return (h[0] >= cost[0] and h[1] >= cost[1] and h[2] >= cost[2]
                and h[3] >= cost[3] and h[4] >= cost[4])

    def trade_rate(self, p: int, r: int) -> int:
        ports = self.players[p].ports
        if RESOURCES[r] in ports:
            return 2
        if GENERIC_PORT in ports:
            return 3
        return 4

    # ---- placement legality
    def _settlement_spot_free(self, v: int) -> bool:
        vo = self.vertex_owner
        if vo[v] != -1:
            return False
        for u in VN[v]:
            if vo[u] != -1:
                return False
        return True

    def _road_ok(self, p: int, e: int) -> bool:
        eo = self.edge_owner
        if eo[e] != -1:
            return False
        for v in EV[e]:
            o = self.vertex_owner[v]
            if o == p:
                return True
            if o == -1:
                for x in VE[v]:
                    if x != e and eo[x] == p:
                        return True
        return False

    def legal_road_edges(self, p: int) -> list[int]:
        pl = self.players[p]
        if len(pl.roads) >= SUPPLY["road"]:
            return []
        # Same set as filtering with _road_ok: a free edge is legal iff one end
        # is the player's building, or is unbuilt and touches one of their roads.
        vo, eo = self.vertex_owner, self.edge_owner
        reach = set(pl.settlements)
        reach.update(pl.cities)
        for e in pl.roads:
            for v in EV[e]:
                o = vo[v]
                if o == -1 or o == p:
                    reach.add(v)
        return sorted({e for v in reach for e in VE[v] if eo[e] == -1})

    def legal_settlement_vertices(self, p: int) -> list[int]:
        pl = self.players[p]
        if len(pl.settlements) >= SUPPLY["settlement"]:
            return []
        cand = set()
        for e in pl.roads:
            cand.update(EV[e])
        return sorted(v for v in cand if self._settlement_spot_free(v))

    def _playable_dev(self, p: int, t: int) -> bool:
        pl = self.players[p]
        return (not self.dev_played and pl.dev[t] - pl.dev_new[t] > 0)

    def _robber_victims(self, p: int) -> list[int]:
        out = set()
        for v in T.hex_vertices[self.robber]:
            o = self.vertex_owner[v]
            if o != -1 and o != p and self.hand_count(o) > 0:
                out.add(o)
        return sorted(out)

    def legal_actions(self) -> list[Action]:
        """All legal actions for ``self.actor``. Domestic OFFER_TRADE and
        COUNTER_OFFER are enumerated from TRADE_TEMPLATES; ``is_legal`` also
        accepts any other well-formed offer."""
        ph = self.phase
        p = self.actor
        if ph == Phase.TRADE_RESPONSE:
            tr = self.trade
            sa = _SIMPLE_ACTS[p]
            out = [sa[A.REJECT_TRADE]]
            h = self.players[p].hand
            g = tr["get"]
            if (h[0] >= g[0] and h[1] >= g[1] and h[2] >= g[2] and h[3] >= g[3] and h[4] >= g[4]):
                out.append(sa[A.ACCEPT_TRADE])
            # Counters depend only on the responder's own hand: checking the
            # proposer's hand here would leak it. Affordability for the
            # proposer is enforced at CONFIRM_TRADE.
            c = _CLAMP2
            acts, index = _COUNTERS[p][c[h[0]] + 3 * c[h[1]] + 9 * c[h[2]] + 27 * c[h[3]] + 81 * c[h[4]]]
            i = index.get((tr["give"], g))  # the offer itself is not a counter
            if i is None:
                out += acts
            else:
                out += acts[:i]
                out += acts[i + 1:]
            return out
        if ph == Phase.MAIN or ph == Phase.PRE_ROLL:
            return self._legal_turn_actions(p, ph)
        if ph == Phase.GAME_OVER:
            return []
        if ph == Phase.SETUP_SETTLEMENT:
            acts = _SETTLE_ACTS[p]
            return [acts[v] for v in range(NV) if self._settlement_spot_free(v)]
        if ph == Phase.SETUP_ROAD:
            acts = _ROAD_ACTS[p]
            return [acts[e] for e in VE[self.last_settlement] if self.edge_owner[e] == -1]
        if ph == Phase.DISCARD:
            h = self.players[p].hand
            acts = _DISCARD_ACTS[p]
            return [acts[r] for r in range(5) if h[r] > 0]
        if ph == Phase.MOVE_ROBBER:
            acts = _ROBBER_ACTS[p]
            return [acts[h] for h in range(NH) if h != self.robber]
        if ph == Phase.STEAL:
            acts = _STEAL_ACTS[p]
            return [acts[v] for v in self._robber_victims(p)]
        if ph == Phase.ROAD_BUILDING:
            acts = _ROAD_ACTS[p]
            return [acts[e] for e in self.legal_road_edges(p)]
        if ph == Phase.TRADE_CONFIRM:
            out = [_SIMPLE_ACTS[p][A.CANCEL_TRADE]]
            acts = _CONFIRM_ACTS[p]
            for q, resp in sorted(self.trade["responses"].items()):
                if resp is not None and self.can_afford(p, resp[0]):
                    out.append(acts[q])
            return out
        return []

    def _legal_turn_actions(self, p, ph):
        """legal_actions() for PRE_ROLL and MAIN."""
        out = []
        pl = self.players[p]
        h = pl.hand
        sa = _SIMPLE_ACTS[p]
        if not self.dev_played:
            d, dn = pl.dev, pl.dev_new
            if d[KNIGHT] > dn[KNIGHT]:
                out.append(sa[A.PLAY_KNIGHT])
            if d[ROAD_BUILDING] > dn[ROAD_BUILDING] and self.legal_road_edges(p):
                out.append(sa[A.PLAY_ROAD_BUILDING])
            if d[YEAR_OF_PLENTY] > dn[YEAR_OF_PLENTY]:
                bank = self.bank
                for a, b, act in _YOP_ACTS[p]:
                    if bank[a] >= (2 if a == b else 1) and bank[b] >= 1:
                        out.append(act)
            if d[MONOPOLY] > dn[MONOPOLY]:
                out += _MONO_ACTS[p]
        if ph == Phase.PRE_ROLL:
            out.append(sa[A.ROLL])
            return out

        out.append(sa[A.END_TURN])
        # COST is read here (not hard-coded) so is_legal and legal_actions can't disagree
        if _affords(h, COST["road"]):
            acts = _ROAD_ACTS[p]
            out += [acts[e] for e in self.legal_road_edges(p)]
        if _affords(h, COST["settlement"]):
            acts = _SETTLE_ACTS[p]
            out += [acts[v] for v in self.legal_settlement_vertices(p)]
        if _affords(h, COST["city"]) and len(pl.cities) < SUPPLY["city"]:
            acts = _CITY_ACTS[p]
            out += [acts[v] for v in pl.settlements]
        if _affords(h, COST["dev"]) and self.dev_deck:
            out.append(sa[A.BUY_DEV])
        ports = pl.ports
        bank = self.bank
        base = 3 if GENERIC_PORT in ports else 4
        for g in range(5):
            if h[g] >= (2 if RESOURCES[g] in ports else base):
                for r, act in _MARITIME_ACTS[p][g]:
                    if bank[r] > 0:
                        out.append(act)
        if self._can_offer(p):
            for gi, (ones, every) in enumerate(_OFFER_ACTS[p]):
                c = h[gi]
                if c >= 2:
                    out += every
                elif c == 1:
                    out += ones
        return out

    def _can_offer(self, p):
        lim = self.config.max_trade_offers_per_turn
        return (lim is None or self.trades_this_turn < lim) and sum(self.players[p].hand) > 0

    def _valid_offer(self, give, get, payer_give: int | None, payer_get: int | None) -> bool:
        if not (_valid_vec(give) and _valid_vec(get)):
            return False
        if sum(give) == 0 or sum(get) == 0:
            return False  # no gifts
        if any(give[i] and get[i] for i in range(5)):
            return False  # no trading like resources
        if payer_give is not None and not all(self.players[payer_give].hand[i] >= give[i] for i in range(5)):
            return False
        if payer_get is not None and not all(self.players[payer_get].hand[i] >= get[i] for i in range(5)):
            return False
        return True

    def is_legal(self, action: Action) -> bool:
        if not isinstance(action, Action):
            return False
        p, t, v = action
        ph = self.phase
        if (type(p) is not int or ph == Phase.GAME_OVER or p != self.actor
                or not (v is None or type(v) is int or id(v) in _TPL_BY_ID or _well_formed(v))):
            return False
        if t == A.OFFER_TRADE and ph == Phase.MAIN:
            if not self._can_offer(p):
                return False
            tpl = _TPL_BY_ID.get(id(v)) or _TPL_INDEX.get(v)  # safe: _well_formed excluded bools
            if tpl is not None:
                return self.players[p].hand[tpl[0]] >= tpl[1]
            return (isinstance(v, tuple) and len(v) == 2
                    and self._valid_offer(v[0], v[1], p, None))
        if t == A.COUNTER_OFFER and ph == Phase.TRADE_RESPONSE:
            tr = self.trade
            if v == (tr["give"], tr["get"]):
                return False
            tpl = _TPL_BY_ID.get(id(v)) or _TPL_INDEX.get(v)
            if tpl is not None:
                return self.players[p].hand[tpl[2]] >= tpl[3]
            return (isinstance(v, tuple) and len(v) == 2
                    and self._valid_offer(v[0], v[1], None, p))
        # Direct checks mirroring legal_actions() (tests/engine/test_oracle.py
        # verifies equivalence) instead of enumerating every legal action.
        is_int = type(v) is int
        if ph == Phase.TRADE_RESPONSE:
            if v is not None:
                return False
            return t == A.REJECT_TRADE or (t == A.ACCEPT_TRADE and self.can_afford(p, self.trade["get"]))
        if ph == Phase.TRADE_CONFIRM:
            if t == A.CANCEL_TRADE:
                return v is None
            if t != A.CONFIRM_TRADE or not is_int:
                return False
            resp = self.trade["responses"].get(v)
            return resp is not None and self.can_afford(p, resp[0])
        if ph == Phase.SETUP_SETTLEMENT:
            return t == A.BUILD_SETTLEMENT and is_int and 0 <= v < NV and self._settlement_spot_free(v)
        if ph == Phase.SETUP_ROAD:
            return (t == A.BUILD_ROAD and is_int and v in VE[self.last_settlement]
                    and self.edge_owner[v] == -1)
        if ph == Phase.DISCARD:
            return t == A.DISCARD and is_int and 0 <= v < 5 and self.players[p].hand[v] > 0
        if ph == Phase.MOVE_ROBBER:
            return t == A.MOVE_ROBBER and is_int and 0 <= v < NH and v != self.robber
        if ph == Phase.STEAL:
            return t == A.STEAL and is_int and v in self._robber_victims(p)
        if ph == Phase.ROAD_BUILDING:
            return (t == A.BUILD_ROAD and is_int and 0 <= v < NE
                    and len(self.players[p].roads) < SUPPLY["road"] and self._road_ok(p, v))
        # PRE_ROLL / MAIN
        if t == A.PLAY_KNIGHT:
            return v is None and self._playable_dev(p, KNIGHT)
        if t == A.PLAY_ROAD_BUILDING:
            return v is None and self._playable_dev(p, ROAD_BUILDING) and bool(self.legal_road_edges(p))
        if t == A.PLAY_YEAR_OF_PLENTY:
            if not (type(v) is tuple and len(v) == 2 and type(v[0]) is int and type(v[1]) is int
                    and 0 <= v[0] <= v[1] < 5 and self._playable_dev(p, YEAR_OF_PLENTY)):
                return False
            return self.bank[v[0]] >= (2 if v[0] == v[1] else 1) and self.bank[v[1]] >= 1
        if t == A.PLAY_MONOPOLY:
            return is_int and 0 <= v < 5 and self._playable_dev(p, MONOPOLY)
        if ph == Phase.PRE_ROLL:
            return t == A.ROLL and v is None
        if ph != Phase.MAIN:
            return False
        pl = self.players[p]
        if t == A.END_TURN:
            return v is None
        if t == A.BUILD_ROAD:
            return (is_int and 0 <= v < NE and self.can_afford(p, COST["road"])
                    and len(pl.roads) < SUPPLY["road"] and self._road_ok(p, v))
        if t == A.BUILD_SETTLEMENT:
            if not (is_int and 0 <= v < NV and self.can_afford(p, COST["settlement"])
                    and len(pl.settlements) < SUPPLY["settlement"] and self._settlement_spot_free(v)):
                return False
            eo = self.edge_owner
            return any(eo[e] == p for e in VE[v])
        if t == A.BUILD_CITY:
            return (is_int and v in pl.settlements and self.can_afford(p, COST["city"])
                    and len(pl.cities) < SUPPLY["city"])
        if t == A.BUY_DEV:
            return v is None and bool(self.dev_deck) and self.can_afford(p, COST["dev"])
        if t == A.MARITIME_TRADE:
            if not (type(v) is tuple and len(v) == 2 and type(v[0]) is int and type(v[1]) is int):
                return False
            g, r = v
            return (0 <= g < 5 and 0 <= r < 5 and g != r and self.bank[r] > 0
                    and pl.hand[g] >= self.trade_rate(p, g))
        return False

    # ------------------------------------------------------------- transitions
    def step(self, action: Action) -> list[dict]:
        """Apply a legal action; returns events. Raises IllegalAction otherwise."""
        if not self.is_legal(action):
            raise IllegalAction(f"illegal action {action} in phase {self.phase} (actor {self.actor})")
        self._log.append(action)  # own list or own tail (see action_log)
        ev: list[dict] = []
        p, t, v = action
        _HANDLERS[t](self, p, v, ev)
        self._check_win(ev)
        return ev

    def _pay(self, p, cost):
        h = self.players[p].hand
        for i in range(5):
            h[i] -= cost[i]
            self.bank[i] += cost[i]

    def _place_settlement(self, p, v):
        self.vertex_owner[v] = p
        self.vertex_level[v] = 1
        self.players[p].settlements.append(v)
        port = self._vertex_port.get(v)
        if port:
            self.players[p].ports.add(port)

    def _do_build_settlement(self, p, v, ev):
        if self.phase == Phase.SETUP_SETTLEMENT:
            self._place_settlement(p, v)
            ev.append({"type": "settlement_built", "player": p, "vertex": v, "free": True})
            if self.setup_index >= self.n:  # second settlement: starting resources
                gained = [0] * 5
                for h in T.vertex_hexes[v]:
                    terr = self.layout.hex_terrain[h]
                    if terr != DESERT:
                        r = RESOURCES.index(terr)
                        if self.bank[r] > 0:
                            self.bank[r] -= 1
                            self.players[p].hand[r] += 1
                            gained[r] += 1
                ev.append({"type": "resources_gained", "player": p, "resources": gained, "reason": "setup"})
            self.last_settlement = v
            self.phase = Phase.SETUP_ROAD
            return
        self._pay(p, COST["settlement"])
        self._place_settlement(p, v)
        ev.append({"type": "settlement_built", "player": p, "vertex": v})
        # may break an opponent's road; only roads through v can change
        self._update_longest_road(ev, {self.edge_owner[e] for e in VE[v]} - {-1, p})

    def _do_build_city(self, p, v, ev):
        self._pay(p, COST["city"])
        pl = self.players[p]
        pl.settlements.remove(v)
        pl.cities.append(v)
        self.vertex_level[v] = 2
        ev.append({"type": "city_built", "player": p, "vertex": v})

    def _do_build_road(self, p, e, ev):
        self.edge_owner[e] = p
        self.players[p].roads.append(e)
        if self.phase == Phase.SETUP_ROAD:
            ev.append({"type": "road_built", "player": p, "edge": e, "free": True})
            self.setup_index += 1
            if self.setup_index >= len(self.setup_order):
                self.phase = Phase.PRE_ROLL
                self.current = 0
                ev.append({"type": "turn_started", "player": 0, "turn": 0})
            else:
                self.phase = Phase.SETUP_SETTLEMENT
            self._update_longest_road(ev, (p,))
            return
        if self.phase == Phase.ROAD_BUILDING:
            self.free_roads -= 1
            ev.append({"type": "road_built", "player": p, "edge": e, "free": True})
            self._update_longest_road(ev, (p,))
            if self.free_roads <= 0 or not self.legal_road_edges(p):
                self.free_roads = 0
                self.phase = self.resume_phase
            return
        self._pay(p, COST["road"])
        ev.append({"type": "road_built", "player": p, "edge": e})
        self._update_longest_road(ev, (p,))

    def _do_roll(self, p, v, ev):
        rng = self.rng
        d1, d2 = rng.randint(1, 6), rng.randint(1, 6)
        self.last_roll = (d1, d2)
        self.rolled = True
        total = d1 + d2
        ev.append({"type": "dice_rolled", "player": p, "dice": [d1, d2], "total": total})
        if total == 7:
            lim = self.config.discard_limit
            for q in range(self.n):
                c = sum(self.players[q].hand)
                self.discard_remaining[q] = c // 2 if c > lim else 0
            self.resume_phase = Phase.MAIN
            if any(self.discard_remaining):
                self.phase = Phase.DISCARD
                ev.append({"type": "discard_required", "amounts": self.discard_remaining[:]})
            else:
                self.phase = Phase.MOVE_ROBBER
            return
        self._produce(total, ev)
        self.phase = Phase.MAIN

    def _produce(self, total, ev):
        n, vo, lv, robber, hex_res = self.n, self.vertex_owner, self.vertex_level, self.robber, self._hex_res
        owed = None
        for h in self._hexes_by_number.get(total, ()):
            if h == robber:
                continue
            r = hex_res[h]
            for v in T.hex_vertices[h]:
                o = vo[v]
                if o != -1:
                    if owed is None:
                        owed = [[0] * 5 for _ in range(n)]
                    owed[o][r] += lv[v]
        if owed is None:
            return  # nothing adjacent: no gains, no events
        bank = self.bank
        gained = [[0] * 5 for _ in range(n)]
        for r in range(5):
            need = 0
            claimants = []
            for q in range(n):
                x = owed[q][r]
                if x > 0:
                    need += x
                    claimants.append(q)
            if need == 0:
                continue
            if need <= bank[r]:
                for q in claimants:
                    gained[q][r] = owed[q][r]
            elif len(claimants) == 1:
                gained[claimants[0]][r] = bank[r]
            # else: shortage affecting several players -> nobody receives it
            for q in claimants:
                bank[r] -= gained[q][r]
                self.players[q].hand[r] += gained[q][r]
        for q in range(n):
            if any(gained[q]):
                ev.append({"type": "resources_gained", "player": q, "resources": gained[q], "reason": "production"})

    def _do_discard(self, p, r, ev):
        self.players[p].hand[r] -= 1
        self.bank[r] += 1
        self.discard_remaining[p] -= 1
        ev.append({"type": "discarded", "player": p, "resource": r})
        if not any(self.discard_remaining):
            self.phase = Phase.MOVE_ROBBER

    def _do_move_robber(self, p, h, ev):
        self.robber = h
        ev.append({"type": "robber_moved", "player": p, "hex": h})
        victims = self._robber_victims(p)
        if len(victims) == 1:
            self._steal(p, victims[0], ev)
        elif len(victims) > 1:
            self.phase = Phase.STEAL
            return
        self.phase = self.resume_phase

    def _steal(self, p, victim, ev):
        hand = self.players[victim].hand
        pool = [r for r in range(5) for _ in range(hand[r])]
        r = pool[self.rng.randrange(len(pool))]
        hand[r] -= 1
        self.players[p].hand[r] += 1
        ev.append({"type": "stolen", "thief": p, "victim": victim})
        ev.append({"type": "stolen_detail", "thief": p, "victim": victim, "resource": r,
                   "visible_to": [p, victim]})

    def _do_steal(self, p, victim, ev):
        self._steal(p, victim, ev)
        self.phase = self.resume_phase

    def _do_buy_dev(self, p, v, ev):
        self._pay(p, COST["dev"])
        card = self.dev_deck.pop()
        pl = self.players[p]
        pl.dev[card] += 1
        pl.dev_new[card] += 1
        ev.append({"type": "dev_bought", "player": p})
        ev.append({"type": "dev_bought_detail", "player": p, "card": DEV_TYPES[card], "visible_to": [p]})

    def _use_dev(self, p, t, ev):
        self.players[p].dev[t] -= 1
        self.dev_played = True
        ev.append({"type": "dev_played", "player": p, "card": DEV_TYPES[t]})

    def _back_phase(self):
        return Phase.MAIN if self.rolled else Phase.PRE_ROLL

    def _do_play_knight(self, p, v, ev):
        self._use_dev(p, KNIGHT, ev)
        self.players[p].knights += 1
        self._update_largest_army(p, ev)
        self.resume_phase = self._back_phase()
        self.phase = Phase.MOVE_ROBBER

    def _do_play_road_building(self, p, v, ev):
        self._use_dev(p, ROAD_BUILDING, ev)
        self.free_roads = min(2, self.supply(p, "road"))
        self.resume_phase = self._back_phase()
        self.phase = Phase.ROAD_BUILDING

    def _do_play_year_of_plenty(self, p, v, ev):
        self._use_dev(p, YEAR_OF_PLENTY, ev)
        gained = [0] * 5
        for r in v:
            self.bank[r] -= 1
            self.players[p].hand[r] += 1
            gained[r] += 1
        ev.append({"type": "resources_gained", "player": p, "resources": gained, "reason": "year_of_plenty"})

    def _do_play_monopoly(self, p, r, ev):
        self._use_dev(p, MONOPOLY, ev)
        total = 0
        for q in range(self.n):
            if q != p:
                total += self.players[q].hand[r]
                self.players[q].hand[r] = 0
        self.players[p].hand[r] += total
        ev.append({"type": "monopoly", "player": p, "resource": r, "amount": total})

    def _do_maritime_trade(self, p, v, ev):
        g, r = v
        rate = self.trade_rate(p, g)
        h = self.players[p].hand
        h[g] -= rate
        self.bank[g] += rate
        h[r] += 1
        self.bank[r] -= 1
        ev.append({"type": "maritime_trade", "player": p, "give": g, "amount": rate, "get": r})

    def _do_offer_trade(self, p, v, ev):
        give, get = v
        self.trades_this_turn += 1
        pending = [(p + i) % self.n for i in range(1, self.n)]
        self.trade = {"proposer": p, "give": give, "get": get, "pending": pending,
                      "responses": {}}
        ev.append({"type": "trade_offered", "player": p, "give": list(give), "get": list(get)})
        self.phase = Phase.TRADE_RESPONSE

    def _next_responder(self, ev):
        tr = self.trade
        tr["pending"].pop(0)
        if tr["pending"]:
            return
        if any(r is not None for r in tr["responses"].values()):
            self.phase = Phase.TRADE_CONFIRM
        else:
            ev.append({"type": "trade_cancelled", "player": tr["proposer"], "reason": "no_takers"})
            self.trade = None
            self.phase = Phase.MAIN

    def _do_accept_trade(self, p, v, ev):
        self.trade["responses"][p] = (self.trade["give"], self.trade["get"])
        ev.append({"type": "trade_response", "player": p, "response": "accept"})
        self._next_responder(ev)

    def _do_reject_trade(self, p, v, ev):
        self.trade["responses"][p] = None
        ev.append({"type": "trade_response", "player": p, "response": "reject"})
        self._next_responder(ev)

    def _do_counter_offer(self, p, v, ev):
        self.trade["responses"][p] = (v[0], v[1])
        ev.append({"type": "trade_response", "player": p, "response": "counter",
                   "give": list(v[0]), "get": list(v[1])})
        self._next_responder(ev)

    def _do_confirm_trade(self, p, q, ev):
        give, get = self.trade["responses"][q]
        a, b = self.players[p].hand, self.players[q].hand
        for i in range(5):
            if a[i] < give[i] or b[i] < get[i]:
                raise IllegalAction("trade parties can no longer pay")  # unreachable: hands frozen
        for i in range(5):
            a[i] += get[i] - give[i]
            b[i] += give[i] - get[i]
        ev.append({"type": "trade_executed", "player": p, "partner": q, "give": list(give), "get": list(get)})
        self.trade = None
        self.phase = Phase.MAIN

    def _do_cancel_trade(self, p, v, ev):
        ev.append({"type": "trade_cancelled", "player": p, "reason": "proposer"})
        self.trade = None
        self.phase = Phase.MAIN

    def _do_end_turn(self, p, v, ev):
        pl = self.players[p]
        pl.dev_new = [0] * 5
        self.current = (self.current + 1) % self.n
        self.turn += 1
        self.rolled = False
        self.dev_played = False
        self.trades_this_turn = 0
        self.phase = Phase.PRE_ROLL
        ev.append({"type": "turn_started", "player": self.current, "turn": self.turn})
        mt = self.config.max_turns
        if mt is not None and self.turn >= mt and self.victory_points(self.current) < self.config.vp_to_win:
            self.truncated = True
            self.phase = Phase.GAME_OVER
            ev.append({"type": "game_truncated", "turn": self.turn})

    # ----------------------------------------------------------------- awards
    def longest_road(self, p: int) -> int:
        edges = self.players[p].roads
        if not edges:
            return 0
        vo = self.vertex_owner
        blocked = frozenset(v for e in edges for v in EV[e] if vo[v] != -1 and vo[v] != p)
        return _longest_trail(frozenset(edges), blocked)

    def _update_longest_road(self, ev, players=None):
        """Recompute road lengths of ``players`` (default all; the others
        cannot have changed) and reassign the award."""
        lengths = self.road_length
        for q in (range(self.n) if players is None else players):
            lengths[q] = self.longest_road(q)
        holder = self.longest_road_owner
        best = max(lengths)
        if holder != -1 and lengths[holder] >= 5 and lengths[holder] >= best:
            return
        cands = [q for q in range(self.n) if lengths[q] == best]
        new = cands[0] if best >= 5 and len(cands) == 1 else -1
        if new != holder:
            self.longest_road_owner = new
            ev.append({"type": "award_changed", "award": "longest_road", "from": holder, "to": new})

    def _update_largest_army(self, p, ev):
        k = self.players[p].knights
        holder = self.largest_army_owner
        if k >= 3 and holder != p and (holder == -1 or k > self.players[holder].knights):
            self.largest_army_owner = p
            ev.append({"type": "award_changed", "award": "largest_army", "from": holder, "to": p})

    def _check_win(self, ev):
        ph = self.phase
        if ph == Phase.SETUP_SETTLEMENT or ph == Phase.SETUP_ROAD or ph == Phase.GAME_OVER:
            return
        p = self.current
        pl = self.players[p]  # victory_points(p), inlined: this runs after every step
        vp = len(pl.settlements) + 2 * len(pl.cities) + pl.dev[VICTORY_POINT]
        if self.longest_road_owner == p:
            vp += 2
        if self.largest_army_owner == p:
            vp += 2
        if vp >= self.config.vp_to_win:
            self.winner = p
            self.phase = Phase.GAME_OVER
            ev.append({"type": "game_won", "player": p, "victory_points": self.victory_points(p)})

    # -------------------------------------------------------- serialization
    def to_dict(self) -> dict:
        st = self._rng.getstate()  # read-only: no need to unshare
        return {
            "version": 1,
            "seed": self.seed,
            "config": self.config.to_dict(),
            "layout": self.layout.to_dict(),
            "rng_state": [st[0], list(st[1]), st[2]],
            "players": [pl.to_dict() for pl in self.players],
            "bank": self.bank[:],
            "dev_deck": self.dev_deck[:],
            "vertex_owner": self.vertex_owner[:],
            "vertex_level": self.vertex_level[:],
            "edge_owner": self.edge_owner[:],
            "robber": self.robber,
            "phase": self.phase,
            "setup_index": self.setup_index,
            "current": self.current,
            "turn": self.turn,
            "rolled": self.rolled,
            "dev_played": self.dev_played,
            "last_roll": list(self.last_roll) if self.last_roll else None,
            "discard_remaining": self.discard_remaining[:],
            "free_roads": self.free_roads,
            "last_settlement": self.last_settlement,
            "resume_phase": self.resume_phase,
            "trade": None if self.trade is None else {
                "proposer": self.trade["proposer"], "give": list(self.trade["give"]),
                "get": list(self.trade["get"]), "pending": self.trade["pending"][:],
                "responses": {str(k): (None if r is None else [list(r[0]), list(r[1])])
                              for k, r in self.trade["responses"].items()}},
            "trades_this_turn": self.trades_this_turn,
            "longest_road_owner": self.longest_road_owner,
            "largest_army_owner": self.largest_army_owner,
            "road_length": self.road_length[:],
            "winner": self.winner,
            "truncated": self.truncated,
            "action_log": [a.to_json() for a in self.action_log],
        }

    @staticmethod
    def from_dict(d: dict) -> "Game":
        cfg = GameConfig(**d["config"])
        g = Game(config=cfg, seed=d["seed"], layout=BoardLayout.from_dict(d["layout"]))
        rs = d["rng_state"]
        g.rng.setstate((rs[0], tuple(rs[1]), rs[2]))
        g.players = [Player.from_dict(x) for x in d["players"]]
        for k in ("bank", "dev_deck", "vertex_owner", "vertex_level", "edge_owner",
                  "discard_remaining", "road_length"):
            setattr(g, k, list(d[k]))
        for k in ("robber", "phase", "setup_index", "current", "turn", "rolled", "dev_played",
                  "free_roads", "last_settlement", "resume_phase", "trades_this_turn",
                  "longest_road_owner", "largest_army_owner", "winner", "truncated"):
            setattr(g, k, d[k])
        g.last_roll = tuple(d["last_roll"]) if d["last_roll"] else None
        tr = d["trade"]
        if tr is not None:
            g.trade = {"proposer": tr["proposer"], "give": tuple(tr["give"]), "get": tuple(tr["get"]),
                       "pending": list(tr["pending"]),
                       "responses": {int(k): (None if r is None else (tuple(r[0]), tuple(r[1])))
                                     for k, r in tr["responses"].items()}}
        g.action_log = [Action.from_json(a) for a in d["action_log"]]
        return g

    def clone(self) -> "Game":
        """Fast deep copy for search/rollouts (shares immutable layout; RNG and
        action log are copy-on-write, see the rng/action_log properties)."""
        g = Game.__new__(Game)
        g.__dict__.update(self.__dict__)
        self._rng_shared = g._rng_shared = True  # copied on first use (see rng)
        self._share_log(g)
        g.players = [pl.copy() for pl in self.players]
        g.bank = list(self.bank)
        g.dev_deck = list(self.dev_deck)
        g.vertex_owner = list(self.vertex_owner)
        g.vertex_level = list(self.vertex_level)
        g.edge_owner = list(self.edge_owner)
        g.discard_remaining = list(self.discard_remaining)
        g.road_length = list(self.road_length)
        g.setup_order = list(self.setup_order)
        if self.trade is not None:
            g.trade = dict(self.trade, pending=list(self.trade["pending"]),
                           responses=dict(self.trade["responses"]))
        return g

    def _share_log(self, g: "Game") -> None:
        """Give ``g`` a copy-on-write view of this game's action log."""
        if self._log_base is None:
            g._log_base, g._log_n, g._log = self._log, len(self._log), []
        else:
            g._log_base, g._log_n = self._log_base, self._log_n
            g._log = list(self._log)  # share the same base, copy the short tail

    # ------------------------------------------------------ information sets
    _reveals_when_over = True  # player_view shows every hand once the game is over

    def hand_count(self, p: int) -> int:
        """Public: number of resource cards seat ``p`` holds."""
        return sum(self.players[p].hand)

    def dev_count(self, p: int) -> int:
        """Public: number of unplayed dev cards seat ``p`` holds."""
        return sum(self.players[p].dev)

    def dev_new_count(self, p: int) -> int:
        """Public: number of dev cards seat ``p`` bought this turn."""
        return sum(self.players[p].dev_new)

    def unseen_dev_pool(self, me: int) -> list[int]:
        """Sorted multiset of dev cards ``me`` cannot see (deck + opponents' unplayed).
        Its composition is public (all cards minus mine minus the played ones);
        sorting hides order and ownership."""
        pool = list(self.dev_deck)
        for q in range(self.n):
            if q != me:
                for t in range(5):
                    pool.extend([t] * self.players[q].dev[t])
        return sorted(pool)

    def observe_for(self, seat: int) -> "Game":
        """Information-set view for ``seat`` (catan.engine.infoset.InfoSetGame):
        public state plus the seat's own cards; hidden state raises."""
        from .infoset import observe_for
        return observe_for(self, seat)

    def _determinization_base(self) -> "Game":
        return self.clone()

    def determinized(self, me: int, rng: random.Random, trade_safe: bool = True) -> "Game":
        """A clone with everything hidden from ``me`` resampled: opponents'
        hands (keeping per-player counts and the public per-resource totals),
        opponents' dev cards (keeping counts), deck order and the RNG. Reads
        only what ``me`` may know, so it also works on an InfoSetGame.

        With ``trade_safe`` a pending domestic trade stays executable: the
        proposer holds what they offered and counter-offerers what they asked for
        (rejection sampling, 50 tries, then the constraint is dropped)."""
        c = self._determinization_base()
        opps = [q for q in range(self.n) if q != me]
        own = self.players[me].hand
        totals = [BANK_START - self.bank[r] - own[r] for r in range(5)]  # public by conservation
        base_pool = [r for r in range(5) for _ in range(totals[r])]
        counts = [self.hand_count(q) for q in opps]
        req = {}
        if trade_safe and self.trade is not None:
            tr = self.trade
            if tr["proposer"] != me:
                req[tr["proposer"]] = list(tr["give"])
            for q, resp in tr["responses"].items():
                if resp is not None and q != me:
                    need = req.get(q, [0] * 5)
                    req[q] = [max(need[i], resp[1][i]) for i in range(5)]
        for _ in range(50):
            pool = base_pool[:]
            rng.shuffle(pool)
            hands, i = {}, 0
            for q, k in zip(opps, counts):
                h = [0] * 5
                for r in pool[i:i + k]:
                    h[r] += 1
                hands[q] = h
                i += k
            if all(all(hands[q][j] >= need[j] for j in range(5)) for q, need in req.items()):
                break
        for q in opps:
            c.players[q].hand = hands[q]
        dpool = self.unseen_dev_pool(me)
        rng.shuffle(dpool)
        i = 0
        for q in opps:
            k = self.dev_count(q)
            new_k = self.dev_new_count(q)  # cards bought this turn: count is public
            d, dn = [0] * 5, [0] * 5
            for j, t in enumerate(dpool[i:i + k]):
                d[t] += 1
                if j < new_k:
                    dn[t] += 1
            c.players[q].dev, c.players[q].dev_new = d, dn
            i += k
        c.dev_deck = dpool[i:]
        c.rng = random.Random(rng.getrandbits(64))
        return c

    @staticmethod
    def replay(seed: int, config: GameConfig, actions, layout: BoardLayout | None = None) -> "Game":
        g = Game(config=config, seed=seed, layout=layout)
        for a in actions:
            g.step(a if isinstance(a, Action) else Action.from_json(a))
        return g

    # ------------------------------------------------------------ views
    def player_view(self, viewer: int | None) -> dict:
        """State as visible to ``viewer`` (None = spectator/public). Hidden:
        opponents' hands and dev cards (counts only), deck order, RNG."""
        players = []
        for q, pl in enumerate(self.players):
            own = viewer == q
            info = {
                "id": q,
                "resource_count": self.hand_count(q),
                "dev_count": self.dev_count(q),
                "knights_played": pl.knights,
                "settlements": pl.settlements[:],
                "cities": pl.cities[:],
                "roads": pl.roads[:],
                "ports": sorted(pl.ports),
                "public_vp": self.public_victory_points(q),
                "road_length": self.road_length[q],
                "supply": {k: self.supply(q, k) for k in SUPPLY},
            }
            if own or (self.is_over and self._reveals_when_over):
                info["hand"] = pl.hand[:]
                info["dev_cards"] = pl.dev[:]
                info["dev_new"] = pl.dev_new[:]
                info["vp"] = self.victory_points(q)
            players.append(info)
        tr = None
        if self.trade is not None:
            tr = {"proposer": self.trade["proposer"], "give": list(self.trade["give"]),
                  "get": list(self.trade["get"]), "pending": self.trade["pending"][:],
                  "responses": {str(k): (None if r is None else {"give": list(r[0]), "get": list(r[1])})
                                for k, r in self.trade["responses"].items()}}
        return {
            "viewer": viewer,
            "num_players": self.n,
            "layout": self.layout.to_dict(),
            "vertex_owner": self.vertex_owner[:],
            "vertex_level": self.vertex_level[:],
            "edge_owner": self.edge_owner[:],
            "robber": self.robber,
            "bank": self.bank[:],
            "dev_deck_count": len(self.dev_deck),
            "phase": self.phase,
            "current": self.current,
            "actor": self.actor if not self.is_over else None,
            "turn": self.turn,
            "rolled": self.rolled,
            "dev_played": self.dev_played,
            "last_roll": list(self.last_roll) if self.last_roll else None,
            "discard_remaining": self.discard_remaining[:],
            "free_roads": self.free_roads,
            "trade": tr,
            "trades_this_turn": self.trades_this_turn,
            "longest_road_owner": self.longest_road_owner,
            "largest_army_owner": self.largest_army_owner,
            "winner": self.winner,
            "truncated": self.truncated,
            "players": players,
            "vp_to_win": self.config.vp_to_win,
        }


_HANDLERS = {t: getattr(Game, "_do_" + t.lower()) for t in ALL_ACTION_TYPES}
