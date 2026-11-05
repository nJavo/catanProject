"""Exact posterior over opponents' hands from one seat's public view.

Every resource movement in this engine is public except the card taken in a
steal, which only the thief and the victim see. So the joint distribution of
the opponents' hands, given everything ``me`` has seen, is a finite mixture:
start from the known setup hands, apply each public movement to every state,
branch on each unseen steal (card r with probability victim[r] / victim
total), and drop states that contradict a public fact:

- a build, dev purchase, bank trade, discard or executed trade the player
  could not have paid for;
- an offer or counter the player could not cover (the engine only allows
  covered offers), or an accept without the requested cards;
- a Monopoly haul whose amount differs from what the state holds.

States with identical hands are merged, so the mixture stays small (usually
1 to a few hundred states). Above ``max_states`` the least likely states are
pruned and ``approx`` is set (near-exact). Only events visible to ``me`` are
used (``visible_to``). The posterior never reads hidden state; the tests
check that it is invariant to resampling hidden information and consistent
with public counts.

``BeliefTracker`` in trading.py keeps only expected hands (a mean field);
this module keeps the full joint distribution, which is what determinization
needs.
"""
from __future__ import annotations

import random

from catan.engine.game import COST, BANK_START

SETTLEMENT, CITY, DEV, ROAD = COST["settlement"], COST["city"], COST["dev"], COST["road"]


def visible(events, me):
    return [e for e in events if e.get("visible_to") is None or me in e["visible_to"]]


class HandPosterior:
    def __init__(self, n: int, me: int, max_states: int = 4000):
        self.n, self.me = n, me
        self.opps = [q for q in range(n) if q != me]
        self.idx = {q: i for i, q in enumerate(self.opps)}
        zero = tuple((0,) * 5 for _ in self.opps)
        self.states: dict[tuple, float] = {zero: 1.0}
        self.max_states = max_states
        self.synced = 0          # actions whose events were observed
        self.lost = False        # a step was missed or the evidence became contradictory
        self.approx = False      # states were pruned
        self.unseen_steals = 0
        self._trade = None

    # ------------------------------------------------------------ updates
    def _map(self, fn):
        out: dict[tuple, float] = {}
        for s, p in self.states.items():
            for s2, f in fn(s):
                out[s2] = out.get(s2, 0.0) + p * f
        tot = sum(out.values())
        if tot <= 0:
            self.lost = True
            return
        if len(out) > self.max_states:
            keep = sorted(out.items(), key=lambda kv: -kv[1])[:self.max_states]
            out = dict(keep)
            tot = sum(out.values())
            self.approx = True
        self.states = {s: p / tot for s, p in out.items()}

    def _add(self, q, vec, sign=1):
        """Add (or, with sign=-1, pay) ``vec`` to opponent q; paying drops states that can't."""
        i = self.idx[q]

        def fn(s):
            h = tuple(s[i][r] + sign * vec[r] for r in range(5))
            if min(h) < 0:
                return ()
            return ((s[:i] + (h,) + s[i + 1:], 1.0),)
        self._map(fn)

    def _require(self, q, vec):
        i = self.idx[q]
        self._map(lambda s: ((s, 1.0),) if all(s[i][r] >= vec[r] for r in range(5)) else ())

    def observe(self, game, events):
        """Feed the events of the action just applied (``len(game.action_log)`` must be ``synced + 1``)."""
        if self.lost or game is not None and self.synced + 1 != len(game.action_log):
            self.lost = True
            return
        self.synced += 1
        self.apply(events)
        if game is not None:
            self.consistent(game)

    def apply(self, events):
        me, opp = self.me, self.idx
        evs = visible(events, me)
        for i, e in enumerate(evs):
            if self.lost:
                return
            t = e["type"]
            p = e.get("player")
            if t == "resources_gained" and p in opp:
                self._add(p, e["resources"])
            elif t == "discarded" and p in opp:
                self._add(p, [int(r == e["resource"]) for r in range(5)], -1)
            elif t == "settlement_built" and p in opp and not e.get("free"):
                self._add(p, SETTLEMENT, -1)
            elif t == "city_built" and p in opp:
                self._add(p, CITY, -1)
            elif t == "road_built" and p in opp and not e.get("free"):
                self._add(p, ROAD, -1)
            elif t == "dev_bought" and p in opp:
                self._add(p, DEV, -1)
            elif t == "maritime_trade" and p in opp:
                self._add(p, [e["amount"] * (r == e["give"]) for r in range(5)], -1)
                self._add(p, [int(r == e["get"]) for r in range(5)])
            elif t == "trade_offered":
                self._trade = (p, list(e["give"]), list(e["get"]))
                if p in opp:
                    self._require(p, e["give"])
            elif t == "trade_response" and p in opp and self._trade is not None:
                if e["response"] == "accept":
                    self._require(p, self._trade[2])
                elif e["response"] == "counter":
                    self._require(p, e["get"])
            elif t == "trade_executed":
                q = e["partner"]
                if p in opp:
                    self._add(p, e["give"], -1)
                    self._add(p, e["get"])
                if q in opp:
                    self._add(q, e["get"], -1)
                    self._add(q, e["give"])
                self._trade = None
            elif t == "trade_cancelled":
                self._trade = None
            elif t == "monopoly":
                self._monopoly(p, e["resource"], e["amount"])
            elif t == "stolen":
                th, v = e["thief"], e["victim"]
                det = next((d for d in evs[i + 1:] if d["type"] == "stolen_detail"
                            and d["thief"] == th and d["victim"] == v), None)
                if det is not None:  # I am the thief or the victim
                    one = [int(r == det["resource"]) for r in range(5)]
                    if v in opp:
                        self._add(v, one, -1)
                    if th in opp:
                        self._add(th, one)
                else:
                    self.unseen_steals += 1
                    self._steal(th, v)

    def _monopoly(self, p, r, amount):
        # The amount also counts my cards, so it is not checked here; the hand-size and
        # per-resource checks in ``consistent`` (run after every observed step) condition on it.
        def fn(s):
            h = list(s)
            for q in self.opps:
                row = list(h[self.idx[q]])
                row[r] = row[r] + amount if q == p else 0
                h[self.idx[q]] = tuple(row)
            return ((tuple(h), 1.0),)
        self._map(fn)

    def _steal(self, th, v):
        iv, it = self.idx[v], self.idx[th]

        def fn(s):
            row = s[iv]
            tot = sum(row)
            out = []
            for r in range(5):
                if row[r]:
                    h = list(s)
                    a = list(h[iv]); a[r] -= 1; h[iv] = tuple(a)
                    b = list(h[it]); b[r] += 1; h[it] = tuple(b)
                    out.append((tuple(h), row[r] / tot))
            return out
        self._map(fn)

    # ------------------------------------------------------------ queries
    def consistent(self, g) -> bool:
        """Drop states that disagree with public counts in ``g`` (hand sizes, and per-resource
        totals 19 - bank - my hand). In an exact posterior none should; returns False and marks
        the posterior lost if nothing survives."""
        if self.lost:
            return False
        sizes = [g.hand_count(q) for q in self.opps]                  # hand sizes are public
        cols = [BANK_START - g.bank[r] - g.players[self.me].hand[r] for r in range(5)]
        keep = {s: p for s, p in self.states.items()
                if [sum(h) for h in s] == sizes and [sum(h[r] for h in s) for r in range(5)] == cols}
        if not keep:
            self.lost = True
            return False
        if len(keep) != len(self.states):
            tot = sum(keep.values())
            self.states = {s: p / tot for s, p in keep.items()}
        return True

    def sample(self, rng: random.Random, require: dict | None = None) -> dict[int, list[int]] | None:
        """Opponent hands drawn from the posterior; ``require`` = {seat: min hand} conditions on
        a pending trade. None if lost or no state satisfies ``require``."""
        if self.lost:
            return None
        items = list(self.states.items())
        if require:
            items = [(s, p) for s, p in items
                     if all(all(s[self.idx[q]][r] >= need[r] for r in range(5)) for q, need in require.items())]
            if not items:
                return None
        x = rng.random() * sum(p for _, p in items)
        for s, p in items:
            x -= p
            if x <= 0:
                break
        return {q: list(s[self.idx[q]]) for q in self.opps}

    def marginals(self) -> dict:
        """Per opponent: expected count and P(count = k) per resource, and whether the hand is known."""
        out = {}
        for q in self.opps:
            i = self.idx[q]
            dist = [dict() for _ in range(5)]
            hands = set()
            for s, p in self.states.items():
                hands.add(s[i])
                for r in range(5):
                    dist[r][s[i][r]] = dist[r].get(s[i][r], 0.0) + p
            out[q] = {"known": len(hands) == 1,
                      "expected": [round(sum(k * p for k, p in d.items()), 4) for d in dist],
                      "dist": [{str(k): round(p, 4) for k, p in sorted(d.items())} for d in dist]}
        return out

    @classmethod
    def from_history(cls, seed, config, actions, me, upto=None):
        """Replay a game's first ``upto`` actions and feed ``me``'s visible events. Info-safe:
        the replay regenerates the true events, and only those visible to ``me`` are read."""
        from catan.engine import Game
        g = Game(config=config, seed=seed)
        post = cls(g.n, me)
        for a in actions[:upto]:
            ev = g.step(a)
            post.observe(g, ev)
        return post, g
