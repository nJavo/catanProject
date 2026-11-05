"""SearchAgent: determinized rollout search over a pruned candidate set.

At MAIN decisions the intermediate heuristic proposes its top
``candidates`` actions. Each candidate is scored by ``rollouts`` simulations:
the hidden state is resampled consistently with what the actor sees
(``common.determinize``: opponents' hands keep their sizes and the public
per-resource totals, opponents' dev cards keep their counts, deck order and
dice are fresh), the candidate is applied, the actor finishes its turn with
the intermediate policy and every seat then plays the basic policy for
``horizon`` turns in total (default 1: just the rest of the actor's own turn,
which captured more strength per ms than a full round in our sweeps; None =
one full round). The leaf value is
``evaluate(me) - opp_weight * max(evaluate(opponent))``. The same
determinizations are reused across candidates (common random numbers).

The budget is a fixed number of rollouts, so the agent is deterministic given
its seed. ``time_limit`` (seconds) optionally stops early; that trades
determinism for a hard latency bound, so it defaults to off.
"""
from __future__ import annotations

import random
import time

from catan.engine.actions import A, Phase

from . import common as C
from .basic import BasicHeuristicAgent
from .belief import HandPosterior
from .intermediate import IntermediateAgent, evaluate


class SearchAgent(IntermediateAgent):
    name = "search"

    def __init__(self, seed=None, rollouts: int = 8, candidates: int = 6, horizon: int | None = 1,
                 opp_weight: float = 0.3, margin: float = 0.6, time_limit: float | None = None, max_offers: int = 2,
                 weights: dict | None = None, domestic: bool = True, beliefs: bool = False):
        super().__init__(seed, max_offers=max_offers, weights=weights, domestic=domestic)
        # beliefs: sample opponents' hands from the exact public-event posterior (belief.py)
        # instead of uniformly. Needs observe() calls from the host; otherwise falls back.
        self.use_beliefs = beliefs
        self.posterior = None
        self.belief_fallbacks = 0
        self.rollouts = rollouts
        self.candidates = candidates
        self.horizon = horizon
        self.opp_weight = opp_weight
        self.margin = margin
        self.time_limit = time_limit
        self._me_policy = IntermediateAgent(max_offers=max_offers, weights=weights, domestic=domestic)
        self._opp_policy = BasicHeuristicAgent()

    def reset(self, seat, seed=None):
        super().reset(seat, seed)
        self.posterior = None

    def observe(self, game, events):
        if not self.use_beliefs:
            return
        if self.posterior is None:
            self.posterior = HandPosterior(game.n, self.seat if self.seat is not None else 0)
        self.posterior.observe(game, events)

    def _belief(self, g, me):
        p = self.posterior
        if not self.use_beliefs:
            return None
        if p is None or p.lost or p.me != me or p.synced != len(g.action_log):
            self.belief_fallbacks += 1
            return None
        return p

    def main(self, g, me, legal):
        return self._search(g, me, legal, super().main(g, me, legal))

    def _search(self, g, me, legal, scored):
        legal_set = set(legal)
        ranked = sorted(((s, a) for s, a in scored if a in legal_set
                         or (a.type == A.OFFER_TRADE and g.is_legal(a))), key=lambda x: -x[0])
        cands, seen = [], set()
        for s, a in ranked:
            if a not in seen:
                seen.add(a)
                cands.append(a)
            if len(cands) >= self.candidates:
                break
        if len(cands) <= 1 or self.rollouts <= 0 or ranked[0][0] - ranked[1][0] > self.margin:
            self.last_search = None  # read by explain.py: the heuristic decided alone
            return scored  # nothing to decide, or the heuristic is already sure
        t0 = time.perf_counter()
        self._post = self._belief(g, me)
        seeds = [self.rng.getrandbits(64) for _ in range(self.rollouts)]
        totals = {a: 0.0 for a in cands}
        self.last_search = {a: [] for a in cands}  # per-rollout leaf values, read by explain.py
        done = 0
        for s in seeds:
            for a in cands:
                v = self._rollout(g, me, a, s)
                totals[a] += v
                self.last_search[a].append(v)
            done += 1
            if self.time_limit is not None and time.perf_counter() - t0 > self.time_limit:
                break
        # tiny prior from the heuristic rank breaks exact ties deterministically
        return [(totals[a] / done - 1e-6 * i, a) for i, a in enumerate(cands)]

    def _rollout(self, g, me, action, seed) -> float:
        rng = random.Random(seed)
        c = C.determinize(g, me, rng, belief=getattr(self, "_post", None))
        c.step(action)
        horizon = self.horizon if self.horizon is not None else g.n
        stop_turn = g.turn + horizon
        steps = 0
        while not c.is_over and steps < 600:
            if c.turn >= stop_turn and c.phase == Phase.PRE_ROLL:
                break
            p = c.actor
            legal = c.legal_actions()
            if p == me and c.current == me and c.turn == g.turn:
                pol = self._me_policy
            else:
                pol = self._opp_policy
            a = pol.act(c, legal)
            if a.type == A.OFFER_TRADE and p != me:
                a = next(x for x in legal if x.type == A.END_TURN)
            c.step(a)
            steps += 1
        bi = C.board_info(c.layout)
        mine = evaluate(c, me, bi, self.w)
        if c.is_over:
            return mine
        opp = max(evaluate(c, q, bi, self.w) for q in range(c.n) if q != me)
        return mine - self.opp_weight * opp
