"""NegoAgent: intermediate play with trade decisions from NegoNet.

    NegoAgent(checkpoint="models/nego_best.pt")   # greedy learned trades
    NegoAgent()                                   # no checkpoint: heuristic trader choices

Building, robber and dev-card play come from IntermediateAgent. Every negotiation
decision is a choice among enumerated options (catan/ai/nego/options.py);
the learned head picks one.
"""
from __future__ import annotations

import torch

from catan.bots.intermediate import IntermediateAgent
from catan.bots.trading import TraderMixin
from catan.engine.actions import A

from . import options as O
from .model import batch, load

_NET_CACHE: dict = {}


def _net(path):
    if path not in _NET_CACHE:
        _NET_CACHE[path] = load(path)[0]
    return _NET_CACHE[path]


class NegoMixin:
    def _nego_init(self, checkpoint=None, sample=False, record=False, temperature=1.0, net=None):
        self.net = net if net is not None else (_net(checkpoint) if checkpoint else None)
        self.sample = sample
        self.record = record
        self.temperature = temperature
        self.trajectory: list[dict] = []
        self.gen = torch.Generator().manual_seed(self.seed if self.seed is not None else 0)

    def reset(self, seat, seed=None):
        super().reset(seat, seed)
        self.trajectory = []
        if seed is not None:
            self.gen.manual_seed(seed)

    def _state_vec(self, g, me, est, decision):
        """State features for the head (overridden by D15 to append core-policy features)."""
        return O.state_vec(g, me, est, decision)

    def _nego_choose(self, g, me, decision, opts, teacher, est):
        if len(opts) == 1:
            return opts[0][0], {}
        s = self._state_vec(g, me, est, decision)
        sample = {"s": s, "o": [o[1] for o in opts], "give": [list(o[2]) for o in opts],
                  "get": [list(o[3]) for o in opts]}
        if self.net is None:
            idx, logp = teacher, 0.0
            probs = None
        else:
            with torch.no_grad():
                logits, v = self.net(*batch([sample]))
            logits = logits[0] / max(self.temperature, 1e-6)
            probs = torch.softmax(logits, -1)
            if self.sample:
                idx = int(torch.multinomial(probs, 1, generator=self.gen))
            else:
                idx = int(torch.argmax(logits))
            logp = float(torch.log(probs[idx] + 1e-12))
        if self.record:
            sample.update(a=idx, logp=logp, teacher=teacher, decision=decision, turn=g.turn)
            self.trajectory.append(sample)
        info = {"nego_decision": decision, "teacher": teacher, "chosen": idx}
        if probs is not None:
            info["nego_probs"] = {i: round(float(p), 3) for i, p in enumerate(probs) if p > 0.01}
        # Read by act_with_info for the decision inspector: the learned head's pick and the teacher's.
        self._nego_last = {"decision": decision, "learned": probs is not None, "chosen": idx, "teacher": teacher,
                           "options": [{"action": o[0].to_json(), **({"p": round(float(probs[i]), 4)} if probs is not None else {})}
                                       for i, o in enumerate(opts)]}
        return opts[idx][0], info

    def act_with_info(self, game, legal, explain=True):
        self._nego_last = None
        a, info = super().act_with_info(game, legal, explain)
        n = self._nego_last
        if n and a.to_json() == n["options"][n["chosen"]]["action"]:
            info["nego"] = n  # this move came from the negotiation head, not the heuristic rule in info["explain"]
        return a, info

    # ---- the three negotiation decisions
    def main(self, g, me, legal):
        out = self.base_main(g, me, legal)
        best_v, best_a = max(out, key=lambda x: x[0])
        if (self.domestic and best_a.type in (A.END_TURN, A.MARITIME_TRADE)
                and g.trades_this_turn < self.max_offers
                and any(a.type == A.OFFER_TRADE for a in legal)):
            opts, teacher, est = O.offer_options(self, g, me, best_a)
            a, _ = self._nego_choose(g, me, "offer", opts, teacher, est)
            if a.type == A.OFFER_TRADE:
                return out + [(best_v + 1.0, a)]
        return out

    def trade_response(self, g, me, legal):
        opts, teacher, est = O.respond_options(self, g, me, legal)
        a, _ = self._nego_choose(g, me, "respond", opts, teacher, est)
        return [(1.0, a)]

    def trade_confirm(self, g, me, legal):
        opts, teacher, est = O.confirm_options(self, g, me, legal)
        a, _ = self._nego_choose(g, me, "confirm", opts, teacher, est)
        return [(1.0, a)]


class NegoAgent(NegoMixin, TraderMixin, IntermediateAgent):
    name = "nego"

    def __init__(self, seed=None, checkpoint=None, sample=False, record=False, temperature=1.0, net=None,
                 counters=True, bundle_offers=True, danger_vp=3, **kw):
        IntermediateAgent.__init__(self, seed, **kw)
        self._trader_init(counters, bundle_offers, danger_vp)
        self._nego_init(checkpoint, sample, record, temperature, net)

    def base_main(self, g, me, legal):
        return self.candidate_values(g, me, [a for a in legal if a.type != A.OFFER_TRADE])
