"""``jointb_gated``: value_trade-v1 with the jointb trade head used only against negotiators (D18 follow-up).

    jointb_gated                                 # bundle models/jointb_best.pt, classifier models/opp_best.pt
    jointb_gated:checkpoint=...,opp=models/opp_best.pt,threshold=0.1

D21's public-log classifier labels each opponent seat. A seat counts as basic when P(basic) >= threshold (default 0.1: basic seats dip to ~0.2 in the first turns, negotiators sit near 0).
Per trade decision the trade layer is:
  TRADE_RESPONSE  v1's nego layer if the proposer is basic, else the jointb head;
  TRADE_CONFIRM   v1's if every responder I can confirm is basic, else jointb;
  MAIN (offer)    v1's if every opponent is basic, else jointb.
Both layers see every observe/reset, so their belief trackers stay current. Against 3x basic, once
the classifier is sure, this plays exactly as value_trade-v1 (same core, same nego layer and seed).
"""
from __future__ import annotations

from catan.agents.registry import register
from catan.engine.actions import A, Phase
from catan.research.combo.agent import DEFAULT_NEGO
from catan.research.nego.agent import NegoAgent
from catan.research.opp.classifier import load as load_opp
from catan.research.opp.features import CLASSES, seat_features

from .agent import DEFAULT_BUNDLE, JointBValueTradeAgent

BASIC = CLASSES.index("basic")
_CLF: dict = {}


class JointBGatedAgent(JointBValueTradeAgent):
    name = "jointb_gated"

    def __init__(self, checkpoint: str = DEFAULT_BUNDLE, opp: str = "models/opp_best.pt", threshold: float = 0.1,
                 nego_checkpoint: str = DEFAULT_NEGO, seed=None, **kw):
        super().__init__(checkpoint=checkpoint, seed=seed, **kw)
        self.trade_b = self.trade
        self.trade_v1 = NegoAgent(seed=seed, checkpoint=nego_checkpoint)  # = value_trade-v1's trade layer
        if opp not in _CLF:
            _CLF[opp] = load_opp(opp)
        self.clf, self.threshold = _CLF[opp], float(threshold)
        self._key, self._basic = None, set()
        self.uses = {"v1": 0, "jointb": 0}

    def reset(self, seat, seed=None):
        self.trade = self.trade_v1
        super().reset(seat, seed)
        self.trade_b.reset(seat, seed)
        self._key, self._basic = None, set()

    def observe(self, game, events):
        self.trade_v1.observe(game, events)
        self.trade_b.observe(game, events)

    def basic_seats(self, game) -> set:
        log = game.action_log
        if self._key != len(log):
            p = self.clf.proba(seat_features(game.layout, game.config, log))
            self._basic = {q for q in range(game.n) if p[q][BASIC] >= self.threshold}
            self._key = len(log)
        return self._basic

    def _layer(self, game, legal):
        ph, me = game.phase, game.actor
        if ph == Phase.TRADE_RESPONSE:
            partners = {game.trade["proposer"]}
        elif ph == Phase.TRADE_CONFIRM:
            partners = {a.value for a in legal if a.type == A.CONFIRM_TRADE}
        elif ph == Phase.MAIN and any(a.type == A.OFFER_TRADE for a in legal):
            partners = {q for q in range(game.n) if q != me}
        else:
            return None
        return self.trade_v1 if partners <= self.basic_seats(game) else self.trade_b

    def act_with_info(self, game, legal, explain=True):
        layer = self._layer(game, legal)
        if layer is not None:
            self.trade = layer
            self.uses["v1" if layer is self.trade_v1 else "jointb"] += 1
        a, info = super().act_with_info(game, legal, explain=explain)
        if layer is not None:
            info["trade_layer"] = "v1" if layer is self.trade_v1 else "jointb"
        return a, info


register("jointb_gated", lambda seed=None, **kw: JointBGatedAgent(seed=seed, **kw))
