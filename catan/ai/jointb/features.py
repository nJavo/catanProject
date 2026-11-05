"""Per-opponent trade-history features (D18) and the extended core feature map.

``opp_history(g, me)`` reads only the public action log. For each opponent in
relative seat order (as in ``negotiation_features``):
  offers per own turn, log-count of responses, smoothed accept rate,
  smoothed counter rate.
Rule bots and negotiators separate on these: basic never offers or counters and
accepts rarely; intermediate_trader / nego / value_trade offer and counter.

``JointBCore`` = D15 CoreFeatures + opp_history, so JointTradeAgent's state is
nego state + [v_gnn, p_win, emb256] + opp_history.
"""
from __future__ import annotations

import math

from catan.engine.actions import A
from catan.ai.joint.features import EXTRA_F, CoreFeatures

OPP_K = 4
_RESP = (A.ACCEPT_TRADE, A.REJECT_TRADE, A.COUNTER_OFFER)


def opp_features_n(n: int = 4) -> int:
    return OPP_K * (n - 1)


OPP_F = opp_features_n(4)
EXTRA_B = EXTRA_F + OPP_F


def opp_history(g, me) -> list[float]:
    n = g.n
    offers, turns, acc, cnt, resp = ([0] * n for _ in range(5))
    for a in g.action_log:
        t = a.type
        if t == A.OFFER_TRADE:
            offers[a.player] += 1
        elif t == A.END_TURN:
            turns[a.player] += 1
        elif t in _RESP:
            resp[a.player] += 1
            if t == A.ACCEPT_TRADE:
                acc[a.player] += 1
            elif t == A.COUNTER_OFFER:
                cnt[a.player] += 1
    x = []
    for k in range(1, n):
        q = (me + k) % n
        r = resp[q]
        x += [min(offers[q] / max(1, turns[q]), 3.0) / 3.0, math.log1p(r) / 4.0,
              (acc[q] + 0.5) / (r + 2.0), (cnt[q] + 0.5) / (r + 2.0)]
    return x


class JointBCore(CoreFeatures):
    def __call__(self, g, me) -> list[float]:
        return super().__call__(g, me) + opp_history(g, me)
