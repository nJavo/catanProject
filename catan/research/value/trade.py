"""``value_trade``: value_search for core play + the combo/nego trade layer for trades.

    value_trade                                   # value net models/value_best.pt, league prior, 32 sims
    value_trade:checkpoint=runs/value_v2/best.pt,prior=models/league_best.pt,sims=64
    value_trade:nego_checkpoint=none              # trader heuristics instead of the nego head

``checkpoint`` is the D12 value net (so ``scripts.promote --agent value_trade --checkpoint`` tracks
the trained file); ``prior`` is the search's policy prior. Same split as
``catan.research.combo.agent.ComboAgent``, with its GNN core replaced by search:
TRADE_RESPONSE / TRADE_CONFIRM go to the nego layer; search never sees OFFER_TRADE; when search
picks END_TURN or a bank trade in MAIN, the nego offer head may replace it with a domestic offer.
"""
from __future__ import annotations

from catan.research.combo.agent import DEFAULT_GNN, DEFAULT_NEGO, ComboAgent
from catan.research.nego.agent import NegoAgent

from .search import ValueSearchAgent

DEFAULT_VALUE = "models/value_best.pt"


class ValueTradeAgent(ComboAgent):
    name = "value_trade"

    def __init__(self, checkpoint: str = DEFAULT_VALUE, prior: str = DEFAULT_GNN,
                 nego_checkpoint: str = DEFAULT_NEGO, sims: int = 32, qnorm: bool = False,
                 seed: int | None = None, offers: bool = True, deterministic: bool = True, **kw):
        # no super().__init__: it would load a GNN policy agent we never use.
        # deterministic: accepted for the gate's --deterministic; the search is deterministic (gumbel off).
        self.net = ValueSearchAgent(value=checkpoint, checkpoint=prior, sims=sims, qnorm=qnorm, seed=seed, **kw)
        nck = None if nego_checkpoint in (None, "", "none") else nego_checkpoint
        self.trade = NegoAgent(seed=seed, checkpoint=nck)
        self.offers = offers

    def act_with_info(self, game, legal, explain=True):
        a, info = super().act_with_info(game, legal, explain=explain)
        if info.get("source") == "gnn":  # ComboAgent's label for its core; here the core is search
            info["source"] = "value_search"
        return a, info
