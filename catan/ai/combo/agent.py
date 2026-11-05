"""ComboAgent (D13): learned core play + learned/heuristic negotiation.

    make_agent("combo")                                    # league GNN + nego head, argmax GNN
    make_agent("combo", checkpoint="models/league_best.pt", nego_checkpoint="models/nego_best.pt")
    make_agent("combo", nego_checkpoint="none")            # trader heuristics instead of the nego head

* Building, robber, dev cards, setup, discards, bank trades: the arch_rl GNN,
  with OFFER_TRADE removed from its legal set (the RL policy spams offers).
* TRADE_RESPONSE / TRADE_CONFIRM: the nego layer (TraderMixin + NegoNet head).
* Proposals: when the GNN chooses END_TURN or a bank trade in MAIN and offers
  are left, the nego offer head may replace it with a domestic offer (the
  trader's own rule). The nego layer's belief tracker is fed through
  ``observe``.
"""
from __future__ import annotations

from catan.agents.base import Agent
from catan.engine.actions import A, Phase
from catan.ai.league.agents import SnapshotAgent
from catan.ai.nego import options as NO
from catan.ai.nego.agent import NegoAgent

DEFAULT_GNN = "models/league_best.pt"
DEFAULT_NEGO = "models/nego_best.pt"


class ComboAgent(Agent):
    name = "combo"

    def __init__(self, checkpoint: str = DEFAULT_GNN, nego_checkpoint: str = DEFAULT_NEGO,
                 deterministic: bool = True, seed: int | None = None, offers: bool = True, **_):
        self.net = SnapshotAgent(checkpoint, deterministic=deterministic, seed=seed)
        nck = None if nego_checkpoint in (None, "", "none") else nego_checkpoint
        self.trade = NegoAgent(seed=seed, checkpoint=nck)
        self.offers = offers

    def reset(self, seat, seed=None):
        self.net.reset(seat, seed)
        self.trade.reset(seat, seed)

    def observe(self, game, events):
        self.trade.observe(game, events)

    def act_with_info(self, game, legal, explain=True):
        ph = game.phase
        if ph in (Phase.TRADE_RESPONSE, Phase.TRADE_CONFIRM):
            a, info = self.trade.act_with_info(game, legal)
            return a, {**info, "source": "nego"}
        core = [a for a in legal if a.type != A.OFFER_TRADE]
        a, info = self.net.act_with_info(game, core, explain=explain)
        info = {**info, "source": "gnn"}
        tr = self.trade
        if (self.offers and ph == Phase.MAIN and a.type in (A.END_TURN, A.MARITIME_TRADE)
                and game.trades_this_turn < tr.max_offers and len(core) < len(legal)):
            me = game.actor
            opts, teacher, est = NO.offer_options(tr, game, me, a)
            b, ninfo = tr._nego_choose(game, me, "offer", opts, teacher, est)
            if b.type == A.OFFER_TRADE and game.is_legal(b):
                return b, {**ninfo, "source": "nego_offer", "replaced": a.to_json(),
                           **{k: info[k] for k in ("root", "sims", "leaf") if k in info}}  # search stats of the replaced pick
        return a, info

    def act(self, game, legal):
        return self.act_with_info(game, legal, explain=False)[0]
