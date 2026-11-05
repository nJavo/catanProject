"""Training env for PPO on the GNN with trades delegated to the nego layer (D13).

Learner seats get a per-seat ComboAgent-style trade delegate:
* TRADE_RESPONSE / TRADE_CONFIRM decisions of a learner seat are made by the
  delegate (never returned to the learner);
* OFFER_TRADE / COUNTER_OFFER are masked for the learner (``disable_domestic_trade``);
* when the learner picks END_TURN or a bank trade in MAIN, the delegate's offer
  head may play a domestic offer instead (as ComboAgent does at play time). The
  transition keeps the learner's chosen index, meaning "end/bank, or let the
  trade layer try first".
Delegates and opponent agents receive seat-filtered events (beliefs).
"""
from __future__ import annotations

from catan.engine.actions import A, Phase
from catan.research.league.agents import ObservingEnv
from catan.research.nego import options as NO
from catan.research.nego.agent import NegoAgent
from catan.rl import env as rl_env

NEGO_CHECKPOINT = "models/nego_best.pt"


class ComboEnv(ObservingEnv):
    def reset(self, seed, agents):
        self.delegates = {s: NegoAgent(seed=seed * 10 + s, checkpoint=NEGO_CHECKPOINT)
                          for s, a in enumerate(agents) if a is None}
        for s, d in self.delegates.items():
            d.reset(s, seed * 10 + s)
        return super().reset(seed, agents)

    def _dispatch(self, evs):
        super()._dispatch(evs)
        for s, d in self.delegates.items():
            d.observe(self.game, [e for e in evs if e.get("visible_to") is None or s in e["visible_to"]])

    def step(self, index):
        g = self.game
        seat = g.actor
        act = rl_env.decode_action(index, seat, g.n)
        d = self.delegates.get(seat)
        if (d is not None and g.phase == Phase.MAIN and act.type in (A.END_TURN, A.MARITIME_TRADE)
                and g.trades_this_turn < d.max_offers):
            legal = g.legal_actions()
            if any(a.type == A.OFFER_TRADE for a in legal):
                opts, teacher, est = NO.offer_options(d, g, seat, act)
                b, _ = d._nego_choose(g, seat, "offer", opts, teacher, est)
                if b.type == A.OFFER_TRADE and g.is_legal(b):
                    self.prev_phi[seat] = self.phi(seat)
                    self._dispatch(g.step(b))
                    return self._advance()
        return super().step(index)

    def _advance(self):
        g, c = self.game, self.cfg
        while not g.is_over:
            seat = g.actor
            legal = g.legal_actions()
            d = self.delegates.get(seat)
            agent = self.agents[seat]
            if d is not None and g.phase in (Phase.TRADE_RESPONSE, Phase.TRADE_CONFIRM):
                self._dispatch(g.step(d.act(g, legal)))
                continue
            if agent is not None:
                self._dispatch(g.step(agent.act(g, legal)))
                continue
            mask = rl_env.filter_mask(rl_env.action_mask(legal, seat, g.n), g, seat, c)
            if c.auto_forced and mask.sum() == 1:
                self._dispatch(g.step(rl_env.decode_action(int(mask.argmax()), seat, g.n)))
                continue
            break
        return rl_env.CatanEnv._advance(self)  # builds the learner Step (or the final one)


def install_env():
    from catan.rl import rollout
    rollout.CatanEnv = ComboEnv
