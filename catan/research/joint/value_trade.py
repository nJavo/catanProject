"""``joint_vt``: D12's value_trade (value_search core) with the D15 joint trade head.

    joint_vt                                   # bundle models/joint_best.pt, value net models/value_best.pt
    joint_vt:checkpoint=models/joint_best.pt,value=models/value_best.pt,sims=32

``checkpoint`` is the joint bundle (league GNN weights + JointNet head): the GNN
supplies the head's core features (value + embedding, as in training) and the
head replaces value_trade's nego head. ``value``/``prior``/``sims`` go to the
value_search core exactly as in value_trade. Nothing in catan/research/value is
edited; this subclasses ValueTradeAgent and swaps its ``trade`` layer.
"""
from __future__ import annotations

from catan.agents.registry import register
from catan.research.combo.agent import DEFAULT_GNN
from catan.research.value.trade import DEFAULT_VALUE, ValueTradeAgent

from .agent import JointTradeAgent, _bundle_head
from .features import CoreFeatures

DEFAULT_BUNDLE = "models/joint_best.pt"
_HEADS: dict = {}


class JointValueTradeAgent(ValueTradeAgent):
    name = "joint_vt"

    def __init__(self, checkpoint: str = DEFAULT_BUNDLE, value: str = DEFAULT_VALUE, prior: str = DEFAULT_GNN,
                 seed=None, **kw):
        super().__init__(checkpoint=value, prior=prior, seed=seed, **kw)
        if checkpoint not in _HEADS:
            _HEADS[checkpoint] = _bundle_head(checkpoint)
        head = _HEADS[checkpoint]
        if head is None:
            raise ValueError(f"{checkpoint}: no joint_head in this checkpoint")
        self.trade = JointTradeAgent(seed=seed, core=CoreFeatures(checkpoint, value), joint_net=head)


register("joint_vt", lambda seed=None, **kw: JointValueTradeAgent(seed=seed, **kw))
