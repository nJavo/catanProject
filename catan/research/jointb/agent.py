"""``jointb_vt``: value_trade core + the D18 trade head (core + opponent-history features).

    jointb_vt                                   # bundle models/jointb_best.pt
    jointb_vt:checkpoint=runs/jointb_r1/bundle.pt,value=models/value_best.pt,sims=32

Same as D15's ``joint_vt`` except the head's state also carries
features.opp_history, so the bundle's head must have extra=EXTRA_B.
"""
from __future__ import annotations

from catan.research.combo.agent import DEFAULT_GNN
from catan.research.joint.agent import JointTradeAgent, _bundle_head
from catan.research.value.trade import DEFAULT_VALUE, ValueTradeAgent

from .features import EXTRA_B, JointBCore

DEFAULT_BUNDLE = "models/jointb_best.pt"
_HEADS: dict = {}
_CORES: dict = {}  # one per (bundle, value): each CoreFeatures adds a forward hook to the shared GNN


class JointBValueTradeAgent(ValueTradeAgent):
    name = "jointb_vt"

    def __init__(self, checkpoint: str = DEFAULT_BUNDLE, value: str = DEFAULT_VALUE, prior: str = DEFAULT_GNN,
                 seed=None, **kw):
        super().__init__(checkpoint=value, prior=prior, seed=seed, **kw)
        if checkpoint not in _HEADS:
            _HEADS[checkpoint] = _bundle_head(checkpoint)
        head = _HEADS[checkpoint]
        if head is None or head.extra != EXTRA_B:
            raise ValueError(f"{checkpoint}: needs a joint_head with extra={EXTRA_B}")
        if (checkpoint, value) not in _CORES:
            _CORES[checkpoint, value] = JointBCore(checkpoint, value)
        self.trade = JointTradeAgent(seed=seed, core=_CORES[checkpoint, value], joint_net=head)
