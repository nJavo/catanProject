"""JointAgent (D15): combo with a trade head that sees the core policy.

    make_agent("joint", trade_checkpoint="runs/joint_r2/joint_ep5.pt")  # league GNN + a JointNet head file
    make_agent("joint", checkpoint="models/joint_best.pt")          # bundle: GNN weights + joint head
    make_agent("joint")                                             # no head: nego choices (= combo_zs)

Core play is the league GNN exactly as in ComboAgent. Trade decisions use
``JointNet`` whose state is the nego state plus CoreFeatures (GNN value, D12
win probability, 256-d GNN embedding).
"""
from __future__ import annotations

from catan.agents.registry import register
from catan.research.combo.agent import DEFAULT_GNN, ComboAgent
from catan.research.nego.agent import NegoAgent

from .features import CoreFeatures

_NETS: dict = {}
_NO_HEAD: set = set()


def _net(path):
    if path not in _NETS:
        from .model import load
        _NETS[path] = load(path)[0]
    return _NETS[path]


def _bundle_head(path):
    """A bundle checkpoint (models/joint_best.pt) holds the GNN weights plus ``joint_head``."""
    import torch
    from .model import JointNet
    ck = torch.load(path, map_location="cpu", weights_only=True)
    h = ck.get("joint_head")
    if h is None:
        return None
    net = JointNet(h.get("hidden", 96), h.get("extra"))
    net.load_state_dict(h["state_dict"])
    net.eval()
    return net


class JointTradeAgent(NegoAgent):
    """NegoAgent whose head state is extended with core-policy features."""

    def __init__(self, seed=None, core: CoreFeatures | None = None, joint_net=None, **kw):
        super().__init__(seed=seed, net=joint_net, **kw)
        self.core = core

    def _state_vec(self, g, me, est, decision):
        s = super()._state_vec(g, me, est, decision)
        return s + self.core(g, me) if self.net is not None or self.record else s


class JointAgent(ComboAgent):
    name = "joint"

    def __init__(self, checkpoint: str = DEFAULT_GNN, trade_checkpoint: str | None = None,
                 value: str = "models/value_best.pt", nego_checkpoint: str = "models/nego_best.pt",
                 deterministic: bool = True, seed=None, **kw):
        super().__init__(checkpoint=checkpoint, nego_checkpoint=nego_checkpoint,
                         deterministic=deterministic, seed=seed, **kw)
        net = None
        if trade_checkpoint not in (None, "", "none"):
            net = _net(trade_checkpoint)
        elif checkpoint not in _NETS and checkpoint not in _NO_HEAD:
            h = _bundle_head(checkpoint)
            if h is None:
                _NO_HEAD.add(checkpoint)
            else:
                _NETS[checkpoint] = h
        if checkpoint in _NETS and net is None:
            net = _NETS[checkpoint]
        if net is not None:
            core = CoreFeatures(checkpoint, value)
            self.trade = JointTradeAgent(seed=seed, core=core, joint_net=net)


register("joint", lambda seed=None, **kw: JointAgent(seed=seed, **kw))
