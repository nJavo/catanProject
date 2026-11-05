"""Trade head that sees the core policy (D15): the nego state plus CoreFeatures
(GNN value, D12 win probability, 256-d GNN embedding)."""
from __future__ import annotations

from catan.ai.nego.agent import NegoAgent

from .features import CoreFeatures


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
