"""JointNet: NegoNet with the core policy's value/win-prob/embedding as extra state.

Same scoring form as catan.research.nego.model.NegoNet (pair MLP + factored
give/get heads over enumerated options); the state encoder input grows by
features.EXTRA_F.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from catan.research.nego.model import NegoNet
from catan.research.nego.options import STATE_F

from .features import EXTRA_F


class JointNet(NegoNet):
    def __init__(self, hidden: int = 96, extra: int = EXTRA_F):
        super().__init__(hidden)
        self.extra = extra
        self.enc = nn.Sequential(nn.Linear(STATE_F + extra, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh())


def load(path):
    ck = torch.load(path, map_location="cpu", weights_only=True)
    net = JointNet(ck.get("hidden", 96), ck.get("extra", EXTRA_F))
    net.load_state_dict(ck["state_dict"])
    net.eval()
    return net, ck.get("meta", {})


