"""JointNet: NegoNet with the core policy's value/win-prob/embedding as extra state.

Same scoring form as catan.research.nego.model.NegoNet (pair MLP + factored
give/get heads over enumerated options); the state encoder input grows by
features.EXTRA_F. ``from_nego`` warm-starts every weight from a NegoNet and
zero-initialises the new input columns, so the initial policy equals the nego head.
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

    @staticmethod
    def from_nego(nego: NegoNet) -> "JointNet":
        j = JointNet(nego.hidden)
        sd = nego.state_dict()
        w = sd["enc.0.weight"]
        new = torch.zeros(w.shape[0], STATE_F + j.extra)
        new[:, :STATE_F] = w
        sd["enc.0.weight"] = new
        j.load_state_dict(sd)
        return j


def save(net, path, meta):
    torch.save({"state_dict": net.state_dict(), "hidden": net.hidden, "extra": net.extra, "kind": "joint",
                "meta": meta}, path)


def load(path):
    ck = torch.load(path, map_location="cpu", weights_only=True)
    net = JointNet(ck.get("hidden", 96), ck.get("extra", EXTRA_F))
    net.load_state_dict(ck["state_dict"])
    net.eval()
    return net, ck.get("meta", {})


