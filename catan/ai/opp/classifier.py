"""Opponent-type classifier over ``features.seat_features`` (D21): a small MLP, softmax over CLASSES."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .features import CLASSES, FEATURE_NAMES, NF


class OppNet(nn.Module):
    def __init__(self, hidden: int = 64):
        super().__init__()
        self.hidden = hidden
        self.register_buffer("mean", torch.zeros(NF))
        self.register_buffer("std", torch.ones(NF))
        self.f = nn.Sequential(nn.Linear(NF, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(),
                               nn.Linear(hidden, len(CLASSES)))

    def forward(self, x):
        return self.f((x - self.mean) / self.std)

    @torch.no_grad()
    def proba(self, x: np.ndarray) -> np.ndarray:
        return torch.softmax(self(torch.from_numpy(np.asarray(x, np.float32))), -1).numpy()


def load(path) -> OppNet:
    d = torch.load(path, map_location="cpu", weights_only=True)
    if d.get("format") != "opp-classifier" or d["feature_names"] != list(FEATURE_NAMES):
        raise ValueError(f"{path}: not an opp classifier for the current feature set")
    m = OppNet(d["hidden"])
    m.load_state_dict(d["state_dict"])
    return m.eval()
