"""NegoNet: scores the options of one negotiation decision.

    logit(o) = pair_mlp([s, phi(o)]) + sum_r G(s)[r, give_r] + sum_r H(s)[r, get_r]

G and H are the factored give/get heads (5 resources x counts 0..2) from
docs/TRADING.md section 6. Options are an enumerated legal set (bundles), so
masking is exact. A value head V(s) is the PPO baseline.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .options import OPT_F, STATE_F

NCOUNT = 3  # counts 0, 1, 2 (larger counts are clipped to 2)


class NegoNet(nn.Module):
    def __init__(self, hidden: int = 96):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(STATE_F, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh())
        self.give = nn.Linear(hidden, 5 * NCOUNT)
        self.get = nn.Linear(hidden, 5 * NCOUNT)
        self.pair = nn.Sequential(nn.Linear(hidden + OPT_F, hidden), nn.Tanh(), nn.Linear(hidden, 1))
        self.value = nn.Sequential(nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1))
        self.hidden = hidden

    def forward(self, s, o, give, get, mask):
        """s: (B, S); o: (B, K, F); give/get: (B, K, 5) long counts; mask: (B, K) bool.
        Returns (logits (B, K) with -inf on padding, value (B,))."""
        h = self.enc(s)
        B, K, _ = o.shape
        hk = h.unsqueeze(1).expand(B, K, self.hidden)
        lp = self.pair(torch.cat([hk, o], -1)).squeeze(-1)
        G = self.give(h).view(B, 1, 5, NCOUNT).expand(B, K, 5, NCOUNT)
        H = self.get(h).view(B, 1, 5, NCOUNT).expand(B, K, 5, NCOUNT)
        lg = torch.gather(G, 3, give.clamp(max=NCOUNT - 1).unsqueeze(-1)).squeeze(-1).sum(-1)
        lh = torch.gather(H, 3, get.clamp(max=NCOUNT - 1).unsqueeze(-1)).squeeze(-1).sum(-1)
        logits = (lp + lg + lh).masked_fill(~mask, float("-inf"))
        return logits, self.value(h).squeeze(-1)


def batch(samples, device="cpu"):
    """samples: list of dicts with s, o (K x F), give, get (K x 5). Pads to max K."""
    K = max(len(x["o"]) for x in samples)
    B = len(samples)
    s = torch.tensor([x["s"] for x in samples], dtype=torch.float32)
    o = torch.zeros(B, K, OPT_F)
    gv = torch.zeros(B, K, 5, dtype=torch.long)
    gt = torch.zeros(B, K, 5, dtype=torch.long)
    m = torch.zeros(B, K, dtype=torch.bool)
    for i, x in enumerate(samples):
        k = len(x["o"])
        o[i, :k] = torch.tensor(x["o"], dtype=torch.float32)
        gv[i, :k] = torch.tensor(x["give"], dtype=torch.long)
        gt[i, :k] = torch.tensor(x["get"], dtype=torch.long)
        m[i, :k] = True
    return s.to(device), o.to(device), gv.to(device), gt.to(device), m.to(device)


def save(net, path, meta):
    torch.save({"state_dict": net.state_dict(), "hidden": net.hidden, "meta": meta, "kind": "nego",
                "update": meta.get("update"), "step": meta.get("step")}, path)


def load(path):
    ck = torch.load(path, map_location="cpu", weights_only=True)
    net = NegoNet(ck.get("hidden", 96))
    net.load_state_dict(ck["state_dict"])
    net.eval()
    return net, ck.get("meta", {})
