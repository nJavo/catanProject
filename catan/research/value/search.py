"""``value_search``: exit_search whose leaf is a separately trained value net (D12).

    value_search:value=runs/value_v1/best.pt,sims=16,leaf=value,qnorm=false
    value_search:value=...,leaf=mix          # sigmoid(V) + heuristic margin / mix_scale

``checkpoint`` is still the prior (default ppo_baseline); ``value`` is a catan-arch checkpoint
whose value head outputs a win-probability logit (``value_logit`` in its metadata).
"""
from __future__ import annotations

import numpy as np
import torch

from catan.research.arch.agent import encoder, load_checkpoint
from catan.research.exit.search import ExitSearchAgent


class ValueSearchAgent(ExitSearchAgent):
    name = "value_search"

    def __init__(self, value: str, **kw):
        kw.setdefault("leaf", "value")
        super().__init__(**kw)
        self.vmodel, vmeta = load_checkpoint(value)
        if not vmeta.get("value_logit"):
            raise ValueError(f"{value}: not a D12 value checkpoint (no value_logit flag)")
        self.vencode = encoder(vmeta["obs_version"])

    @torch.no_grad()
    def _leaf_values(self, leaves, me):
        obs = torch.from_numpy(np.stack([self.vencode(c, me, self._static) for c in leaves]).astype(np.float32))
        _, v = self.vmodel(obs)
        return torch.sigmoid(v).tolist()
