"""Core-policy features for the trade head (D15) and the D12 value scorer.

``CoreFeatures(gnn, value)`` maps (game, seat) to
  [v_gnn, p_win (D12 value net), g (256-d pooled GNN embedding)]
All three come from the actor's information-set observation (arch obs v2), so
they are info-safe. The embedding is read with a forward hook on the GNN's
``norm_g`` (no edits to catan/research/arch). Results are cached per
(action_log length, phase, seat).
"""
from __future__ import annotations

import numpy as np
import torch

from catan.research.arch.agent import encoder, load_checkpoint
from catan.rl import spaces as S

EMB = 256
EXTRA_F = 2 + EMB

_CACHE: dict = {}


def _load(path):
    if path not in _CACHE:
        _CACHE[path] = load_checkpoint(path)
    return _CACHE[path]


class CoreFeatures:
    def __init__(self, gnn: str = "models/league_best.pt", value: str = "models/value_best.pt"):
        self.gnn, gmeta = _load(gnn)
        self.vnet, vmeta = _load(value)
        if not vmeta.get("value_logit"):
            raise ValueError(f"{value}: not a D12 value checkpoint")
        self.genc = encoder(gmeta["obs_version"])
        self.venc = encoder(vmeta["obs_version"])
        self._emb = None
        self.gnn.norm_g.register_forward_hook(lambda m, i, o: setattr(self, "_emb", o.detach()))
        self._static = self._layout = None
        self._key = None
        self._val = None

    def _static_for(self, g):
        if self._layout is not g.layout:
            self._static, self._layout = S.static_features(g.layout), g.layout
        return self._static

    @torch.no_grad()
    def __call__(self, g, me) -> list[float]:
        key = (id(g.layout), len(g.action_log), g.phase, me, g.current)
        if key == self._key:
            return self._val
        st = self._static_for(g)
        obs = torch.from_numpy(self.genc(g, me, st).astype(np.float32))[None]
        _, v = self.gnn(obs)
        emb = self._emb[0].tolist()
        p = self.win_prob(g, me)
        self._key, self._val = key, [float(v[0]), p] + emb
        return self._val

    @torch.no_grad()
    def win_prob(self, g, me) -> float:
        st = self._static_for(g)
        obs = torch.from_numpy(self.venc(g, me, st).astype(np.float32))[None]
        _, v = self.vnet(obs)
        return float(torch.sigmoid(v[0]))

    @torch.no_grad()
    def win_probs(self, games, me) -> list[float]:
        if not games:
            return []
        st = self._static_for(games[0])
        obs = torch.from_numpy(np.stack([self.venc(c, me, st) for c in games]).astype(np.float32))
        _, v = self.vnet(obs)
        return torch.sigmoid(v).tolist()
