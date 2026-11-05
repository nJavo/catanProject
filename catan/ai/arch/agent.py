"""Checkpoint I/O and the ``arch_rl`` registry agent for D5 models.

    make_agent("arch_rl", checkpoint="runs/arch_gnn_v2/checkpoints/ckpt_00030.pt")

Checkpoints carry ``arch`` (the model spec, including its obs version), so one
agent loads MLP/GNN and obs v1/v2. ``rl`` cannot load them (different format).
"""
from __future__ import annotations

import random

import torch

from catan.agents.base import Agent
from catan.rl import spaces as S
from catan.rl.env import EnvConfig, filter_mask
from . import obs as O
from .model import build

FORMAT = "catan-arch"
DEFAULT_CHECKPOINT = "models/arch_best.pt"  # D5 best model; lets "arch_rl" work in agent dropdowns


def encoder(obs_version: int):
    return O.encode_obs if obs_version == 2 else S.encode_obs


def save_checkpoint(path, model, cfg: dict, extra: dict | None = None, optimizer=None):
    spec = cfg["arch"]
    ck = {"format": FORMAT, "arch": dict(spec), "obs_version": spec["obs"], "action_version": S.ACTION_VERSION,
          "obs_dim": int(O.OBS_DIM if spec["obs"] == 2 else S.OBS_DIM), "act_dim": S.ACT_DIM,
          "config": cfg, "state_dict": model.state_dict()}
    if optimizer is not None:
        ck["optimizer"] = optimizer.state_dict()
    ck.update(extra or {})
    torch.save(ck, path)


def load_checkpoint(path, device="cpu"):
    ck = torch.load(path, map_location=device, weights_only=True)
    if ck.get("format") != FORMAT or ck.get("action_version") != S.ACTION_VERSION:
        raise ValueError(f"{path}: not a {FORMAT} checkpoint with action_version {S.ACTION_VERSION}")
    model = build(ck["arch"]).to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, ck


class ArchAgent(Agent):
    name = "arch_rl"

    def __init__(self, checkpoint: str = DEFAULT_CHECKPOINT, deterministic: bool = False, device: str = "cpu",
                 seed: int | None = None, top_k: int = 5, **_):
        self.model, self.meta = load_checkpoint(checkpoint, device)
        self.encode = encoder(self.meta["obs_version"])
        self.device, self.deterministic, self.top_k = device, deterministic, top_k
        self.env_cfg = EnvConfig.from_dict((self.meta.get("config") or {}).get("env"))
        self.gen = torch.Generator().manual_seed(seed if seed is not None else random.randrange(2**31))
        self._static = self._layout = None

    def reset(self, seat, seed=None):
        if seed is not None:
            self.gen.manual_seed(seed)

    def mask(self, game, legal):
        me = game.actor
        return filter_mask(S.action_mask(legal, me, game.n), game, me, self.env_cfg)

    @torch.no_grad()
    def act_with_info(self, game, legal, explain=True):
        me = game.actor
        if self._layout is not game.layout:
            self._static, self._layout = S.static_features(game.layout), game.layout
        obs = torch.from_numpy(self.encode(game, me, self._static))[None].to(self.device)
        m = self.mask(game, legal)
        logits, v = self.model(obs, torch.from_numpy(m)[None].to(self.device))
        probs = torch.softmax(logits[0].float().cpu(), -1)
        i = int(probs.argmax()) if self.deterministic else int(torch.multinomial(probs, 1, generator=self.gen))
        top = torch.topk(probs, min(self.top_k, int(m.sum())))
        info = {"source": "model_output",
                "probs": {S.action_label(int(j)): round(float(p), 4) for p, j in zip(top.values, top.indices)},
                "value": round(float(v[0]), 4), "chosen_prob": round(float(probs[i]), 4)}
        if explain:  # read-only, after the choice; never uses self.gen (catan/rl/explain.py)
            from catan.rl.explain import explain_policy
            try:
                info["explain"] = explain_policy(self, game, legal, obs, probs, i)
            except Exception as e:  # an explanation bug must not cost the agent its move
                info["explain_error"] = f"{type(e).__name__}: {e}"
        return S.decode_action(i, me, game.n), info

    def act(self, game, legal):
        return self.act_with_info(game, legal, explain=False)[0]
