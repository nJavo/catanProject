"""Actor-critic network and checkpoint I/O."""
from __future__ import annotations

import torch
from torch import nn

from .spaces import OBS_DIM, ACT_DIM, OBS_VERSION, ACTION_VERSION

MASK_VALUE = -1e9  # large negative (not -inf) so masked softmax/entropy never NaN


class ResBlock(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.norm = nn.LayerNorm(h)
        self.fc = nn.Linear(h, h)

    def forward(self, x):
        return x + torch.relu(self.fc(self.norm(x)))


class PolicyNet(nn.Module):
    """Shared-trunk residual MLP. Actor and critic both see only the acting
    seat's observation (no privileged critic)."""

    def __init__(self, obs_dim: int = OBS_DIM, act_dim: int = ACT_DIM, hidden: int = 512, blocks: int = 2):
        super().__init__()
        self.inp = nn.Linear(obs_dim, hidden)
        self.blocks = nn.Sequential(*[ResBlock(hidden) for _ in range(blocks)])
        self.norm = nn.LayerNorm(hidden)
        self.pi = nn.Linear(hidden, act_dim)
        self.v = nn.Linear(hidden, 1)
        nn.init.orthogonal_(self.pi.weight, 0.01)
        nn.init.zeros_(self.pi.bias)

    def forward(self, obs, mask=None):
        x = self.norm(self.blocks(torch.relu(self.inp(obs))))
        logits = self.pi(x)
        if mask is not None:
            logits = logits.masked_fill(~mask, MASK_VALUE)
        return logits, self.v(x).squeeze(-1)


def model_kwargs(cfg: dict) -> dict:
    m = cfg.get("model", {}) or {}
    return {"hidden": int(m.get("hidden", 512)), "blocks": int(m.get("blocks", 2))}


def save_checkpoint(path, model: PolicyNet, cfg: dict, extra: dict | None = None, optimizer=None):
    ck = {
        "format": "catan-rl-ppo",
        "obs_version": OBS_VERSION, "action_version": ACTION_VERSION,
        "obs_dim": OBS_DIM, "act_dim": ACT_DIM,
        "model_kwargs": model_kwargs(cfg),
        "config": cfg,
        "state_dict": model.state_dict(),
    }
    if optimizer is not None:
        ck["optimizer"] = optimizer.state_dict()
    ck.update(extra or {})
    torch.save(ck, path)


def load_checkpoint(path, device="cpu"):
    """Returns (model, checkpoint dict). Only tensors/primitives are unpickled."""
    ck = torch.load(path, map_location=device, weights_only=True)
    if ck.get("obs_version") != OBS_VERSION or ck.get("action_version") != ACTION_VERSION:
        raise ValueError(f"checkpoint schema {ck.get('obs_version')}/{ck.get('action_version')} "
                         f"!= current {OBS_VERSION}/{ACTION_VERSION}")
    model = PolicyNet(ck["obs_dim"], ck["act_dim"], **ck["model_kwargs"]).to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, ck
