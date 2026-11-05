"""League snapshot agent: an ArchAgent over a cached model (one load per checkpoint)."""
from __future__ import annotations

from catan.research.arch.agent import ArchAgent, load_checkpoint

_MODELS: dict = {}


class SnapshotAgent(ArchAgent):
    """ArchAgent over a cached model (sampling, like the learner it was)."""

    def __init__(self, checkpoint, deterministic=False, seed=None, **_):
        import random

        import torch
        if checkpoint not in _MODELS:
            _MODELS[checkpoint] = load_checkpoint(checkpoint)
        self.model, self.meta = _MODELS[checkpoint]
        from catan.research.arch.agent import encoder
        from catan.rl.env import EnvConfig
        self.encode = encoder(self.meta["obs_version"])
        self.device, self.deterministic, self.top_k = "cpu", deterministic, 5
        self.env_cfg = EnvConfig.from_dict((self.meta.get("config") or {}).get("env"))
        self.gen = torch.Generator().manual_seed(seed if seed is not None else random.randrange(2**31))
        self._static = self._layout = None
