"""League members as registry agents, plus an event-observing env for training.

Members are registered under colon-free names (``lg_*``) because
catan.rl.ppo.episode_metrics groups episodes by ``mode.split(":")[0]``.
Snapshot agents share one loaded model per checkpoint (opponents are created
per game).
"""
from __future__ import annotations

from catan.agents.registry import make_agent, register
from catan.research.arch.agent import ArchAgent, load_checkpoint
from catan.rl import env as rl_env

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


def register_member(mid: str, info: dict):
    """Make ``mid`` constructible through catan.agents.registry.make_agent."""
    if info["kind"] == "snapshot":
        path = info["path"]
        register(mid, lambda seed=None, **kw: SnapshotAgent(path, seed=seed))
    elif mid != info["spec"]:
        name, _, rest = info["spec"].partition(":")
        kw0 = dict(x.split("=", 1) for x in rest.split(",")) if rest else {}
        register(mid, lambda seed=None, **kw: make_agent(name, seed=seed, **kw0))


class ObservingEnv(rl_env.CatanEnv):
    """CatanEnv that feeds seat-filtered step events to agents with an
    ``observe(game, events)`` hook (trader/nego belief trackers), like the
    tournament runner and the server. Proposed upstream: the same dispatch
    after each ``g.step`` in catan.rl.env.CatanEnv. ``step``/``_advance`` are
    copies of the originals plus the dispatch (no Game patching: clones made by
    search agents must not share it)."""

    def _dispatch(self, evs):
        for s, a in enumerate(self.agents):
            f = getattr(a, "observe", None) if a is not None else None
            if callable(f):
                f(self.game, [e for e in evs if e.get("visible_to") is None or s in e["visible_to"]])

    def step(self, index):
        g = self.game
        seat = g.actor
        self.prev_phi[seat] = self.phi(seat)
        self._dispatch(g.step(rl_env.decode_action(index, seat, g.n)))
        return self._advance()

    def _advance(self):
        g, c = self.game, self.cfg
        while not g.is_over:
            seat = g.actor
            legal = g.legal_actions()
            agent = self.agents[seat]
            if agent is not None:
                self._dispatch(g.step(agent.act(g, legal)))
                continue
            mask = rl_env.filter_mask(rl_env.action_mask(legal, seat, g.n), g, seat, c)
            if c.auto_forced and mask.sum() == 1:
                self._dispatch(g.step(rl_env.decode_action(int(mask.argmax()), seat, g.n)))
                continue
            break
        return super()._advance()  # no external decision left to auto-play: original builds the Step


def install_env():
    from catan.rl import rollout
    rollout.CatanEnv = ObservingEnv
