"""Agent that plays a trained PPO checkpoint. Registered as ``rl``:

    make_agent("rl", checkpoint="models/ppo_baseline.pt", deterministic=True)

Also registers ``rl_greedy``, a tiny scripted baseline used for RL sanity checks.
"""
from __future__ import annotations

import random

import torch

from catan.agents.base import Agent
from catan.agents.registry import register
from catan.engine import A, TOPOLOGY
from .model import load_checkpoint
from .spaces import encode_obs, action_mask, decode_action, action_label, static_features, PIPS
from .env import EnvConfig, filter_mask

DEFAULT_CHECKPOINT = "models/ppo_baseline.pt"


class PolicyAgent(Agent):
    name = "rl"

    def __init__(self, checkpoint: str = DEFAULT_CHECKPOINT, deterministic: bool = False,
                 device: str = "cpu", seed: int | None = None, top_k: int = 5, **_):
        self.model, self.meta = load_checkpoint(checkpoint, device)
        self.device = device
        self.deterministic = deterministic
        self.top_k = top_k
        self.env_cfg = EnvConfig.from_dict((self.meta.get("config") or {}).get("env"))  # trained action filters
        self.gen = torch.Generator().manual_seed(seed if seed is not None else random.randrange(2**31))
        self._static = None
        self._layout = None

    def reset(self, seat, seed=None):
        if seed is not None:
            self.gen.manual_seed(seed)

    @torch.no_grad()
    def act_with_info(self, game, legal, explain=True):
        me = game.actor
        if self._layout is not game.layout:
            self._static, self._layout = static_features(game.layout), game.layout
        obs = torch.from_numpy(encode_obs(game, me, self._static))[None].to(self.device)
        m = filter_mask(action_mask(legal, me, game.n), game, me, self.env_cfg)
        mask = torch.from_numpy(m)[None].to(self.device)
        logits, v = self.model(obs, mask)
        probs = torch.softmax(logits[0].float().cpu(), -1)
        if self.deterministic:
            i = int(probs.argmax())
        else:
            i = int(torch.multinomial(probs, 1, generator=self.gen))
        action = decode_action(i, me, game.n)
        top = torch.topk(probs, min(self.top_k, int(m.sum())))
        info = {
            "source": "model_output",
            "probs": {action_label(int(j)): round(float(p), 4) for p, j in zip(top.values, top.indices)},
            "value": round(float(v[0]), 4),  # critic estimate of this seat's return (win prob scale)
            "chosen_prob": round(float(probs[i]), 4),
        }
        if explain:  # read-only, after the choice; never uses self.gen (catan/rl/explain.py)
            from .explain import explain_policy
            try:
                info["explain"] = explain_policy(self, game, legal, obs, probs, i)
            except Exception as e:  # an explanation bug must not cost the agent its move
                info["explain_error"] = f"{type(e).__name__}: {e}"
        return action, info

    def act(self, game, legal):
        return self.act_with_info(game, legal, explain=False)[0]


class GreedyAgent(Agent):
    """Scripted sanity baseline: best-pip settlements, then city > settlement > dev > road,
    reject trades, rob a random opponent hex, otherwise end turn."""
    name = "rl_greedy"
    ORDER = (A.BUILD_CITY, A.BUILD_SETTLEMENT, A.BUY_DEV, A.BUILD_ROAD, A.PLAY_KNIGHT, A.ROLL,
             A.PLAY_ROAD_BUILDING, A.PLAY_YEAR_OF_PLENTY, A.PLAY_MONOPOLY)

    def __init__(self, seed: int | None = None, **_):
        self.rng = random.Random(seed)

    def reset(self, seat, seed=None):
        if seed is not None:
            self.rng = random.Random(seed)

    def act(self, game, legal):
        if len(legal) == 1:
            return legal[0]
        by = {}
        for a in legal:
            by.setdefault(a.type, []).append(a)
        if game.phase == "MOVE_ROBBER":
            me = game.actor
            good = [a for a in legal if any(game.vertex_owner[v] not in (-1, me)
                                            for v in TOPOLOGY.hex_vertices[a.value])
                    and not any(game.vertex_owner[v] == me for v in TOPOLOGY.hex_vertices[a.value])]
            return self.rng.choice(good or legal)
        if A.BUILD_SETTLEMENT in by:  # best production spot
            lay = game.layout
            return max(by[A.BUILD_SETTLEMENT], key=lambda a: sum(
                PIPS.get(lay.hex_number[h], 0) for h in TOPOLOGY.vertex_hexes[a.value]))
        for t in (A.REJECT_TRADE, A.CANCEL_TRADE):
            if t in by:
                return by[t][0]
        for t in self.ORDER:
            if t in by:
                return self.rng.choice(by[t])
        if A.MARITIME_TRADE in by and sum(game.players[game.actor].hand) > 7:
            return self.rng.choice(by[A.MARITIME_TRADE])  # avoid losing half on a 7
        if A.END_TURN in by:
            return by[A.END_TURN][0]
        return self.rng.choice(legal)



register("rl", lambda **kw: PolicyAgent(**kw))
register("rl_greedy", lambda seed=None, **kw: GreedyAgent(seed))
