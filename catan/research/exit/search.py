"""ExitSearchAgent: policy-guided determinized search (Gumbel root, PIMC leaves).

Root (Danihelka et al. 2022, "Policy improvement by planning with Gumbel"):
sample ``m`` candidate actions without replacement with Gumbel-top-k over the
policy logits, then spend ``sims`` simulations with Sequential Halving. Each
simulation resamples the hidden state from the actor's information set
(``catan.bots.common.determinize``: perfect-information Monte Carlo), applies
the candidate, lets the rollout policy finish the actor's own turn and the
basic bot play everyone else until the actor's next real decision, then scores
the leaf. Determinizations are shared across candidates within a phase (common
random numbers, as in ``catan.bots.search``).

Leaf value ``q`` (``leaf=``):
- ``value``: the network's value head V(leaf) from the actor's seat (win prob);
- ``heur``: ``evaluate(me) - opp_weight * max_opp evaluate`` (intermediate's model);
- ``mix``:  V + heur / mix_scale.
q is min-max normalized over the visited candidates, so scales never matter
(``qnorm=false`` keeps raw q, for value leaves that already are win probabilities).
Checkpoints: ``rl`` (MLP, obs v1) or D5 ``catan-arch`` (GNN, obs v2), detected by format.

The chosen action is argmax(g + logits + sigma(q)) over the survivors, and
the training target is the improved policy softmax(logits + sigma(completed q))
over all legal actions (unvisited actions get the prior-weighted mean of the
visited q). ``sims=0`` is the raw policy (argmax / sample of the logits).
"""
from __future__ import annotations

import math
import random

import numpy as np
import torch

from catan.agents.base import Agent
from catan.bots import common as C
from catan.bots.basic import BasicHeuristicAgent
from catan.bots.intermediate import IntermediateAgent, evaluate
from catan.engine.actions import A
from catan.engine.game import IllegalAction
from catan.rl.env import CatanEnv, EnvConfig, filter_mask
from catan.rl.model import load_checkpoint as load_rl_checkpoint
from catan.rl.spaces import action_mask, decode_action, encode_obs, static_features

MAX_ROLLOUT_STEPS = 400


class ExitSearchAgent(Agent):
    name = "exit_search"

    def __init__(self, checkpoint: str = "models/ppo_baseline.pt", sims: int = 16, m: int = 8,
                 leaf: str = "heur", rollout: str = "intermediate", gumbel: bool = False,
                 c_visit: float = 50.0, c_scale: float = 1.0, opp_weight: float = 0.3,
                 mix_scale: float = 10.0, qnorm: bool = True, seed: int | None = None, record: bool = False, **_):
        if leaf not in ("value", "heur", "mix"):
            raise ValueError(f"leaf must be value|heur|mix, got {leaf!r}")
        if rollout not in ("intermediate", "policy"):
            raise ValueError(f"rollout must be intermediate|policy, got {rollout!r}")
        if torch.load(checkpoint, map_location="cpu", weights_only=True).get("format") == "catan-arch":
            from catan.research.arch.agent import encoder, load_checkpoint as load_arch_checkpoint
            self.model, self.meta = load_arch_checkpoint(checkpoint)  # D5 GNN/MLP, obs v1 or v2
            self.encode = encoder(self.meta["obs_version"])
        else:
            self.model, self.meta = load_rl_checkpoint(checkpoint)
            self.encode = encode_obs
        self.env_cfg = EnvConfig.from_dict((self.meta.get("config") or {}).get("env"))
        # PPO with potential shaping learns V = E[return] - Phi(s); add Phi back to get a win-prob
        # scale. ExIt and BC checkpoints regress V on the outcome z directly, so they skip this.
        pure_z = "exit_config" in self.meta or "bc_teacher" in self.meta
        self.value_phi = not pure_z and bool(self.env_cfg.shaping_vp or self.env_cfg.shaping_prod)
        self._phi_env = CatanEnv(self.env_cfg)
        self.sims, self.m, self.leaf, self.rollout = int(sims), int(m), leaf, rollout
        self.gumbel, self.c_visit, self.c_scale = bool(gumbel), c_visit, c_scale
        self.opp_weight, self.mix_scale = opp_weight, mix_scale
        self.qnorm = bool(qnorm)  # False: use raw q (only sensible for leaf=value, already a win prob)
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)
        self._me_policy = IntermediateAgent()
        self._opp_policy = BasicHeuristicAgent()
        self._static = self._layout = self._bi = None
        self.record = record
        self.records: list = []  # (seat, obs f16, mask, target f32) when record=True

    def reset(self, seat, seed=None):
        if seed is not None:
            self.rng = random.Random(seed)
            self.np_rng = np.random.default_rng(seed)
            self._me_policy = IntermediateAgent(seed=seed)
            self._opp_policy = BasicHeuristicAgent(seed=seed + 1)
        self.records = []

    # ------------------------------------------------------------ network
    def _board(self, g):
        if self._layout is not g.layout:
            self._static, self._layout, self._bi = static_features(g.layout), g.layout, C.board_info(g.layout)

    def _mask(self, g, me, legal):
        return filter_mask(action_mask(legal, me, g.n), g, me, self.env_cfg)

    @torch.no_grad()
    def _net(self, obs: np.ndarray, mask: np.ndarray | None = None):
        o = torch.from_numpy(np.asarray(obs, np.float32))
        m = None if mask is None else torch.from_numpy(mask)
        logits, v = self.model(o, m)
        return logits.numpy(), v.numpy()

    # ------------------------------------------------------------ act
    def act_with_info(self, game, legal, **_):  # **_: explain= from wrappers (catan.research.value.trade)
        me = game.actor
        self._board(game)
        mask = self._mask(game, me, legal)
        idx = np.flatnonzero(mask)
        if len(idx) == 1:
            return decode_action(int(idx[0]), me, game.n), {"source": "forced"}
        obs = self.encode(game, me, self._static)
        logits, v = self._net(obs[None], mask[None])
        logits = logits[0]
        if self.sims <= 0:
            i = int(idx[np.argmax(logits[idx])])
            return decode_action(i, me, game.n), {"source": "model_output", "value": float(v[0])}
        i, target, info = self._gumbel_search(game, me, idx, logits)
        if self.record:
            self.records.append((me, obs.astype(np.float16), mask, target))
        info["value"] = round(float(v[0]), 4)
        return decode_action(i, me, game.n), info

    def act(self, game, legal):
        return self.act_with_info(game, legal)[0]

    # ------------------------------------------------------------ search
    def _sigma(self, q, max_n):
        return (self.c_visit + max_n) * self.c_scale * q

    def _gumbel_search(self, g, me, idx, logits):
        lg = logits[idx].astype(np.float64)
        gum = self.np_rng.gumbel(size=len(idx)) if self.gumbel else np.zeros(len(idx))
        m = min(self.m, len(idx), max(2, self.sims))
        order = np.argsort(-(gum + lg), kind="stable")[:m]
        alive = list(order)                     # positions into idx
        qsum = np.zeros(len(idx))
        n = np.zeros(len(idx))
        phases = max(1, math.ceil(math.log2(m)))
        used = 0
        while True:
            per = max(1, self.sims // (phases * len(alive)))
            if len(alive) == 1 or used + per * len(alive) > self.sims and used > 0:
                break
            for _ in range(per):
                seed = self.rng.getrandbits(64)
                for vals, j in zip(self._evaluate([int(idx[j]) for j in alive], g, me, seed), alive):
                    qsum[j] += vals
                    n[j] += 1
            used += per * len(alive)
            q = self._norm_q(qsum, n)
            alive.sort(key=lambda j: -(gum[j] + lg[j] + self._sigma(q[j], n.max())))
            alive = alive[:max(1, math.ceil(len(alive) / 2))]
        q = self._norm_q(qsum, n)
        best = alive[0]
        # improved policy: completed q for unvisited = prior-weighted mean of visited q
        prior = np.exp(lg - lg.max())
        prior /= prior.sum()
        vis = n > 0
        vmix = float((prior[vis] * q[vis]).sum() / prior[vis].sum())
        cq = np.where(vis, q, vmix)
        z = lg + self._sigma(cq, n.max())
        p = np.exp(z - z.max())
        p /= p.sum()
        target = np.zeros(len(logits), np.float32)
        target[idx] = p
        info = {"source": "exit_search", "sims": int(n.sum()), "candidates": int(vis.sum()), "leaf": self.leaf,
                # per root candidate: raw mean leaf value q (before normalisation), visits n, prior p (analysis mode)
                "root": [{"action": decode_action(int(idx[j]), me, g.n).to_json(), "n": int(n[j]),
                          "q": round(float(qsum[j] / n[j]), 4) if n[j] else None, "prior": round(float(prior[j]), 4)}
                         for j in order]}
        return int(idx[best]), target, info

    def _norm_q(self, qsum, n):
        vis = n > 0
        q = np.zeros_like(qsum)
        q[vis] = qsum[vis] / n[vis]
        if not self.qnorm:
            return np.clip(q, 0.0, 1.0)
        if vis.sum() >= 2:
            lo, hi = q[vis].min(), q[vis].max()
            q[vis] = (q[vis] - lo) / (hi - lo) if hi > lo else 0.5
        else:
            q[vis] = 0.5
        return q

    def _evaluate(self, actions, g, me, seed):
        """Leaf values for each candidate on one shared determinization."""
        leaves = []
        for a in actions:
            rng = random.Random(seed)
            c = C.determinize(g, me, rng)
            try:
                c.step(decode_action(a, me, g.n))
            except IllegalAction:
                pass  # shortcut: determinize gave up on trade consistency; score the unstepped state
            leaves.append(self._rollout(c, me, g.turn))
        out = [0.0] * len(actions)
        need = [k for k, c in enumerate(leaves) if not c.is_over]
        for k, c in enumerate(leaves):
            if c.is_over:
                out[k] = 1.0 if c.winner == me else 0.0
                if self.leaf == "heur":
                    out[k] = evaluate(c, me, self._bi)
                elif self.leaf == "mix":
                    out[k] += evaluate(c, me, self._bi) / self.mix_scale
        if need and self.leaf in ("value", "mix"):
            for k, x in zip(need, self._leaf_values([leaves[k] for k in need], me)):
                out[k] = x
        if self.leaf in ("heur", "mix"):
            for k in need:
                c = leaves[k]
                h = evaluate(c, me, self._bi) - self.opp_weight * max(
                    evaluate(c, q, self._bi) for q in range(c.n) if q != me)
                out[k] = h if self.leaf == "heur" else out[k] + h / self.mix_scale
        return out

    def _leaf_values(self, leaves, me) -> list[float]:
        """Network value of non-terminal leaves from me's seat (hook: catan.research.value)."""
        _, v = self._net(np.stack([self.encode(c, me, self._static) for c in leaves]))
        return [float(x) + (self._phi(c, me) if self.value_phi else 0.0) for c, x in zip(leaves, v)]

    def _phi(self, c, me):
        e = self._phi_env
        if e.game is None or e.game.layout is not c.layout:
            e._pip = CatanEnv._vertex_pips(c.layout)
        e.game = c
        return e.phi(me)

    def _rollout(self, c, me, turn0):
        """Play until me's next real decision outside its current turn, or until
        me ends its current turn (leaf = state where me is about to END_TURN)."""
        for _ in range(MAX_ROLLOUT_STEPS):
            if c.is_over:
                return c
            p = c.actor
            legal = c.legal_actions()
            if p == me and len(legal) > 1:
                if not (c.current == me and c.turn == turn0):
                    return c
                a = self._my_rollout_action(c, me, legal)
                if a.type == A.END_TURN:
                    return c
            elif p == me:
                a = legal[0]
            else:
                a = self._opp_policy.act(c, legal)
                if a.type == A.OFFER_TRADE:  # keep opponent rollouts cheap, as catan.bots.search
                    a = next((x for x in legal if x.type == A.END_TURN), a)
            try:
                c.step(a)
            except IllegalAction:  # inconsistent determinized trade: stop and score here
                return c
        return c

    def _my_rollout_action(self, c, me, legal):
        if self.rollout == "intermediate":
            return self._me_policy.act(c, legal)
        mask = self._mask(c, me, legal)
        logits, _ = self._net(self.encode(c, me, self._static)[None], mask[None])
        return decode_action(int(np.argmax(logits[0])), me, c.n)
