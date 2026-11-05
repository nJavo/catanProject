"""Multi-agent Catan environment for RL.

One ``CatanEnv`` wraps one engine game. Each seat is either *external* (its
decisions are returned to the caller: learning seats and frozen checkpoints)
or controlled by an in-process ``Agent`` (random, heuristic bots...), which the
env plays automatically. Every engine decision of an external seat is one env
step; turn sequencing follows ``game.actor`` (discards, trade responses etc.
happen out of turn).

Rewards are per seat and credited to that seat's *previous* decision:
- when a seat is asked to act again, ``reward`` is the shaping reward
  ``gamma*Phi(s') - Phi(s)`` accrued since its previous decision;
- when the game ends, ``final[seat]`` holds terminal reward plus final shaping.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, fields

import numpy as np

from catan.engine import Game, GameConfig, Phase, TOPOLOGY
from catan.engine.board import DESERT
from .spaces import encode_obs, action_mask, decode_action, static_features, PIPS, PROPOSE_MASK, ROBBER_BASE


@dataclass
class EnvConfig:
    num_players: int = 4
    max_turns: int = 500             # engine truncation (turn counter counts every END_TURN)
    win_reward: float = 1.0
    loss_reward: float = 0.0
    # "loss": truncation is terminal with truncation_reward.
    # "bootstrap": treat as a time limit; the learner bootstraps from V(s_T).
    truncation: str = "loss"
    truncation_reward: float = 0.0
    gamma: float = 0.999             # must match PPO gamma for policy-invariant shaping
    shaping_vp: float = 0.0          # Phi += shaping_vp * VP / vp_to_win
    shaping_prod: float = 0.0        # Phi += shaping_prod * expected cards per roll / 3
    terminal_vp_reward: float = 0.0  # terminal += coef * final VP / vp_to_win (all seats, win or lose)
    prod_robber_aware: bool = False  # subtract own production on the robber hex from Phi_prod
    auto_forced: bool = True         # play single-legal-action decisions without asking
    # Remove OFFER_TRADE / COUNTER_OFFER from external seats' masks (accepting/rejecting
    # opponents' offers stays). PolicyAgent applies the same filter from the checkpoint config.
    disable_domestic_trade: bool = False
    # Mask MOVE_ROBBER targets touching the seat's own buildings unless all legal targets do.
    mask_self_robber: bool = False

    @staticmethod
    def from_dict(d: dict | None) -> "EnvConfig":
        names = {f.name for f in fields(EnvConfig)}
        d = dict(d or {})
        unknown = set(d) - names
        if unknown:
            raise ValueError(f"unknown env config keys {sorted(unknown)}")
        return EnvConfig(**d)

    def to_dict(self):
        return asdict(self)


def filter_mask(mask: np.ndarray, g, seat: int, c: EnvConfig) -> np.ndarray:
    """Optional action filters (shared by env and PolicyAgent). Never empties the mask."""
    if c.disable_domestic_trade and (mask & ~PROPOSE_MASK).any():
        mask &= ~PROPOSE_MASK
    if c.mask_self_robber and g.phase == Phase.MOVE_ROBBER:
        own = np.array([any(g.vertex_owner[v] == seat for v in TOPOLOGY.hex_vertices[h])
                        for h in range(TOPOLOGY.num_hexes)])
        keep = mask.copy()
        keep[ROBBER_BASE:ROBBER_BASE + TOPOLOGY.num_hexes] &= ~own
        if keep.any():
            mask = keep
    return mask


@dataclass
class Step:
    seat: int = -1                   # next external seat to act (-1 when done)
    obs: np.ndarray | None = None
    mask: np.ndarray | None = None
    reward: float = 0.0              # shaping reward for ``seat``'s previous decision
    done: bool = False
    final: dict | None = None        # seat -> reward for its last decision (when done)
    truncated: bool = False
    bootstrap: bool = False          # truncated in "bootstrap" mode: caller bootstraps V


class CatanEnv:
    def __init__(self, config: EnvConfig | None = None):
        self.cfg = config or EnvConfig()
        self.game: Game | None = None

    # ------------------------------------------------------------ api
    def reset(self, seed: int, agents: list) -> Step:
        """Start a game. ``agents[seat]`` is an Agent, or None for external seats."""
        c = self.cfg
        if len(agents) != c.num_players:
            raise ValueError("need one agent slot per seat")
        self.game = Game(config=GameConfig(num_players=c.num_players, max_turns=c.max_turns), seed=seed)
        self.agents = agents
        self.static = static_features(self.game.layout)
        self._pip = self._vertex_pips(self.game.layout)
        self.prev_phi: dict[int, float] = {}   # external seat -> Phi at its last decision
        for s, a in enumerate(agents):
            if a is not None:
                a.reset(s, seed * 10 + s)
        return self._advance()

    def step(self, index: int) -> Step:
        g = self.game
        seat = g.actor
        self.prev_phi[seat] = self.phi(seat)
        g.step(decode_action(index, seat, g.n))
        return self._advance()

    def observe(self, seat: int) -> np.ndarray:
        return encode_obs(self.game, seat, self.static)

    def pending_reward(self, seat: int) -> float:
        """Shaping reward accrued since ``seat``'s last decision (for bootstrapping
        at rollout boundaries)."""
        if seat not in self.prev_phi:
            return 0.0
        return self.cfg.gamma * self.phi(seat) - self.prev_phi[seat]

    def phi(self, seat: int) -> float:
        c, g = self.cfg, self.game
        v = 0.0
        if c.shaping_vp:
            v += c.shaping_vp * g.victory_points(seat) / g.config.vp_to_win
        if c.shaping_prod:
            pl = g.players[seat]
            pips = sum(self._pip[x] for x in pl.settlements) + 2 * sum(self._pip[x] for x in pl.cities)
            if c.prod_robber_aware:  # production blocked by the robber doesn't count
                h = g.robber
                p = PIPS.get(g.layout.hex_number[h], 0)
                pips -= p * sum(g.vertex_level[v] for v in TOPOLOGY.hex_vertices[h] if g.vertex_owner[v] == seat)
            v += c.shaping_prod * pips / 36 / 3
        return v

    # ------------------------------------------------------------ internals
    @staticmethod
    def _vertex_pips(layout):
        out = [0] * TOPOLOGY.num_vertices
        for v in range(TOPOLOGY.num_vertices):
            for h in TOPOLOGY.vertex_hexes[v]:
                if layout.hex_terrain[h] != DESERT:
                    out[v] += PIPS[layout.hex_number[h]]
        return out

    def _advance(self) -> Step:
        g, c = self.game, self.cfg
        while not g.is_over:
            seat = g.actor
            legal = g.legal_actions()
            agent = self.agents[seat]
            if agent is not None:
                g.step(agent.act(g, legal))
                continue
            mask = filter_mask(action_mask(legal, seat, g.n), g, seat, c)
            if c.auto_forced and mask.sum() == 1:
                g.step(decode_action(int(mask.argmax()), seat, g.n))
                continue
            r = self.pending_reward(seat)
            return Step(seat=seat, obs=encode_obs(g, seat, self.static), mask=mask, reward=r)
        boot = g.truncated and c.truncation == "bootstrap"
        final = {}
        for s in self.prev_phi:
            if g.winner == s:
                term = c.win_reward
            elif g.truncated:
                term = 0.0 if boot else c.truncation_reward
            else:
                term = c.loss_reward
            if c.terminal_vp_reward:  # changes the objective (not potential-based): credit for close losses
                term += c.terminal_vp_reward * min(g.victory_points(s), g.config.vp_to_win) / g.config.vp_to_win
            # Phi(terminal) = 0 keeps shaping policy-invariant; with bootstrap we
            # continue the potential instead.
            shaping = self.pending_reward(s) if boot else -self.prev_phi[s]
            final[s] = term + shaping
        return Step(done=True, final=final, truncated=g.truncated, bootstrap=boot)
