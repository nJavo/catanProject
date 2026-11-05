"""Agent (policy) interface shared by bots, RL policies, the server and evaluation.

Contract:
- ``act(game, legal)`` is called only when ``game.actor`` is this agent's seat.
- ``legal`` is ``game.legal_actions()``; the returned action must be in it
  (or otherwise pass ``game.is_legal``).
- Agents receive the authoritative ``Game`` for speed, but MUST only use
  information visible to ``game.actor``: board, public counts, their own hand
  and dev cards. Never read opponents' ``hand``/``dev``, ``dev_deck`` order or
  ``rng``. Search agents must determinize hidden info (see tests enforcing this).
- ``act_with_info`` may return diagnostics for the AI lab. Only report real
  model/heuristic outputs (e.g. {"probs": {...}, "value": 0.3}); never fabricate.
"""
from __future__ import annotations

import random

from catan.engine.actions import Action
from catan.engine.game import Game


class Agent:
    name = "agent"

    def act(self, game: Game, legal: list[Action]) -> Action:
        return self.act_with_info(game, legal)[0]

    def act_with_info(self, game: Game, legal: list[Action]) -> tuple[Action, dict]:
        return self.act(game, legal), {}

    def reset(self, seat: int, seed: int | None = None) -> None:
        """Called at the start of each game."""


class RandomAgent(Agent):
    """Uniform over legal actions."""
    name = "random"

    def __init__(self, seed: int | None = None):
        self.rng = random.Random(seed)

    def reset(self, seat, seed=None):
        if seed is not None:
            self.rng = random.Random(seed)

    def act(self, game, legal):
        return self.rng.choice(legal)
