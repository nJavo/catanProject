"""Run one headless game between named agents.

    python -m catan.play --agents random random random random --seed 1
"""
from __future__ import annotations

import argparse
import json

from catan.agents.registry import make_agent
from catan.engine import Game, GameConfig
from catan.engine.replay import make_replay


def play_game(agents, seed: int, config: GameConfig | None = None):
    config = config or GameConfig(num_players=len(agents))
    g = Game(config=config, seed=seed)
    for i, a in enumerate(agents):
        a.reset(i, seed * 1000 + i)
    while not g.is_over:
        g.step(agents[g.actor].act(g, g.legal_actions()))
    return g


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--agents", nargs="+", default=["random"] * 4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-turns", type=int, default=1000)
    ap.add_argument("--save", help="write replay JSON here")
    a = ap.parse_args(argv)
    cfg = GameConfig(num_players=len(a.agents), max_turns=a.max_turns)
    g = play_game([make_agent(n) for n in a.agents], a.seed, cfg)
    vps = [g.victory_points(p) for p in range(g.n)]
    print(json.dumps({"winner": g.winner, "turns": g.turn, "vp": vps, "truncated": g.truncated}))
    if a.save:
        with open(a.save, "w") as f:
            json.dump(make_replay(g.seed, cfg.to_dict(), a.agents, g.action_log), f)


if __name__ == "__main__":
    main()
