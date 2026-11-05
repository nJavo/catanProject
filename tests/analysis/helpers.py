import random
from pathlib import Path

from catan.agents.registry import make_agent
from catan.engine import Game, GameConfig

ROOT = Path(__file__).resolve().parents[2]
CKPT = next(iter(sorted((ROOT / "runs").glob("*/checkpoints/ckpt_*.pt"))), None) if (ROOT / "runs").is_dir() else None


def play_until(seed, pred, agent="intermediate", max_turns=300):
    """Play a game with ``agent`` in every seat until ``pred(game)`` holds."""
    ag = make_agent(agent, seed=seed)
    g = Game(seed=seed, config=GameConfig(max_turns=max_turns))
    while not g.is_over and not pred(g):
        g.step(ag.act(g, g.legal_actions()))
    return g


def sampled_positions(n_games=8, p=0.06, agent="intermediate", seed=0):
    """(game, viewer) pairs at random points of bot games (cloned)."""
    rnd = random.Random(seed)
    out = []
    for s in range(n_games):
        ag = make_agent(agent, seed=s)
        g = Game(seed=100 + s, config=GameConfig(max_turns=150))
        while not g.is_over:
            if rnd.random() < p:
                out.append((g.clone(), rnd.randrange(g.n)))
            g.step(ag.act(g, g.legal_actions()))
    return out
