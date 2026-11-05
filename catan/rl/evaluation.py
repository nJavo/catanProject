"""Checkpoint evaluation: one RL seat vs opponents, seat rotation, held-out seeds."""
from __future__ import annotations

import math

from catan.agents.registry import make_agent
from catan.engine import Game, GameConfig
from catan.eval.seeds import TRAIN_EVAL_BASE


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


_AGENT = None


def _init(agent_name, agent_kwargs):
    global _AGENT
    import torch
    torch.set_num_threads(1)
    _AGENT = make_agent(agent_name, **agent_kwargs)


def _play(job):
    """job = (seed, seat, opponent name, num_players, max_turns)."""
    seed, seat, opp, n, max_turns = job
    g = Game(config=GameConfig(num_players=n, max_turns=max_turns), seed=seed)
    agents = [make_agent(opp, seed=seed * 10 + s) if s != seat else _AGENT for s in range(n)]
    for s, a in enumerate(agents):
        a.reset(s, seed * 10 + s)
    decisions = 0
    while not g.is_over:
        if g.actor == seat:
            decisions += 1
        g.step(agents[g.actor].act(g, g.legal_actions()))
    return {"seed": seed, "seat": seat, "opponent": opp, "winner": g.winner,
            "won": g.winner == seat, "truncated": g.truncated, "turns": g.turn,
            "vp": g.victory_points(seat), "opp_vp": max(g.victory_points(q) for q in range(n) if q != seat),
            "decisions": decisions}


def evaluate(checkpoint: str, opponents=("random",), games: int = 200, seed: int = TRAIN_EVAL_BASE,
             num_players: int = 4, max_turns: int = 500, deterministic: bool = False,
             workers: int = 8, agent_name: str = "rl", agent_kwargs: dict | None = None) -> dict:
    """Seat rotation: game i uses seed ``seed + i`` and puts the agent in seat i % n.
    The same seeds are reused for each opponent type."""
    kw = dict(agent_kwargs or {})
    if agent_name == "rl":
        kw.update(checkpoint=checkpoint, deterministic=deterministic)
    jobs = [(seed + i, i % num_players, o, num_players, max_turns) for o in opponents for i in range(games)]
    if workers <= 1:
        _init(agent_name, kw)
        rows = [_play(j) for j in jobs]
    else:
        import multiprocessing as mp
        with mp.get_context("spawn").Pool(workers, initializer=_init, initargs=(agent_name, kw)) as pool:
            rows = pool.map(_play, jobs, chunksize=max(1, len(jobs) // (workers * 4)))
    out = {"checkpoint": checkpoint, "agent": agent_name, "deterministic": deterministic,
           "seeds": [seed, seed + games - 1], "num_players": num_players, "max_turns": max_turns,
           "per_opponent": {}}
    for o in opponents:
        r = [x for x in rows if x["opponent"] == o]
        k = sum(x["won"] for x in r)
        out["per_opponent"][o] = {
            "games": len(r), "wins": k, "win_rate": k / len(r), "ci95": list(wilson(k, len(r))),
            "truncated": sum(x["truncated"] for x in r),
            "avg_vp": sum(x["vp"] for x in r) / len(r),
            "avg_turns": sum(x["turns"] for x in r) / len(r),
            "by_seat": {s: sum(x["won"] for x in r if x["seat"] == s) / max(1, sum(x["seat"] == s for x in r))
                        for s in range(num_players)},
        }
    return out
