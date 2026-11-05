import random

from catan.agents.registry import make_agent
from catan.engine import Game, GameConfig, Action
from catan.play import play_game


def test_random_games_finish_and_replay():
    for seed in range(5):
        g = play_game([make_agent("random", seed=seed + i) for i in range(4)], seed,
                      GameConfig(max_turns=2000))
        assert g.is_over
        r = Game.replay(g.seed, g.config, g.action_log)
        assert r.to_dict() == g.to_dict()


def test_serialization_roundtrip_midgame():
    g = Game(seed=11)
    rng = random.Random(0)
    for _ in range(300):
        g.step(rng.choice(g.legal_actions()))
    h = Game.from_dict(g.to_dict())
    assert h.to_dict() == g.to_dict()
    for _ in range(200):
        a = rng.choice(g.legal_actions())
        g.step(a)
        h.step(Action.from_json(a.to_json()))
    assert h.to_dict() == g.to_dict()
