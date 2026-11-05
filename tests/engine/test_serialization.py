

def test_replay_with_explicit_layout_matches_original():
    import random as _r
    from catan.engine import Game, GameConfig
    g = Game(seed=123, config=GameConfig(max_turns=200))
    rng = _r.Random(1)
    for _ in range(400):
        if g.is_over:
            break
        g.step(rng.choice(g.legal_actions()))
    r = Game.replay(g.seed, g.config, g.action_log, layout=g.layout)
    assert r.to_dict() == g.to_dict()
