import random

import numpy as np
import pytest
import torch

from catan.agents.registry import make_agent, available_agents
from catan.engine import Game, GameConfig
from catan.rl.env import CatanEnv, EnvConfig
from catan.rl.model import PolicyNet, save_checkpoint, load_checkpoint
from catan.rl.ppo import DEFAULTS, merge, train
from catan.rl.rollout import Collector
from catan.rl.spaces import decode_action, OBS_DIM, ACT_DIM


def _random_index(st, rng):
    return int(rng.choice(np.nonzero(st.mask)[0]))


def _play(env, seed, agents, rng):
    """Play one game with random external choices; returns (decisions, final step, seat rewards)."""
    st = env.reset(seed, agents)
    decisions, rewards = [], {}
    while not st.done:
        assert agents[st.seat] is None
        assert st.seat == env.game.actor
        assert st.mask.sum() >= (2 if env.cfg.auto_forced else 1)
        assert env.game.is_legal(decode_action(int(np.nonzero(st.mask)[0][0]), st.seat, env.game.n))
        rewards[st.seat] = rewards.get(st.seat, 0.0) + st.reward
        decisions.append(st.seat)
        st = env.step(_random_index(st, rng))
    for s, r in st.final.items():
        rewards[s] = rewards.get(s, 0.0) + r
    return decisions, st, rewards


def test_multi_agent_sequencing_and_terminal_reward():
    env = CatanEnv(EnvConfig(max_turns=400))
    rng = random.Random(0)
    agents = [None, make_agent("random", seed=1), None, make_agent("random", seed=3)]
    for seed in range(4):
        decisions, st, rewards = _play(env, seed, agents, rng)
        assert set(decisions) <= {0, 2}
        assert set(st.final) == {0, 2}
        g = env.game
        assert g.is_over
        for s in (0, 2):
            assert rewards[s] == pytest.approx(1.0 if g.winner == s else 0.0)


def test_potential_shaping_telescopes():
    """With gamma=1 the shaping sums to Phi(T)-Phi(0) = 0, so the total return is
    the terminal reward exactly."""
    env = CatanEnv(EnvConfig(max_turns=300, gamma=1.0, shaping_vp=1.0, shaping_prod=1.0))
    rng = random.Random(1)
    agents = [None] * 4
    nonzero = False
    for seed in range(3):
        _, st, rewards = _play(env, seed, agents, rng)
        g = env.game
        for s in range(4):
            assert rewards[s] == pytest.approx(1.0 if g.winner == s else 0.0, abs=1e-6)
        nonzero = nonzero or max(env.prev_phi.values()) > 0
    assert nonzero


def test_truncation_modes():
    for mode, expect_boot in (("loss", False), ("bootstrap", True)):
        env = CatanEnv(EnvConfig(max_turns=8, truncation=mode, truncation_reward=-0.5))
        _, st, rewards = _play(env, 0, [None] * 4, random.Random(0))
        assert st.truncated and env.game.winner == -1
        assert st.bootstrap == expect_boot
        for s in range(4):
            assert rewards[s] == pytest.approx(0.0 if expect_boot else -0.5)


def test_seed_reproducibility():
    def run():
        env = CatanEnv(EnvConfig(max_turns=200))
        decisions, _, _ = _play(env, 42, [None, make_agent("random", seed=0), None, None], random.Random(7))
        return decisions, [a.to_json() for a in env.game.action_log]
    assert run() == run()


def _cfg(**over):
    base = merge(DEFAULTS, {"env": {"max_turns": 60, "shaping_vp": 1.0}, "model": {"hidden": 32, "blocks": 1},
                            "opponents": [{"type": "agent", "name": "random", "weight": 1},
                                          {"type": "self", "weight": 1}, {"type": "pool", "weight": 1}]})
    return merge(base, over)


def test_collector_reproducible_and_gae_shapes():
    cfg = _cfg()
    torch.manual_seed(0)
    sd = PolicyNet(hidden=32, blocks=1).state_dict()
    outs = []
    for _ in range(2):
        c = Collector(cfg, 4, seed=3)
        c.set_weights(sd)
        b = c.collect(2500)
        outs.append(b)
    a, b = outs
    for k in ("obs", "mask", "act", "logp", "adv", "ret"):
        assert np.array_equal(a[k], b[k]), k
    n = len(a["act"])
    assert n >= 2500 and a["obs"].shape == (n, OBS_DIM) and a["mask"].shape == (n, ACT_DIM)
    assert a["mask"][np.arange(n), a["act"]].all()  # stored actions were legal
    assert np.isfinite(a["adv"]).all()
    assert len(a["episodes"]) > 0  # max_turns=60 games finish


def test_checkpoint_roundtrip(tmp_path):
    cfg = _cfg()
    m = PolicyNet(hidden=32, blocks=1)
    p = tmp_path / "x.pt"
    save_checkpoint(p, m, cfg, {"update": 3})
    m2, ck = load_checkpoint(p)
    assert ck["update"] == 3 and ck["config"]["model"]["hidden"] == 32
    x = torch.randn(5, OBS_DIM)
    for t1, t2 in zip(m(x), m2(x)):
        assert torch.allclose(t1, t2)


def test_policy_agent_legal_and_info(tmp_path):
    p = tmp_path / "a.pt"
    save_checkpoint(p, PolicyNet(hidden=32, blocks=1), _cfg())
    assert "rl" in available_agents()
    for det in (False, True):
        agents = [make_agent("rl", checkpoint=str(p), deterministic=det, seed=1)] + \
                 [make_agent("random", seed=i) for i in range(3)]
        g = Game(config=GameConfig(max_turns=150), seed=9)
        while not g.is_over:
            legal = g.legal_actions()
            if g.actor == 0:
                a, info = agents[0].act_with_info(g, legal)
                assert a in legal
                assert info["source"] == "model_output" and isinstance(info["value"], float)
                assert 0 < sum(info["probs"].values()) <= 1.0001
            else:
                a = agents[g.actor].act(g, legal)
            g.step(a)


def test_train_smoke_and_resume(tmp_path):
    cfg = _cfg(total_steps=1200, rollout_steps=600, envs_per_worker=4, checkpoint_every=1,
               pool={"snapshot_every": 1, "max_size": 2},
               eval={"every": 2, "games": 4, "workers": 1, "max_turns": 60},
               ppo={"minibatch": 256, "epochs": 1})
    run = tmp_path / "run"
    train(cfg, str(run), log=lambda s: None)
    for f in ("config.yaml", "meta.json", "metrics.jsonl", "eval.jsonl", "checkpoints/latest.pt"):
        assert (run / f).exists(), f
    cfg["total_steps"] = 1800
    train(cfg, str(run), resume=True, log=lambda s: None)
    ck = torch.load(run / "checkpoints/latest.pt", weights_only=True)
    assert ck["update"] == 3 and ck["step"] >= 1800


def test_pool_eviction_keeps_snapshots_used_by_live_games():
    cfg = _cfg(opponents=[{"type": "pool", "weight": 1}], env={"max_turns": 400})
    sd = PolicyNet(hidden=32, blocks=1).state_dict()
    c = Collector(cfg, 4, seed=0)
    c.set_pool({1: sd})
    for i in range(4):
        c._new_game(i)  # every game now uses snapshot 1
    c.collect(50)
    c.set_pool({2: sd, 1: None})  # evict 1 while its games are still running
    assert 1 in c.pool and c.active == {2}
    c.collect(200)


def test_disable_domestic_trade_filter_env_and_agent(tmp_path):
    from catan.rl.spaces import PROPOSE_MASK
    env = CatanEnv(EnvConfig(max_turns=200, disable_domestic_trade=True))
    st = env.reset(0, [None] * 4)
    rng = random.Random(0)
    n = 0
    while not st.done:
        assert not (st.mask & PROPOSE_MASK).any() and st.mask.sum() >= 2
        st = env.step(_random_index(st, rng))
        n += 1
    assert n > 100
    cfg = _cfg(env={"disable_domestic_trade": True})
    p = tmp_path / "nt.pt"
    save_checkpoint(p, PolicyNet(hidden=32, blocks=1), cfg)
    agent = make_agent("rl", checkpoint=str(p), seed=0)
    g = Game(config=GameConfig(max_turns=100), seed=4)
    others = [make_agent("random", seed=i) for i in range(4)]
    while not g.is_over:
        legal = g.legal_actions()
        a = agent.act(g, legal) if g.actor == 0 else others[g.actor].act(g, legal)
        assert a in legal
        if g.actor == 0:
            assert a.type not in ("OFFER_TRADE", "COUNTER_OFFER")
        g.step(a)


def test_mixed_table_opponents():
    cfg = _cfg(opponents=[{"type": "agent", "name": ["basic", "random", "random"], "weight": 1}])
    c = Collector(cfg, 6, seed=1)
    for i in range(6):
        names = sorted(type(a).__name__ for a in c.envs[i].agents if a is not None)
        assert len(names) == 3 and names.count("RandomAgent") == 2, names
        assert c.mode[i] == "basic+random+random"


def test_mask_self_robber_filter_and_robber_aware_phi():
    from catan.engine import TOPOLOGY
    from catan.rl.env import filter_mask
    from catan.rl.spaces import ROBBER_BASE, action_mask
    env = CatanEnv(EnvConfig(max_turns=300, mask_self_robber=True))
    rng = random.Random(2)
    seen = 0
    for seed in range(6):
        st = env.reset(seed, [None] * 4)
        while not st.done:
            g = env.game
            if g.phase == "MOVE_ROBBER":
                legal_mask = action_mask(g.legal_actions(), st.seat, g.n)
                own = [any(g.vertex_owner[v] == st.seat for v in TOPOLOGY.hex_vertices[h]) for h in range(19)]
                allowed = [h for h in range(19) if st.mask[ROBBER_BASE + h]]
                if any(legal_mask[ROBBER_BASE + h] and not own[h] for h in range(19)):
                    assert not any(own[h] for h in allowed)
                assert allowed
                seen += 1
            st = env.step(_random_index(st, rng))
    assert seen > 5
    # robber-aware potential drops when the robber sits on own production
    e = CatanEnv(EnvConfig(shaping_prod=1.0, prod_robber_aware=True))
    e.reset(1, [None] * 4)
    g = e.game
    r = random.Random(0)
    while g.phase in ("SETUP_SETTLEMENT", "SETUP_ROAD"):
        g.step(r.choice(g.legal_actions()))
    v = g.players[0].settlements[0]
    h = next(x for x in TOPOLOGY.vertex_hexes[v] if g.layout.hex_number[x])
    before = e.phi(0)
    g.robber = h
    assert e.phi(0) < before
    assert filter_mask(np.ones(378, bool), g, 0, EnvConfig()).all()  # no filters -> unchanged


def test_terminal_vp_reward():
    env = CatanEnv(EnvConfig(max_turns=400, terminal_vp_reward=0.5))
    _, st, rewards = _play(env, 3, [None] * 4, random.Random(3))
    g = env.game
    for s in range(4):
        vp = min(g.victory_points(s), 10) / 10
        assert rewards[s] == pytest.approx((1.0 if g.winner == s else 0.0) + 0.5 * vp)


def test_worker_pool_shuts_down_children():
    from catan.rl.rollout import WorkerPool
    wp = WorkerPool(_cfg(), 1, 2, seed=0)
    wp.set_weights(PolicyNet(hidden=32, blocks=1).state_dict())
    assert len(wp.collect(50)["act"]) >= 50
    wp.close()
    assert not any(p.is_alive() for p in wp.procs)
