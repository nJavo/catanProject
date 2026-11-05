import os

import numpy as np

from catan.agents.registry import make_agent
from catan.engine import Game, GameConfig
from catan.research.arch.obs import OBS_DIM
from catan.research.value import data as D
from catan.research.value.train import calibration, fit_logistic, main as train_main


def test_calibration_and_logistic():
    z = np.array([0, 1, 0, 1], float)
    c = calibration(z, z)
    assert c["brier"] < 1e-9 and c["ece"] < 1e-5  # p clipped to [1e-6, 1-1e-6]
    rng = np.random.default_rng(0)
    x = rng.normal(size=20000)
    zz = (rng.random(20000) < 1 / (1 + np.exp(-(2 * x - 1)))).astype(float)
    a, b = fit_logistic(x, zz)
    assert abs(a - 2) < 0.15 and abs(b + 1) < 0.15


def _fake_shard(path, gid0, n_games, rows=6, d2=False):
    gid = np.repeat(np.arange(gid0, gid0 + n_games), rows)
    z = (gid % 2).astype(np.float32)
    obs = np.random.default_rng(gid0).normal(size=(len(gid), OBS_DIM)).astype(np.float16)
    if d2:  # D2's shard schema
        np.savez(path, obs=obs, win=z, seed=gid, turn=np.zeros(len(gid), np.int16))
    else:
        np.savez(path, obs=obs, z=z, gid=gid, turn=np.zeros(len(gid), np.int16),
                 margin=z * 2 - 1 + np.zeros(len(gid), np.float32))


def test_pack_train_and_search_agent(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir(), b.mkdir()
    _fake_shard(a / "shard_0.npz", 0, 40)
    _fake_shard(b / "shard_0.npz", 1000, 40, d2=True)
    out = tmp_path / "packed"
    D.main(["pack", "--src", f"{a}:1.0", f"{b}:0.5", "--out", str(out)])
    meta = np.load(out / "meta.npz")
    obs = D.ObsFile(str(out / "obs.npy"))
    assert obs.shape == (len(meta["z"]), OBS_DIM)
    full = np.load(out / "obs.npy")
    assert (obs.rows([0, 1, 2, 7, 9]) == full[[0, 1, 2, 7, 9]]).all()
    assert len(np.unique(meta["gid"][meta["source"] == 1])) < 40  # whole-game subsample
    run = tmp_path / "run"
    train_main(["--data", str(out), "--out", str(run), "--epochs", "1", "--batch", "16", "--threads", "1",
                "--evals-per-epoch", "2"])
    assert sum(1 for x in open(run / "metrics.jsonl") if "\"step\": 0," in x) == 1  # step-0 eval of the init
    ck = str(run / "best.pt")
    assert os.path.exists(ck)
    ag = make_agent("value_search", value=ck, sims=2, m=2, seed=0)
    g = Game(config=GameConfig(max_turns=8), seed=10_000_321)
    bots = [ag] + [make_agent("basic", seed=s) for s in range(3)]
    while not g.is_over:
        act = bots[g.actor].act(g, g.legal_actions())
        assert g.is_legal(act)
        g.step(act)


def test_value_trade_routes_trades_to_nego():
    from catan.engine.actions import A, Phase
    ag = make_agent("value_trade", sims=2, m=2, seed=0)
    sources = set()
    for seed in (10_000_400, 10_000_401):
        g = Game(config=GameConfig(max_turns=40), seed=seed)
        bots = [ag] + [make_agent("intermediate", seed=s) for s in range(3)]
        ag.reset(0, seed)
        while not g.is_over:
            legal = g.legal_actions()
            if g.actor == 0:
                act, info = ag.act_with_info(g, legal)
                sources.add(info.get("source"))
                if g.phase in (Phase.TRADE_RESPONSE, Phase.TRADE_CONFIRM):
                    assert info["source"] == "nego"
                elif act.type == A.OFFER_TRADE:
                    assert info["source"] == "nego_offer"  # search never proposes trades itself
            else:
                act = bots[g.actor].act(g, legal)
            assert g.is_legal(act)
            ag.observe(g, g.step(act))  # the nego belief tracker, as the tournament feeds it
    assert "value_search" in sources and "nego" in sources
