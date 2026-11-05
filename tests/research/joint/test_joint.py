import random

import torch

from catan.agents.registry import make_agent
from catan.engine import Game, GameConfig
from catan.research.joint.features import EXTRA_F, CoreFeatures
from catan.research.joint.model import JointNet, load, save
from catan.research.nego.model import NegoNet, batch
from catan.research.nego.options import STATE_F, OPT_F


def _sample(rng, k=5, extra=EXTRA_F):
    return {"s": [rng.random() for _ in range(STATE_F + extra)], "o": [[rng.random() for _ in range(OPT_F)] for _ in range(k)],
            "give": [[rng.randrange(3) for _ in range(5)] for _ in range(k)],
            "get": [[rng.randrange(3) for _ in range(5)] for _ in range(k)]}


def test_from_nego_is_identical_at_init():
    torch.manual_seed(0)
    nego = NegoNet().eval()
    j = JointNet.from_nego(nego).eval()
    rng = random.Random(0)
    x = [_sample(rng) for _ in range(4)]
    lj, vj = j(*batch(x))
    ln, vn = nego(*batch([{**d, "s": d["s"][:STATE_F]} for d in x]))
    assert torch.allclose(lj, ln, atol=1e-6) and torch.allclose(vj, vn, atol=1e-6)


def test_core_features_shape_and_cache():
    g = Game(config=GameConfig(max_turns=30), seed=1)
    rng = random.Random(0)
    for _ in range(60):
        g.step(rng.choice(g.legal_actions()))
    cf = CoreFeatures()
    f = cf(g, g.actor)
    assert len(f) == EXTRA_F and 0.0 <= f[1] <= 1.0
    assert cf(g, g.actor) is f


def test_joint_agent_plays(tmp_path):
    nego = NegoNet()
    path = tmp_path / "j.pt"
    save(JointNet.from_nego(nego), str(path), {})
    load(str(path))
    agents = [make_agent("joint", seed=0, trade_checkpoint=str(path))] + \
             [make_agent("intermediate_trader", seed=i) for i in (1, 2, 3)]
    for i, a in enumerate(agents):
        a.reset(i, i)
    g = Game(config=GameConfig(max_turns=60), seed=3)
    while not g.is_over:
        legal = g.legal_actions()
        a = agents[g.actor].act(g, legal)
        assert g.is_legal(a)
        evs = g.step(a)
        for i, ag in enumerate(agents):
            if hasattr(ag, "observe"):
                ag.observe(g, [e for e in evs if "visible_to" not in e or i in e["visible_to"]])


def test_bundle_equals_head_file():
    import os
    import pytest
    if not os.path.isfile("runs/joint_r2/joint_ep5.pt"):
        pytest.skip("head file not available")
    logs = []
    for kw in ({"checkpoint": "models/joint_best.pt"}, {"trade_checkpoint": "runs/joint_r2/joint_ep5.pt"}):
        agents = [make_agent("joint", seed=0, **kw)] + [make_agent("intermediate_trader", seed=i) for i in (1, 2, 3)]
        for i, a in enumerate(agents):
            a.reset(i, i)
        g = Game(config=GameConfig(max_turns=50), seed=8)
        while not g.is_over:
            evs = g.step(agents[g.actor].act(g, g.legal_actions()))
            for i, ag in enumerate(agents):
                if hasattr(ag, "observe"):
                    ag.observe(g, [e for e in evs if "visible_to" not in e or i in e["visible_to"]])
        logs.append(g.action_log)
    assert logs[0] == logs[1]
