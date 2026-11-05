import copy

import numpy as np
import torch

from catan.rl.model import PolicyNet
from catan.rl.ppo import DEFAULTS, ppo_update
from catan.rl.spaces import OBS_DIM, ACT_DIM


def _batch(n=512, seed=0):
    r = np.random.default_rng(seed)
    mask = r.random((n, ACT_DIM)) < 0.05
    mask[:, 0] = True
    act = np.array([r.choice(np.nonzero(m)[0]) for m in mask])
    return {"obs": r.random((n, OBS_DIM), dtype=np.float32), "mask": mask, "act": act,
            "logp": np.full(n, -3.0, np.float32), "adv": r.standard_normal(n).astype(np.float32),
            "ret": r.standard_normal(n).astype(np.float32), "val": np.zeros(n, np.float32)}


def _kl(ref, m, b):
    with torch.no_grad():
        o, k = torch.from_numpy(b["obs"]), torch.from_numpy(b["mask"])
        lr_ = torch.log_softmax(ref(o, k)[0], -1)
        lm = torch.log_softmax(m(o, k)[0], -1)
        return float((lr_.exp() * (lr_ - lm)).sum(-1).mean())


def test_kl_ref_restrains_drift_from_reference():
    torch.manual_seed(0)
    base = PolicyNet(hidden=32, blocks=1)
    b = _batch()
    out = {}
    for beta in (0.0, 5.0):
        m, ref = copy.deepcopy(base), copy.deepcopy(base)
        p = dict(DEFAULTS["ppo"], kl_ref_coef=beta, minibatch=128, epochs=4, target_kl=None)
        opt = torch.optim.Adam(m.parameters(), lr=3e-3)
        st = ppo_update(m, opt, b, p, "cpu", torch.Generator().manual_seed(0), ref)
        out[beta] = _kl(ref, m, b)
        assert ("kl_ref" in st) == bool(beta)
    assert 0 <= out[5.0] < out[0.0]


def test_float16_rollouts_train():
    from catan.rl.ppo import merge
    from catan.rl.rollout import Collector
    cfg = merge(DEFAULTS, {"env": {"max_turns": 60}, "model": {"hidden": 32, "blocks": 1}, "obs_dtype": "float16"})
    c = Collector(cfg, 3, seed=0)
    b = c.collect(300)
    assert b["obs"].dtype == np.float16 and len(b["obs"]) == len(b["act"])
    m = PolicyNet(hidden=32, blocks=1)
    st = ppo_update(m, torch.optim.Adam(m.parameters()), b, dict(DEFAULTS["ppo"], minibatch=64), "cpu",
                    torch.Generator().manual_seed(0), copy.deepcopy(m))
    assert np.isfinite(st["policy_loss"])
