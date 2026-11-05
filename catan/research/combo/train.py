"""PPO fine-tune of the league GNN with trades delegated to the nego layer (D13).

    python -m scripts.combo_train --config configs/combo_r1.yaml

catan.rl.ppo.train with: the D5 arch installed (catan.research.arch.train.install),
ComboEnv as the rollout env (learner trades handled by the nego layer,
domestic proposals masked for the learner), and a KL penalty toward the
starting GNN (catan.rl.ppo ``kl_ref_coef``). In-process collection: 1 process.
The resulting checkpoint is played by ``combo:checkpoint=...``.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import torch

from catan.agents.registry import make_agent, register


def main(config_path, resume=False, log=print):
    from catan.research.arch.train import install
    from catan.rl import ppo as rppo
    from .env import install_env

    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    cfg = rppo.load_config(config_path)
    install(cfg)  # KL to the start policy: catan.rl.ppo's own ppo.kl_ref_coef
    install_env()
    register("combo_champ", lambda seed=None, **kw: make_agent("nego", seed=seed, checkpoint="models/nego_best.pt"))
    run_dir = os.path.join("runs", cfg["run_name"])
    rppo.train(cfg, run_dir, resume=resume, log=log)
