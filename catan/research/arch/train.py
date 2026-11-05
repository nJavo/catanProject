"""D5 pipeline: BC warm start on intermediate games, then PPO, for any arch/obs spec.

    python -m catan.research.arch.train collect --games 600 --workers 3 --out /tmp/arch_data
    python -m catan.research.arch.train bc  --config configs/arch_gnn_v2.yaml --data /tmp/arch_data
    python -m catan.research.arch.train ppo --config configs/arch_gnn_v2.yaml

Every arm trains on the same BC decisions (stored once with obs v2; v1 is its
prefix) and runs catan.rl.ppo.train unchanged. ``install`` swaps the model
class, observation encoder, checkpoint format and eval agent inside catan.rl
for this process only (in-process collection, num_workers=0), so no catan.rl
code is copied or edited.
"""
from __future__ import annotations

import os

# Rule 3 (CPU budget): one compute thread per process, set before numpy/torch load.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import functools
import json
import time

import numpy as np
import torch

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

from catan.agents.registry import make_agent
from catan.engine import Game, GameConfig
from catan.rl import spaces as S
from . import obs as O
from .agent import save_checkpoint, load_checkpoint
from .model import build, obs_dim, param_count

from catan.eval.seeds import BC_SEED_BASE  # same games as catan.rl.bc (seed 0)


def install(cfg: dict):
    from catan.rl import ppo, rollout, env, evaluation
    spec = cfg["arch"]
    factory = lambda *a, **k: build(spec)  # noqa: E731
    ppo.PolicyNet = rollout.PolicyNet = factory
    ppo.OBS_DIM, ppo.OBS_VERSION = obs_dim(spec["obs"]), spec["obs"]
    ppo.save_checkpoint, ppo.load_checkpoint = save_checkpoint, load_checkpoint
    env.encode_obs = O.encode_obs if spec["obs"] == 2 else S.encode_obs
    if cfg["ppo"].get("kl_bc_coef"):  # anchor to the BC policy the run starts from
        from .klppo import make_update
        ppo.ppo_update = make_update(load_checkpoint(cfg["init_checkpoint"])[0], float(cfg["ppo"]["kl_bc_coef"]))
    if not getattr(evaluation, "_arch_orig", None):
        evaluation._arch_orig = evaluation.evaluate

    @functools.wraps(evaluation._arch_orig)
    def ev(checkpoint, opponents, games, seed, num_players, max_turns, deterministic, workers):
        return evaluation._arch_orig(checkpoint, opponents, games, seed, num_players, max_turns, deterministic,
                                     workers, agent_name="arch_rl",
                                     agent_kwargs={"checkpoint": checkpoint, "deterministic": deterministic})
    evaluation.evaluate = ev


# ------------------------------------------------------------------ BC data
def _gen(job):
    teacher, seeds, max_turns = job
    torch.set_num_threads(1)
    obs, masks, acts, seat_of, wins = [], [], [], [], []
    for seed in seeds:
        g = Game(config=GameConfig(max_turns=max_turns), seed=seed)
        agents = [make_agent(teacher, seed=seed * 10 + s) for s in range(4)]
        for s, a in enumerate(agents):
            a.reset(s, seed * 10 + s)
        st = S.static_features(g.layout)
        start = len(acts)
        while not g.is_over:
            legal = g.legal_actions()
            me = g.actor
            a = agents[me].act(g, legal)
            if len(legal) > 1:
                try:
                    i = S.encode_action(a, me, g.n)
                except KeyError:
                    i = None
                m = S.action_mask(legal, me, g.n)
                if i is not None and m[i]:
                    obs.append(O.encode_obs(g, me, st).astype(np.float16))
                    masks.append(m)
                    acts.append(i)
                    seat_of.append(me)
            g.step(a)
        wins.extend(float(g.winner == s) for s in seat_of[start:])
    return np.stack(obs), np.stack(masks), np.asarray(acts, np.int64), np.asarray(wins, np.float32)


def collect(a):
    seeds = [BC_SEED_BASE + a.seed * 100_000 + i for i in range(a.games)]
    jobs = [(a.teacher, seeds[w::a.workers], 500) for w in range(a.workers)]
    import multiprocessing as mp
    t = time.time()
    with mp.get_context("spawn").Pool(a.workers) as pool:
        parts = pool.map(_gen, jobs)
    os.makedirs(a.out, exist_ok=True)
    for k, name in enumerate(("obs", "mask", "act", "win")):
        np.save(os.path.join(a.out, f"{name}.npy"), np.concatenate([p[k] for p in parts]))
    n = sum(len(p[2]) for p in parts)
    json.dump({"teacher": a.teacher, "games": a.games, "seed": a.seed, "decisions": n, "obs_dim": O.OBS_DIM,
               "collect_s": time.time() - t}, open(os.path.join(a.out, "data.json"), "w"), indent=1)
    print(f"{n} decisions in {time.time() - t:.0f}s", flush=True)


def bc(a):
    from catan.rl import bc as rbc
    from catan.rl.ppo import load_config
    cfg = load_config(a.config)
    torch.set_num_threads(1)
    spec = cfg["arch"]
    rbc.PolicyNet = lambda *x, **k: build(spec)
    load = lambda n: np.load(os.path.join(a.data, f"{n}.npy"), mmap_mode="r")  # noqa: E731
    obs, mask, act, win = load("obs"), load("mask"), np.asarray(load("act")), np.asarray(load("win"))
    if spec["obs"] == 1:
        obs = obs[:, :S.OBS_DIM]
    out = os.path.join("runs", cfg["run_name"] + "_bc")
    os.makedirs(out, exist_ok=True)
    from catan.rl.ppo import _git_commit, _git_dirty, _hardware
    json.dump({"run_id": os.path.basename(out), "git_commit": _git_commit(), "git_dirty": _git_dirty(),
               "seeds": {"bc_data": a.data, "fit": a.seed}, "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
               "hardware": _hardware(), "arch": spec}, open(os.path.join(out, "meta.json"), "w"), indent=2)
    t = time.time()
    model, hist = rbc.fit(cfg, obs, mask, act, win, epochs=a.epochs, seed=a.seed,
                          log=lambda s: print(s, f"t={time.time() - t:.0f}s", flush=True))
    path = os.path.join(out, "bc.pt")
    save_checkpoint(path, model, cfg, {"bc_teacher": "intermediate", "bc_data": a.data})
    json.dump({"arch": spec, "params": param_count(model), "decisions": int(len(act)), "fit_s": time.time() - t,
               "history": hist}, open(os.path.join(out, "bc_summary.json"), "w"), indent=1)
    print("wrote", path, flush=True)


def ppo(a):
    from catan.rl import ppo as rppo
    cfg = rppo.load_config(a.config)
    install(cfg)
    run_dir = os.path.join("runs", cfg["run_name"])
    rppo.train(cfg, run_dir, resume=a.resume, log=lambda s: print(s, flush=True))


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--teacher", default="intermediate")
    c.add_argument("--games", type=int, default=600)
    c.add_argument("--workers", type=int, default=3)
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--out", required=True)
    b = sub.add_parser("bc")
    b.add_argument("--config", required=True)
    b.add_argument("--data", required=True)
    b.add_argument("--epochs", type=int, default=8)
    b.add_argument("--seed", type=int, default=0)
    p = sub.add_parser("ppo")
    p.add_argument("--config", required=True)
    p.add_argument("--resume", action="store_true")
    a = ap.parse_args(argv)
    {"collect": collect, "bc": bc, "ppo": ppo}[a.cmd](a)


if __name__ == "__main__":
    main()
