"""Expert Iteration loop (Anthony et al. 2017; AlphaZero-style targets).

    python -m catan.research.exit.train --config configs/exit_r1.yaml

Each iteration:
1. expert play: ``ExitSearchAgent`` (prior + value from the current apprentice)
   plays ``search_seats`` seats per game against heuristic bots; every searched
   decision records (obs, mask, improved policy pi', seat). Seeds >= 1e7.
2. distill: fine-tune the apprentice on the last ``replay_iters`` iterations:
   soft cross-entropy to pi' + vf_coef * MSE(V, z), z = 1 if that seat won.
3. evaluate raw policy and search agent vs 3x basic on block 53 (selection only).
Writes runs/<run>/metrics.jsonl, eval.jsonl, checkpoints/iter_XX.pt.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import signal
import sys
import time

import numpy as np
import torch
import yaml

from catan.agents.registry import make_agent
from catan.engine import Game, GameConfig
from catan.rl import model as rl_model


def _io(path):
    """(load, save, raw-policy agent name) for an rl (MLP) or D5 arch (GNN) checkpoint."""
    if torch.load(path, map_location="cpu", weights_only=True).get("format") == "catan-arch":
        from catan.research.arch import agent as arch
        return arch.load_checkpoint, arch.save_checkpoint, "arch_rl"
    return rl_model.load_checkpoint, rl_model.save_checkpoint, "rl"

from .search import ExitSearchAgent

from catan.eval.seeds import TRAIN_SEED_BASE as _TRAIN_BASE
TRAIN_SEED_BASE = _TRAIN_BASE + 53 * 1_000_000  # disjoint from bc.py's BC_SEED_BASE block

DEFAULTS = {
    "run_name": "exit_r1", "init_checkpoint": "models/ppo_baseline.pt", "iterations": 4,
    "games_per_iter": 300, "workers": 3, "search_seats": 2, "opponents": ["basic", "intermediate"],
    "search": {"sims": 16, "m": 8, "leaf": "heur", "gumbel": True},
    "train": {"epochs": 3, "lr": 1e-4, "batch": 1024, "vf_coef": 0.5, "replay_iters": 2},
    "eval": {"games": 100, "block": 53, "search_sims": 16, "opponents": ["basic"]},
}


def _merge(a, b):
    out = dict(a)
    for k, v in (b or {}).items():
        out[k] = _merge(a[k], v) if isinstance(v, dict) and isinstance(a.get(k), dict) else v
    return out


def _play(job):
    """Worker: play games, return stacked (obs, mask, target, z) for searched seats."""
    ckpt, seeds, cfg = job
    torch.set_num_threads(1)
    obs, masks, tgts, zs, gids = [], [], [], [], []
    wins = searched = 0
    for seed in seeds:
        rng = random.Random(seed)
        seats = rng.sample(range(4), cfg["search_seats"])
        agents = []
        for s in range(4):
            if s in seats:
                agents.append(ExitSearchAgent(checkpoint=ckpt, record=True, **cfg["search"]))
            else:
                agents.append(make_agent(rng.choice(cfg["opponents"])))
        for s, a in enumerate(agents):
            a.reset(s, seed * 10 + s)
        g = Game(config=GameConfig(max_turns=500), seed=seed)
        try:
            while not g.is_over:
                g.step(agents[g.actor].act(g, g.legal_actions()))
        except Exception:  # drop the game, keep the iteration alive
            import traceback
            traceback.print_exc()
            continue
        for s in seats:
            searched += 1
            wins += g.winner == s
            for me, o, m, t in agents[s].records:
                obs.append(o)
                masks.append(m)
                tgts.append(t)
                zs.append(float(g.winner == me))
                gids.append(seed)
    return (np.stack(obs), np.stack(masks), np.stack(tgts), np.asarray(zs, np.float32),
            np.asarray(gids, np.int64), wins, searched)


def generate(ckpt, it, cfg):
    n, w = cfg["games_per_iter"], max(1, cfg["workers"])
    seeds = [TRAIN_SEED_BASE + it * 100_000 + i for i in range(n)]
    jobs = [(ckpt, seeds[k::w], cfg) for k in range(w)]
    if w == 1:
        parts = [_play(j) for j in jobs]
    else:
        import multiprocessing as mp
        with mp.get_context("spawn").Pool(w) as pool:
            parts = pool.map_async(_play, jobs).get(timeout=10**6)  # timed wait: SIGTERM handler can run
    data = tuple(np.concatenate([p[i] for p in parts]) for i in range(5))
    return data, sum(p[5] for p in parts), sum(p[6] for p in parts)


def distill(model, data, tc, seed):
    obs, mask, tgt, z, gid = data
    # split by game: z is shared by every decision of a game, so a per-decision split leaks it
    is_hold = gid % 20 == 0
    hold = np.flatnonzero(is_hold)
    tr = np.flatnonzero(~is_hold)
    opt = torch.optim.Adam(model.parameters(), lr=tc["lr"])
    to = lambda i: (torch.from_numpy(obs[i].astype(np.float32)), torch.from_numpy(mask[i]),
                    torch.from_numpy(tgt[i]), torch.from_numpy(z[i]))

    def losses(i):
        o, m, t, zz = to(i)
        logits, v = model(o, m)
        logp = torch.log_softmax(logits, -1)
        pl = -(t * logp).sum(-1).mean()
        return pl, ((v - zz) ** 2).mean(), (logits.argmax(-1) == t.argmax(-1)).float().mean()

    def heldout():
        model.eval()
        with torch.no_grad():
            pl, vl, acc = losses(hold)
        model.train()
        return {"heldout_policy_loss": pl.item(), "heldout_value_loss": vl.item(), "heldout_top1": acc.item()}

    hist = [{"epoch": 0, **heldout()}]
    score = lambda h: h["heldout_policy_loss"] + tc["vf_coef"] * h["heldout_value_loss"]
    best = (score(hist[0]), 0, {k: v.clone() for k, v in model.state_dict().items()})
    model.train()
    for ep in range(tc["epochs"]):
        idx = tr[np.random.default_rng(seed + ep + 1).permutation(len(tr))]
        for s in range(0, len(idx) - tc["batch"] + 1, tc["batch"]):
            pl, vl, _ = losses(idx[s:s + tc["batch"]])
            loss = pl + tc["vf_coef"] * vl
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        hist.append({"epoch": ep + 1, **heldout()})
        if score(hist[-1]) < best[0]:
            best = (score(hist[-1]), ep + 1, {k: v.clone() for k, v in model.state_dict().items()})
    model.load_state_dict(best[2])  # early stopping on held-out games
    model.eval()
    hist.append({"best_epoch": best[1], **hist[best[1]]})
    return hist


def _eval(ckpt, it, step, ec, workers):
    """Runs scripts.exit_eval in a fresh interpreter: forking this process after torch
    training deadlocked the tournament workers (fork-after-threads)."""
    import subprocess
    import sys
    import tempfile
    raw = _io(ckpt)[2]
    specs = {raw: f"{raw}:checkpoint={ckpt},deterministic=true",
             "exit_search": f"exit_search:checkpoint={ckpt},sims={ec['search_sims']}"}
    rows = []
    for agent, spec in specs.items():
        with tempfile.NamedTemporaryFile(suffix=".jsonl") as tmp:
            subprocess.run([sys.executable, "-m", "scripts.exit_eval", "--specs", spec, "--opponents",
                            *ec["opponents"], "--games", str(ec["games"]), "--block", str(ec["block"]),
                            "--workers", str(workers), "--out", tmp.name], check=True, stdout=subprocess.DEVNULL)
            res = [json.loads(x) for x in open(tmp.name)]
        per = {r["opponent"]: {k: r[k] for k in ("games", "wins", "win_rate", "ci95", "avg_vp", "ms_per_move")}
               for r in res}
        rows.append({"checkpoint": ckpt, "agent": agent, "spec": spec, "deterministic": True,
                     "seed_block": ec["block"], "num_players": 4, "max_turns": 500, "per_opponent": per,
                     "step": step, "update": it, "time": time.time()})
    return rows


def _log_eval(run, rows):
    for r in rows:
        with open(os.path.join(run, "eval.jsonl"), "a") as f:
            f.write(json.dumps(r) + "\n")
        print(json.dumps({"agent": r["agent"], **{o: (v["win_rate"], v["ci95"]) for o, v in r["per_opponent"].items()}}), flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--start-iter", type=int, default=0, help="resume: iteration to start at")
    a = ap.parse_args(argv)
    # SIGTERM -> SystemExit so `with Pool` exits and terminates its workers (rule 3b: no orphans)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    with open(a.config) as f:
        cfg = _merge(DEFAULTS, yaml.safe_load(f))
    run = os.path.join("runs", cfg["run_name"])
    os.makedirs(os.path.join(run, "checkpoints"), exist_ok=True)
    with open(os.path.join(run, "config.yaml"), "w") as f:
        yaml.safe_dump(cfg, f)
    torch.set_num_threads(1)
    ckpt = cfg["init_checkpoint"] if a.start_iter == 0 else os.path.join(run, "checkpoints", f"iter_{a.start_iter:02d}.pt")
    load_checkpoint, save_checkpoint, _ = _io(ckpt)
    model, meta = load_checkpoint(ckpt)
    base_cfg = meta.get("config") or {}  # keeps env action filters and model size
    replay, step, t0 = [], int(meta.get("step") or 0) if a.start_iter else 0, time.time()
    ev_path = os.path.join(run, "eval.jsonl")
    done = {json.loads(x)["update"] for x in open(ev_path)} if os.path.exists(ev_path) else set()
    if a.start_iter and a.start_iter not in done:  # resumed checkpoint was never evaluated
        _log_eval(run, _eval(ckpt, a.start_iter, step, cfg["eval"], cfg["workers"]))
    for it in range(a.start_iter, cfg["iterations"]):
        tg = time.time()
        data, wins, searched = generate(ckpt, it, cfg)
        t_gen = time.time() - tg
        # saved under checkpoints/ (ignored) so distillation can be re-run offline
        np.savez(os.path.join(run, "checkpoints", f"data_{it + 1:02d}.npz"), obs=data[0], mask=data[1],
                 target=data[2], z=data[3], gid=data[4])
        replay = (replay + [data])[-cfg["train"]["replay_iters"]:]
        cat = tuple(np.concatenate([d[i] for d in replay]) for i in range(5))
        step += len(data[3])
        hist = distill(model, cat, cfg["train"], seed=it)
        ckpt = os.path.join(run, "checkpoints", f"iter_{it + 1:02d}.pt")
        save_checkpoint(ckpt, model, base_cfg, {"update": it + 1, "step": step, "exit_config": cfg})
        row = {"step": step, "time": time.time(), "update": it + 1, "samples_new": int(len(data[3])),
               "samples_train": int(len(cat[3])), "expert_win_rate_selfgen": wins / max(1, searched),
               "expert_games": searched, "gen_s": t_gen, "wall_s": time.time() - t0,
               "target_entropy": float(-(data[2] * np.log(np.clip(data[2], 1e-9, 1))).sum(-1).mean()),
               **{f"before_{k}": v for k, v in hist[0].items() if k != "epoch"},
               **{k: v for k, v in hist[-1].items() if k != "epoch"}}
        with open(os.path.join(run, "metrics.jsonl"), "a") as f:
            f.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)
        _log_eval(run, _eval(ckpt, it + 1, step, cfg["eval"], cfg["workers"]))


if __name__ == "__main__":
    main()
