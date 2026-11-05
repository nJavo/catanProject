"""Masked PPO with self-play opponent pool. Entry point: ``train(cfg, run_dir)``."""
from __future__ import annotations

import json
import os
import platform
import random
import signal
import sys
import threading
import subprocess
import time
from collections import defaultdict

import numpy as np
import torch

from catan.eval.seeds import TRAIN_EVAL_BASE
import yaml

from .model import PolicyNet, model_kwargs, save_checkpoint, load_checkpoint
from .rollout import WorkerPool
from .spaces import OBS_DIM, ACT_DIM, OBS_VERSION, ACTION_VERSION

DEFAULTS = {
    "run_name": "ppo",
    "seed": 1,
    "total_steps": 1_000_000,
    "device": "cpu",
    "torch_threads": 4,
    "num_workers": 0,
    "envs_per_worker": 16,
    "rollout_steps": 8192,
    "learner_seats": 1,
    "env": {},
    "model": {"hidden": 512, "blocks": 2},
    "ppo": {"gamma": 0.999, "gae_lambda": 0.95, "lr": 3e-4, "epochs": 4, "minibatch": 2048,
            "clip": 0.2, "vf_coef": 0.5, "ent_coef": 0.01, "max_grad_norm": 0.5, "target_kl": 0.05,
            "norm_adv": True, "kl_ref_coef": 0.0, "kl_ref_checkpoint": None},
    "opponents": [{"type": "agent", "name": "random", "weight": 1.0}],
    "pool": {"snapshot_every": 10, "max_size": 8},
    "eval": {"every": 10, "games": 100, "opponents": ["random"], "workers": 8, "max_turns": 500,
             "seed": TRAIN_EVAL_BASE, "deterministic": False},  # 1000: "train_eval" range
    "checkpoint_every": 10,
    "init_checkpoint": None,
}


def merge(base, over):
    out = dict(base)
    for k, v in (over or {}).items():
        out[k] = merge(base[k], v) if isinstance(v, dict) and isinstance(base.get(k), dict) else v
    return out


def load_config(path) -> dict:
    with open(path) as f:
        return merge(DEFAULTS, yaml.safe_load(f) or {})


def _git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def _git_dirty():
    try:
        return bool(subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"],
                                            text=True, stderr=subprocess.DEVNULL).strip())
    except Exception:
        return None


def _hardware():
    return {"platform": platform.platform(), "machine": platform.machine(),
            "cpu_count": os.cpu_count(), "python": platform.python_version(),
            "torch": torch.__version__, "mps": torch.backends.mps.is_available()}


class Logger:
    def __init__(self, run_dir):
        self.f = open(os.path.join(run_dir, "metrics.jsonl"), "a")
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.tb = SummaryWriter(os.path.join(run_dir, "tb"))
        except Exception:  # tensorboard optional
            self.tb = None

    def log(self, step, d):
        d = {"step": step, "time": time.time(), **d}
        self.f.write(json.dumps(d) + "\n")
        self.f.flush()
        if self.tb:
            for k, v in d.items():
                if isinstance(v, (int, float)) and k not in ("step", "time"):
                    self.tb.add_scalar(k, v, step)

    def close(self):
        self.f.close()
        if self.tb:
            self.tb.close()


def ppo_update(model, opt, batch, p, device, gen, ref_model=None):
    """One PPO update. With ``ref_model`` and ppo.kl_ref_coef > 0, adds
    kl_ref_coef * KL(pi_ref || pi) over legal actions (AlphaStar-style KL to the
    supervised policy)."""
    obs = torch.from_numpy(batch["obs"]).to(device)
    mask = torch.from_numpy(batch["mask"]).to(device)
    act = torch.from_numpy(batch["act"]).to(device)
    old_logp = torch.from_numpy(batch["logp"]).to(device)
    adv_all = torch.from_numpy(batch["adv"]).to(device)
    ret = torch.from_numpy(batch["ret"]).to(device)
    n = len(act)
    stats = defaultdict(list)
    stop = False
    for _ in range(p["epochs"]):
        perm = torch.randperm(n, generator=gen).to(device)
        for idx in perm.chunk(max(1, n // p["minibatch"])):  # equal-ish chunks, no size-1 tail
            ob = obs[idx].float()  # rollouts may store float16 (cfg obs_dtype)
            logits, v = model(ob, mask[idx])
            logp_all = torch.log_softmax(logits, -1)
            logp = logp_all.gather(1, act[idx, None]).squeeze(1)
            probs = logp_all.exp()
            entropy = -(probs * logp_all).sum(-1).mean()  # masked entries contribute 0
            adv = adv_all[idx]
            if p["norm_adv"]:
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            ratio = torch.exp(logp - old_logp[idx])
            pg = -torch.min(ratio * adv, ratio.clamp(1 - p["clip"], 1 + p["clip"]) * adv).mean()
            vloss = 0.5 * ((v - ret[idx]) ** 2).mean()
            loss = pg + p["vf_coef"] * vloss - p["ent_coef"] * entropy
            beta = float(p.get("kl_ref_coef") or 0.0)
            if ref_model is not None and beta > 0:
                with torch.no_grad():
                    ref_logp = torch.log_softmax(ref_model(ob, mask[idx])[0], -1)
                kl_ref = (ref_logp.exp() * (ref_logp - logp_all)).sum(-1).mean()
                loss = loss + beta * kl_ref
                stats["kl_ref"].append(kl_ref.item())
            opt.zero_grad()
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), p["max_grad_norm"])
            opt.step()
            with torch.no_grad():
                kl = ((ratio - 1) - torch.log(ratio)).mean().item()
            stats["policy_loss"].append(pg.item())
            stats["value_loss"].append(vloss.item())
            stats["entropy"].append(entropy.item())
            stats["approx_kl"].append(kl)
            stats["clipfrac"].append(((ratio - 1).abs() > p["clip"]).float().mean().item())
            stats["grad_norm"].append(float(gn))
            if p.get("target_kl") and kl > 1.5 * p["target_kl"]:
                stop = True
                break
        if stop:
            break
    out = {k: float(np.mean(v)) for k, v in stats.items()}
    out["kl_early_stop"] = float(stop)
    y, yhat = batch["ret"], batch["val"]
    out["explained_variance"] = float(1 - np.var(y - yhat) / (np.var(y) + 1e-8))
    return out


def episode_metrics(eps):
    out = {"episodes": len(eps)}
    if not eps:
        return out
    by = defaultdict(list)
    for e in eps:
        by[e["mode"].split(":")[0]].append(e)
    for m, es in by.items():
        out[f"train/{m}/win_rate"] = float(np.mean([e.get("seat_win_rate", e["learner_won"]) for e in es]))  # per learner seat
        out[f"train/{m}/games"] = len(es)
    out["train/truncated_rate"] = float(np.mean([e["truncated"] for e in eps]))
    out["train/turns"] = float(np.mean([e["turns"] for e in eps]))
    out["train/learner_vp"] = float(np.mean([e["learner_vp"] for e in eps]))
    return out


def train(cfg: dict, run_dir: str, resume: bool = False, log=print):
    os.makedirs(os.path.join(run_dir, "checkpoints"), exist_ok=True)
    ck_dir = os.path.join(run_dir, "checkpoints")
    seed = int(cfg["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(int(cfg["torch_threads"]))
    device = cfg["device"]
    p = cfg["ppo"]
    cfg["env"] = dict(cfg["env"], gamma=p["gamma"])  # shaping must use the PPO discount

    model = PolicyNet(OBS_DIM, ACT_DIM, **model_kwargs(cfg)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=p["lr"], eps=1e-5)
    update, step, pool_ids = 0, 0, []
    latest = os.path.join(ck_dir, "latest.pt")
    if resume:
        ck = torch.load(latest, map_location=device, weights_only=True)
        model.load_state_dict(ck["state_dict"])
        opt.load_state_dict(ck["optimizer"])
        update, step, pool_ids = ck["update"], ck["step"], list(ck["pool_ids"])
        log(f"resumed from update {update}, step {step}")
    else:
        with open(os.path.join(run_dir, "config.yaml"), "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
        if cfg.get("init_checkpoint"):  # fine-tune: copy weights (fresh optimizer); init policy seeds the pool
            init_model, _ = load_checkpoint(cfg["init_checkpoint"], device)
            model.load_state_dict(init_model.state_dict())
            save_checkpoint(os.path.join(ck_dir, "pool_0.pt"), model, cfg, {"update": 0, "step": 0})
            pool_ids = [0]
            log(f"initialized from {cfg['init_checkpoint']}")
    ref_model = None
    if float(p.get("kl_ref_coef") or 0.0) > 0:
        ref_path = p.get("kl_ref_checkpoint") or cfg.get("init_checkpoint")
        if not ref_path:
            raise ValueError("ppo.kl_ref_coef > 0 needs ppo.kl_ref_checkpoint or init_checkpoint")
        ref_model, _ = load_checkpoint(ref_path, device)
        for q in ref_model.parameters():
            q.requires_grad_(False)
        log(f"KL-to-reference coef {p['kl_ref_coef']} vs {ref_path}")
    meta_path = os.path.join(run_dir, "meta.json")
    meta = json.load(open(meta_path)) if resume and os.path.exists(meta_path) else {
        "run_id": os.path.basename(os.path.normpath(run_dir)), "git_commit": _git_commit(), "git_dirty": _git_dirty(),
        "seeds": {"train": seed, "eval": cfg["eval"]["seed"]},
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"), "hardware": _hardware(),
        "obs_dim": OBS_DIM, "act_dim": ACT_DIM, "obs_version": OBS_VERSION,
        "action_version": ACTION_VERSION, "resumes": []}
    if resume:
        meta.setdefault("resumes", []).append({"at_update": update, "total_steps": cfg["total_steps"], "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
                                               "git_commit": _git_commit()})
    json.dump(meta, open(meta_path, "w"), indent=2)

    # Resumed runs reseed workers by update so they don't replay the same games.
    workers = WorkerPool(cfg, int(cfg["num_workers"]), int(cfg["envs_per_worker"]), seed + 7919 * update)
    pool_sd = {}
    for pid in pool_ids:
        sd = torch.load(os.path.join(ck_dir, f"pool_{pid}.pt"), map_location="cpu", weights_only=True)["state_dict"]
        pool_sd[pid] = sd
    if pool_sd:
        workers.set_pool(pool_sd)
    logger = Logger(run_dir)
    gen = torch.Generator().manual_seed(seed + update)
    t0 = time.time()
    if threading.current_thread() is threading.main_thread():
        # SIGTERM (pkill) -> SystemExit so the finally below shuts down the workers
        signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    try:
        while step < cfg["total_steps"]:
            tc = time.time()
            workers.set_weights(model.state_dict())
            batch = None  # free the previous rollout before collecting the next (peak memory)
            batch = workers.collect(int(cfg["rollout_steps"]))
            t_collect = time.time() - tc
            n = len(batch["act"])
            step += n
            update += 1
            tu = time.time()
            frac = 1.0 - (step / cfg["total_steps"]) if p.get("anneal_lr") else 1.0
            for gparam in opt.param_groups:
                gparam["lr"] = p["lr"] * max(frac, 0.05)
            model.train()
            stats = ppo_update(model, opt, batch, p, device, gen, ref_model)
            model.eval()
            t_update = time.time() - tu
            stats.update(episode_metrics(batch["episodes"]))
            stats.update({"update": update, "collect_sps": n / t_collect, "collect_s": t_collect,
                          "update_s": t_update, "wall_s": time.time() - t0,
                          "reward_mean": float(batch["rew"].mean()),
                          "value_mean": float(batch["val"].mean()),
                          "legal_actions_mean": float(batch["mask"].sum(1).mean())})
            logger.log(step, stats)
            log(f"upd {update} step {step} sps {n / t_collect:.0f} ent {stats['entropy']:.3f} "
                f"kl {stats['approx_kl']:.4f} ev {stats['explained_variance']:.3f} "
                + " ".join(f"{k.split('/')[1]}={v:.3f}" for k, v in stats.items() if k.endswith("win_rate")))

            pc = cfg["pool"]
            if pc["snapshot_every"] and update % pc["snapshot_every"] == 0:
                pool_ids.append(update)
                save_checkpoint(os.path.join(ck_dir, f"pool_{update}.pt"), model, cfg, {"update": update, "step": step})
                change = {update: {k: v.detach().cpu() for k, v in model.state_dict().items()}}
                while len(pool_ids) > pc["max_size"]:
                    change[pool_ids.pop(0)] = None
                workers.set_pool(change)

            last = step >= cfg["total_steps"]
            if last or update % cfg["checkpoint_every"] == 0:
                extra = {"update": update, "step": step, "pool_ids": pool_ids}
                path = os.path.join(ck_dir, f"ckpt_{update:05d}.pt")
                save_checkpoint(path, model, cfg, extra)
                save_checkpoint(latest, model, cfg, extra, optimizer=opt)
                ec = cfg["eval"]
                if ec["every"] and (last or update % ec["every"] == 0):
                    _evaluate(path, ec, cfg, run_dir, update, step, logger, log)
    finally:
        workers.close()
        logger.close()
    return run_dir


def _evaluate(path, ec, cfg, run_dir, update, step, logger, log):
    from .evaluation import evaluate
    te = time.time()
    res = evaluate(path, ec["opponents"], ec["games"], ec["seed"], cfg["env"].get("num_players", 4),
                   ec["max_turns"], ec["deterministic"], ec["workers"])
    res.update({"step": step, "update": update, "time": time.time(), "eval_s": time.time() - te})
    with open(os.path.join(run_dir, "eval.jsonl"), "a") as f:
        f.write(json.dumps(res) + "\n")
    flat = {}
    for o, r in res["per_opponent"].items():
        flat[f"eval/{o}/win_rate"] = r["win_rate"]
        flat[f"eval/{o}/ci_lo"] = r["ci95"][0]
        flat[f"eval/{o}/ci_hi"] = r["ci95"][1]
        flat[f"eval/{o}/avg_vp"] = r["avg_vp"]
        log(f"  eval vs {o}: {r['wins']}/{r['games']} = {r['win_rate']:.3f} "
            f"CI95 [{r['ci95'][0]:.3f},{r['ci95'][1]:.3f}] vp {r['avg_vp']:.2f}")
    logger.log(step, flat)
