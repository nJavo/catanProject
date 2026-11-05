"""Train NegoNet: behaviour cloning from the trader heuristic, then PPO on the
game outcome with a KL penalty toward the BC prior (piKL-style).

    python -m scripts.nego_train --config configs/nego_r1.yaml

Each game seats one or two learners (rotating seats) against opponents drawn
from ``opponents``. Only the learners' negotiation decisions are trained; all
other play is the intermediate heuristic. Reward per learner seat:
``win + vp_coef * final_vp`` (no shaping on trade volume). Training seeds are
``seed_base + i`` (>= 1e7). Selection evals use held-out block ``select_block``.
"""
from __future__ import annotations

import io
import json
import math
import os
import random
import subprocess
import time
from multiprocessing import get_context

import torch
import yaml

from catan.agents.registry import make_agent
from catan.engine import Game, GameConfig
from catan.eval.tournament import _safe_play, aggregate, heldout_seeds, lineups

from catan.bots import common as C
from catan.bots.intermediate import evaluate

from .agent import NegoAgent
from .model import NegoNet, batch, save

DEFAULTS = {
    "name": "nego_r1", "hidden": 96, "workers": 2, "seed_base": 10_000_000,  # 2 workers + learner = 3 processes
    "bc_games": 1500, "bc_epochs": 6, "bc_lr": 1e-3,
    "iters": 60, "games_per_iter": 240, "epochs": 4, "minibatch": 512, "lr": 3e-4,
    "clip": 0.2, "kl_coef": 0.05, "ent_coef": 0.003, "vf_coef": 0.5, "vp_coef": 0.05,
    "learners_per_game": 1, "opponents": ["intermediate", "intermediate_trader", "basic", "learner"],
    "max_turns": 500, "eval_every": 10, "eval_games": 100, "select_block": 59,
    "eval_opponents": ["intermediate", "intermediate_trader"],
}

_W = {"key": None, "net": None}


def _worker_net(blob, key):
    if _W["key"] != key:
        net = NegoNet(_W.get("hidden", 96))
        if blob is not None:
            net.load_state_dict(torch.load(io.BytesIO(blob), weights_only=True))
        net.eval()
        _W["key"], _W["net"] = key, net
    return _W["net"]


def _one_thread():
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:  # already set in this process
        pass


def _init_worker(hidden):
    _one_thread()
    _W["hidden"] = hidden


def play_episode(task):
    """One game. Returns learner trajectories with rewards and the outcome."""
    seed, cfg = task["seed"], task["cfg"]
    rng = random.Random(seed)
    net = _worker_net(task["blob"], task["key"]) if task["blob"] is not None else None
    n = 4
    seats = list(range(n))
    rng.shuffle(seats)
    learn = set(seats[:cfg["learners_per_game"]])
    agents, labels = [], []
    for i in range(n):
        if i in learn:
            a = NegoAgent(seed=seed + i, net=net, sample=net is not None, record=True)
            lab = "learner"
        else:
            lab = rng.choice(cfg["opponents"])
            a = NegoAgent(seed=seed + i, net=net, sample=net is not None) if lab == "learner" \
                else make_agent(lab, seed=seed + i)
        a.reset(i, seed + i)
        agents.append(a)
        labels.append(lab)
    g = Game(config=GameConfig(max_turns=cfg["max_turns"]), seed=seed)
    shape = cfg.get("shape_coef", 0.0)
    pending = {i: [] for i in learn}  # decisions awaiting the trade's resolution
    while not g.is_over:
        p = g.actor
        legal = g.legal_actions()
        n_before = len(agents[p].trajectory) if p in learn else 0
        try:
            act = agents[p].act(g, legal)
            if not g.is_legal(act):
                raise ValueError(act)
        except Exception:
            act = rng.choice(legal)
        if shape and p in learn and len(agents[p].trajectory) > n_before:
            # privileged, training-only: true-state relative evaluation before the decision
            pending[p].append((agents[p].trajectory[-1], _rel_eval(g, p)))
        evs = g.step(act)
        if shape and g.phase not in ("TRADE_RESPONSE", "TRADE_CONFIRM"):
            for i in learn:
                if pending[i]:
                    after = _rel_eval(g, i)
                    for d, before in pending[i]:
                        d["delta"] = after - before
                    pending[i] = []
        for i, a in enumerate(agents):
            if hasattr(a, "observe"):
                a.observe(g, [e for e in evs if "visible_to" not in e or i in e["visible_to"]])
    out = []
    for i in learn:
        R = cfg.get("outcome_coef", 1.0) * (float(g.winner == i) + cfg["vp_coef"] * g.victory_points(i))
        for d in agents[i].trajectory:
            d["R"] = R + shape * d.get("delta", 0.0)
        out.append({"seat": i, "win": g.winner == i, "vp": g.victory_points(i), "traj": agents[i].trajectory,
                    "opps": [labels[j] for j in range(n) if j != i]})
    return {"seed": seed, "learners": out, "turns": g.turn, "truncated": g.truncated}


def _rel_eval(g, me):
    bi = C.board_info(g.layout)
    mine = evaluate(g, me, bi)
    return mine - max(evaluate(g, q, bi) for q in range(g.n) if q != me)


def _blob(net):
    b = io.BytesIO()
    torch.save(net.state_dict(), b)
    return b.getvalue()


def _collect(pool, net, key, cfg, seeds):
    blob = _blob(net) if net is not None else None
    tasks = [{"seed": s, "cfg": cfg, "blob": blob, "key": key} for s in seeds]
    return list(pool.imap_unordered(play_episode, tasks, chunksize=4))


def _decisions(eps):
    return [d for e in eps for l in e["learners"] for d in l["traj"]]


def _train_stats(eps):
    ls = [l for e in eps for l in e["learners"]]
    st = {"episodes": len(eps), "train/win_rate": sum(l["win"] for l in ls) / max(1, len(ls)),
          "train/learner_vp": sum(l["vp"] for l in ls) / max(1, len(ls)),
          "train/turns": sum(e["turns"] for e in eps) / max(1, len(eps)),
          "train/truncated_rate": sum(e["truncated"] for e in eps) / max(1, len(eps))}
    ds = [d.get("delta") for l in ls for d in l["traj"] if "delta" in d]
    if ds:
        st["shape/nonzero_frac"] = sum(1 for x in ds if abs(x) > 1e-9) / len(ds)
        st["shape/mean_abs"] = sum(abs(x) for x in ds) / len(ds)
        st["shape/mean"] = sum(ds) / len(ds)
    for opp in set(o for l in ls for o in l["opps"]):
        sub = [l for l in ls if opp in l["opps"]]
        st[f"train/with_{opp}/win_rate"] = sum(l["win"] for l in sub) / len(sub)
    return st


def bc_update(net, data, cfg, log):
    opt = torch.optim.Adam(net.parameters(), lr=cfg["bc_lr"])
    for ep in range(cfg["bc_epochs"]):
        random.Random(ep).shuffle(data)
        tot = acc = 0.0
        for i in range(0, len(data), cfg["minibatch"]):
            mb = data[i:i + cfg["minibatch"]]
            logits, v = net(*batch(mb))
            tgt = torch.tensor([d["teacher"] for d in mb])
            R = torch.tensor([d["R"] for d in mb])
            loss = torch.nn.functional.cross_entropy(logits, tgt) + cfg["vf_coef"] * ((v - R) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            tot += loss.item() * len(mb)
            acc += float((logits.argmax(-1) == tgt).float().sum())
        log({"phase": "bc", "bc_epoch": ep, "bc_loss": tot / len(data), "bc_acc": acc / len(data)})


def ppo_update(net, prior, data, cfg, opt):
    adv_all = None
    with torch.no_grad():
        vs = []
        for i in range(0, len(data), 2048):
            _, v = net(*batch(data[i:i + 2048]))
            vs.append(v)
        V = torch.cat(vs)
        R = torch.tensor([d["R"] for d in data])
        adv_all = R - V
        adv_all = (adv_all - adv_all.mean()) / (adv_all.std() + 1e-8)
    idx = list(range(len(data)))
    stats = {"policy_loss": 0.0, "value_loss": 0.0, "kl_prior": 0.0, "entropy": 0.0, "clipfrac": 0.0,
             "approx_kl": 0.0}
    nb = 0
    for ep in range(cfg["epochs"]):
        random.Random(1000 + ep).shuffle(idx)
        for i in range(0, len(idx), cfg["minibatch"]):
            ids = idx[i:i + cfg["minibatch"]]
            mb = [data[j] for j in ids]
            b = batch(mb)
            logits, v = net(*b)
            with torch.no_grad():
                plog, _ = prior(*b)
            m = b[4]
            lp = torch.log_softmax(logits, -1)
            a = torch.tensor([d["a"] for d in mb])
            old = torch.tensor([d["logp"] for d in mb])
            new = lp.gather(1, a.unsqueeze(1)).squeeze(1)
            ratio = torch.exp(new - old)
            A = adv_all[ids]
            pg = -torch.min(ratio * A, ratio.clamp(1 - cfg["clip"], 1 + cfg["clip"]) * A).mean()
            p = lp.exp()
            plp = torch.log_softmax(plog, -1)
            lpz = torch.where(m, lp, torch.zeros_like(lp))
            plpz = torch.where(m, plp, torch.zeros_like(plp))
            kl = (p * (lpz - plpz)).sum(-1).mean()
            ent = -(p * lpz).sum(-1).mean()
            Rm = torch.tensor([d["R"] for d in mb])
            vl = ((v - Rm) ** 2).mean()
            loss = pg + cfg["vf_coef"] * vl + cfg["kl_coef"] * kl - cfg["ent_coef"] * ent
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            stats["policy_loss"] += float(pg)
            stats["value_loss"] += float(vl)
            stats["kl_prior"] += float(kl)
            stats["entropy"] += float(ent)
            stats["clipfrac"] += float(((ratio - 1).abs() > cfg["clip"]).float().mean())
            stats["approx_kl"] += float((old - new).mean())
            nb += 1
    return {k: v / max(1, nb) for k, v in stats.items()}


def select_eval(pool, ckpt, cfg):
    """Greedy learner vs 3x each eval opponent on the selection block, run in
    the training pool (keeps the process budget)."""
    out = {}
    gc = GameConfig(num_players=4, max_turns=cfg["max_turns"]).to_dict()
    for opp in cfg["eval_opponents"]:
        spec = f"nego:checkpoint={ckpt}"
        lus = lineups([spec, opp, opp, opp], 4, "cyclic")
        seeds = heldout_seeds(cfg["select_block"], math.ceil(cfg["eval_games"] / len(lus)))
        tasks = [{"seed": sd, "rotation": ri, "lineup": lu, "config": gc, "save_replay": False}
                 for sd in seeds for ri, lu in enumerate(lus)][:cfg["eval_games"]]
        recs = [r for r in pool.imap(_safe_play, tasks, chunksize=2) if "error" not in r]
        s = aggregate(recs, [spec, opp])["per_agent"][spec]
        out[opp] = {"games": s["games"], "wins": s["wins"], "win_rate": s["win_rate"], "ci95": s["ci95"],
                    "avg_vp": s["avg_vp"]["mean"], "trade": s["trade"]}
    return out


def _git():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def main(config_path):
    cfg = dict(DEFAULTS, **(yaml.safe_load(open(config_path)) or {}))
    _one_thread()
    torch.manual_seed(cfg["seed_base"])
    run = os.path.join("runs", cfg["name"])
    os.makedirs(os.path.join(run, "checkpoints"), exist_ok=True)
    with open(os.path.join(run, "config.yaml"), "w") as f:
        yaml.safe_dump(cfg, f)
    import platform
    meta = {"run_id": cfg["name"], "git_commit": _git(), "seeds": {"train_base": cfg["seed_base"],
            "select_block": cfg["select_block"]}, "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "hardware": {"platform": platform.platform(), "cpu_count": os.cpu_count(), "torch": torch.__version__,
                         "workers": cfg["workers"]}, "kind": "nego"}
    json.dump(meta, open(os.path.join(run, "meta.json"), "w"), indent=1)
    mlog = open(os.path.join(run, "metrics.jsonl"), "a")
    elog = open(os.path.join(run, "eval.jsonl"), "a")
    t0 = time.time()
    step = 0

    def log(d):
        d = {"step": step, "time": time.time(), **d}
        mlog.write(json.dumps(d) + "\n")
        mlog.flush()
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in d.items()
                          if not k.startswith("train/with")}), flush=True)

    net = NegoNet(cfg["hidden"])
    seed_i = [cfg["seed_base"]]

    def seeds(k):
        s = list(range(seed_i[0], seed_i[0] + k))
        seed_i[0] += k
        return s

    ctx = get_context("fork")
    with ctx.Pool(cfg["workers"], initializer=_init_worker, initargs=(cfg["hidden"],)) as pool:
        if cfg.get("init_checkpoint"):
            from .model import load
            net, _ = load(cfg["init_checkpoint"])
            net.train()
        elif cfg["bc_games"] > 0:
            eps = _collect(pool, None, "teacher", cfg, seeds(cfg["bc_games"]))
            data = _decisions(eps)
            step += len(data)
            log({"phase": "bc_collect", "decisions": len(data), **_train_stats(eps)})
            bc_update(net, data, cfg, log)
        prior = NegoNet(cfg["hidden"])
        prior.load_state_dict(net.state_dict())
        prior.eval()
        ck = os.path.join(run, "checkpoints", "ckpt_00000.pt")
        save(net, ck, {"update": 0, "step": step, "config": cfg["name"]})
        if cfg["eval_every"]:
            e = select_eval(pool, ck, cfg)
            elog.write(json.dumps({"checkpoint": ck, "agent": "nego", "update": 0, "step": step,
                                   "per_opponent": e, "time": time.time()}) + "\n")
            elog.flush()
            print("eval", json.dumps({k: (v["win_rate"], v["ci95"]) for k, v in e.items()}), flush=True)
        opt = torch.optim.Adam(net.parameters(), lr=cfg["lr"])
        for it in range(1, cfg["iters"] + 1):
            net.eval()
            eps = _collect(pool, net, f"it{it}", cfg, seeds(cfg["games_per_iter"]))
            data = _decisions(eps)
            step += len(data)
            net.train()
            st = ppo_update(net, prior, data, cfg, opt) if data else {}
            log({"update": it, "decisions": len(data), **st, **_train_stats(eps), "elapsed": time.time() - t0})
            ck = os.path.join(run, "checkpoints", f"ckpt_{it:05d}.pt")
            save(net, ck, {"update": it, "step": step, "config": cfg["name"]})
            if cfg["eval_every"] and it % cfg["eval_every"] == 0:
                net.eval()
                e = select_eval(pool, ck, cfg)
                elog.write(json.dumps({"checkpoint": ck, "agent": "nego", "update": it, "step": step,
                                       "per_opponent": e, "time": time.time()}) + "\n")
                elog.flush()
                print("eval", it, json.dumps({k: (v["win_rate"], v["ci95"]) for k, v in e.items()}), flush=True)
