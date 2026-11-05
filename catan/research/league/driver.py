"""League learner loop (one process per role). See docs/research/LEAGUE.md.

    python -m scripts.league_run --config configs/league_r1.yaml --role main
    python -m scripts.league_run --config configs/league_r1.yaml --role main_exploiter
    python -m scripts.league_run --config configs/league_r1.yaml --role league_exploiter

Each role trains catan.rl.ppo.train (D5 GNN via catan.research.arch.train.install,
in-process collection = 1 process) in resumable segments. Before every segment
it reads the shared league state, builds an opponent mixture, and afterwards
records per-opponent training win rates (decayed), snapshots itself into the
pool and applies the reset rule. Mixtures (AlphaStar-style, 4-player 1v3):

  main:             sp_frac  vs its own latest snapshot, champ_frac in the gate's
                    mixed table (champion + 2x intermediate), rest PFSP-hard over
                    all members
  main_exploiter:   the current main snapshot only; resets to the init policy when
                    it beats main (pairwise q > reset_q) or after max_segments
  league_exploiter: PFSP-hard over all members; resets like the main exploiter
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import copy
import json
import random
import shutil
import time

import torch

from . import pfsp, pool
from .agents import install_env, register_member

ROLES = ("main", "main_exploiter", "league_exploiter")


def _members(st):
    for mid, info in st["members"].items():
        register_member(mid, info)
    return st["members"]


def mixture(st, role, cfg, rng):
    """Opponent list for catan.rl.ppo (type agent, name, weight)."""
    lc = cfg["league"]
    members = _members(st)
    me_latest = st["roles"].get(role, {}).get("latest")
    main_latest = st["roles"].get("main", {}).get("latest") or lc["init_member"]
    others = [m for m in members if m != me_latest]
    rates = {m: pool.win_rate(st, role, m) for m in others}
    out = []
    if role == "main_exploiter":
        return [{"type": "agent", "name": main_latest, "weight": 1.0}], {main_latest: 1.0}
    dist = pfsp.distribution(rates, lc.get("pfsp", "hard"))
    rest = 1.0
    if role == "main":
        if me_latest:
            out.append({"type": "agent", "name": me_latest, "weight": lc["sp_frac"]})
            rest -= lc["sp_frac"]
        champ = lc.get("champion_member")
        if champ and lc.get("champ_frac"):
            out.append({"type": "agent", "name": [champ, "intermediate", "intermediate"], "weight": lc["champ_frac"]})
            rest -= lc["champ_frac"]
    for m, w in dist.items():
        out.append({"type": "agent", "name": m, "weight": rest * w})
    return out, {o["name"] if isinstance(o["name"], str) else "+".join(o["name"]): o["weight"] for o in out}


def _segment_results(run_dir, since_update):
    """mode -> (wins, games) from metrics.jsonl updates after ``since_update``."""
    res = {}
    path = os.path.join(run_dir, "metrics.jsonl")
    if not os.path.exists(path):
        return res
    for line in open(path):
        d = json.loads(line)
        if d.get("update", 0) <= since_update:
            continue
        for k, v in d.items():
            if k.startswith("train/") and k.endswith("/win_rate"):
                mode = k[len("train/"):-len("/win_rate")]
                n = d.get(f"train/{mode}/games", 0)
                w0, n0 = res.get(mode, (0.0, 0))
                res[mode] = (w0 + v * n, n0 + n)
    return res


def _last(run_dir, key="update"):
    p = os.path.join(run_dir, "checkpoints", "latest.pt")
    if not os.path.exists(p):
        return 0
    return int(torch.load(p, map_location="cpu", weights_only=True)[key])


def _last_update(run_dir):
    return _last(run_dir, "update")


def run(config_path, role, max_segments=None, log=print):
    from catan.research.arch.train import install
    from catan.rl import ppo as rppo

    cfg0 = rppo.load_config(config_path)
    lc = cfg0["league"]
    rc = lc["roles"][role]
    league_dir = lc["dir"]
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    rng = random.Random(lc["seed"] + ROLES.index(role))
    with pool.locked(league_dir) as st:
        for mid, spec in lc["agents"].items():
            pool.add_member(st, mid, kind="agent", spec=spec, role="fixed")
        pool.add_member(st, lc["init_member"], kind="snapshot", path=lc["init_checkpoint"], role="init")
        st["roles"].setdefault(role, {"generation": 0, "segments": 0, "latest": None})
    install_env()
    base = copy.deepcopy(cfg0)
    base.pop("league")
    base["init_checkpoint"] = lc["init_checkpoint"]
    # anchor to the fixed init policy via catan.rl.ppo's kl_ref_coef (r1 used arch.klppo before it existed)
    base["ppo"] = dict(base["ppo"], kl_bc_coef=0, kl_ref_coef=float(rc.get("kl_coef", 0.0)),
                       kl_ref_checkpoint=lc["init_checkpoint"])
    install(base)
    seg_steps = int(lc["segment_updates"]) * int(base["rollout_steps"])
    done = 0
    while max_segments is None or done < max_segments:
        st = pool.read(league_dir)
        rs = st["roles"][role]
        gen = rs["generation"]
        run_dir = os.path.join("runs", f"{lc['name']}_{role}_g{gen}")
        resume = os.path.exists(os.path.join(run_dir, "checkpoints", "latest.pt"))
        since = _last_update(run_dir)
        opps, weights = mixture(st, role, cfg0, rng)
        cfg = copy.deepcopy(base)
        cfg["opponents"] = opps
        cfg["run_name"] = os.path.basename(run_dir)
        cfg["seed"] = int(lc["seed"]) + 1000 * ROLES.index(role) + 100 * gen
        cfg["total_steps"] = _last(run_dir, "step") + seg_steps if resume else seg_steps
        log(f"[{role} g{gen} seg{rs['segments']}] opponents " +
            ", ".join(f"{k}={v:.2f}" for k, v in sorted(weights.items(), key=lambda x: -x[1])))
        t0 = time.time()
        rppo.train(cfg, run_dir, resume=resume, log=lambda s: log(f"[{role}] {s}"))
        res = _segment_results(run_dir, since)
        upd = _last_update(run_dir)
        sid = f"lg_{role}_g{gen}_u{upd}"
        snap = os.path.join(league_dir, "snapshots", f"{sid}.pt")
        os.makedirs(os.path.dirname(snap), exist_ok=True)
        shutil.copy(os.path.join(run_dir, "checkpoints", "latest.pt"), snap)
        with pool.locked(league_dir) as st:
            pool.record(st, role, res, float(lc["decay"]))
            pool.add_member(st, sid, kind="snapshot", path=snap, role=role, generation=gen, update=upd)
            r = st["roles"][role]
            r["segments"] += 1
            r["latest"] = sid
            # cap the pool: drop the oldest exploiter snapshots first
            snaps = [m for m, i in st["members"].items() if i["kind"] == "snapshot" and i["role"] != "init"]
            excess = len(snaps) - int(lc["max_snapshots"])
            for m in sorted(snaps, key=lambda m: (st["members"][m]["role"] == "main", st["members"][m]["created"])):
                if excess <= 0:
                    break
                if m not in {x["latest"] for x in st["roles"].values()}:
                    st["members"].pop(m)
                    excess -= 1
            reset = False
            if role != "main":
                target = st["roles"].get("main", {}).get("latest") if role == "main_exploiter" else None
                if target:
                    q = pfsp.pairwise(pool.win_rate(st, role, target))
                else:
                    qs = [pfsp.pairwise(pool.win_rate(st, role, m)) for m in st["members"] if m != sid]
                    q = sum(qs) / max(1, len(qs))
                seg_in_gen = r.get("seg_in_gen", 0) + 1
                r["seg_in_gen"] = seg_in_gen
                if q > float(rc.get("reset_q", 0.7)) or seg_in_gen >= int(rc.get("max_segments", 6)):
                    r["generation"] += 1
                    r["seg_in_gen"] = 0
                    reset = True
            summary = {"role": role, "generation": gen, "update": upd, "snapshot": sid,
                       "segment_s": round(time.time() - t0, 1), "results": res, "weights": weights,
                       "reset": reset, "time": time.time()}
            with open(os.path.join(league_dir, "segments.jsonl"), "a") as f:
                f.write(json.dumps(summary) + "\n")
        log(f"[{role}] segment done: {sid} " + " ".join(f"{m}={w / max(n, 1):.2f}/{n}" for m, (w, n) in res.items())
            + (" RESET" if reset else ""))
        done += 1
