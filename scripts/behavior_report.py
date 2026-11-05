"""Behavior profile of agents, each seated against 3 copies of an opponent.

    python -m scripts.behavior_report --agents random basic "rl:checkpoint=/abs/ckpt_00200.pt" \
        --vs basic --games 100 --workers 2
    python -m scripts.behavior_report --watch /abs/runs/<run_id>   # profile each new checkpoint

For every agent, plays ``--games`` seat-rotated games (catan.eval.tournament
seeds/lineups, held-out seed range), re-simulates each game and extracts
per-seat behavior (catan.analytics), then aggregates the focal agent's seats
and the opponents' seats. Writes results/behavior/<id>.json.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import sys
import time
from multiprocessing import get_context

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from catan.analytics import aggregate_rows, analyze_game  # noqa: E402
from catan.engine import GameConfig  # noqa: E402
from catan.eval.stats import mean_ci, wilson  # noqa: E402
from catan.eval.tournament import (_safe_play, aggregate, heldout_seeds, lineups,  # noqa: E402
                                   parse_spec)


def label(spec: str) -> str:
    name, kw = parse_spec(spec)
    if "checkpoint" in kw:
        name += "_" + os.path.splitext(os.path.basename(str(kw["checkpoint"])))[0]
    extra = [f"{k}={v}" for k, v in kw.items() if k != "checkpoint"]
    return name + (":" + ",".join(extra) if extra else "")


def _work(task):
    rec = _safe_play(task)
    if "error" in rec:
        return rec, None
    rp = rec.pop("replay")
    return rec, analyze_game(rp["seed"], rp["config"], rp["actions"])


def profile(spec: str, vs: str, games: int, seed: int, workers: int, max_turns: int) -> dict:
    cfg = GameConfig(num_players=4, max_turns=max_turns)
    lus = lineups([spec, vs, vs, vs], 4, "cyclic")
    seeds = heldout_seeds(seed, max(1, math.ceil(games / len(lus))))
    tasks = [{"seed": s, "rotation": ri, "lineup": lu, "config": cfg.to_dict(), "save_replay": True}
             for s in seeds for ri, lu in enumerate(lus)][:games]
    t0 = time.time()
    if workers > 1:
        with get_context("fork").Pool(workers) as pool:
            results = pool.map(_work, tasks, chunksize=1)
    else:
        results = [_work(t) for t in tasks]
    ok = [(r, rows) for r, rows in results if rows is not None]
    if not ok:
        raise RuntimeError(results[0][0]["error"])
    records = [r for r, _ in ok]
    focal = [row for r, rows in ok for row in rows if r["lineup"][row["seat"]] == spec]
    opp = [row for r, rows in ok for row in rows if r["lineup"][row["seat"]] != spec]
    wins = sum(r["won"] for r in focal)
    labels = list(dict.fromkeys([spec, vs]))
    tour = aggregate(records, labels)["per_agent"][spec]
    return {
        "agent": spec, "label": label(spec), "vs": vs,
        "checkpoint_index": _ckpt_index(str(parse_spec(spec)[1].get("checkpoint", ""))), "games": len(records),
        "failed_games": len(results) - len(ok), "elapsed_sec": round(time.time() - t0, 1),
        # per focal seat-game, for paired comparisons between agents on the same seeds
        "game_wins": [[r["seed"], r["rotation"], int(r["winner"] >= 0 and r["lineup"][r["winner"]] == spec)]
                      for r in records],
        # per focal seat, so "basic vs 3 basic" gives the per-seat rate (about 25%)
        "outcome": {"seat_games": len(focal), "wins": wins, "win_rate": wins / len(focal),
                    "ci95": list(wilson(wins, len(focal))), "avg_vp": mean_ci([r["vp"] for r in focal]),
                    "truncation_rate": sum(r["truncated"] for r in records) / len(records),
                    "illegal_actions": tour["illegal_actions"], "sec_per_move": tour["sec_per_move"]},
        "behavior": aggregate_rows(focal),
        "opponents_behavior": aggregate_rows(opp) if opp else None,
    }


def _report(p):
    o, m = p["outcome"], p["behavior"]["metrics"]
    print(f"{p['label']:<22} win {o['win_rate']:.1%} [{o['ci95'][0]:.1%},{o['ci95'][1]:.1%}] "
          f"vp {o['avg_vp']['mean']:.2f} open {m['open_pips_total']['mean']:.1f} "
          f"cities {m['cities_built']['mean']:.2f} setl {m['settlements_built']['mean']:.2f} "
          f"selfblock {p['behavior']['ratios']['robber_self_block_rate']['rate']} ({p['elapsed_sec']}s)",
          flush=True)


def _write(res, path):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(res, f, separators=(",", ":"))
    os.replace(tmp, path)  # atomic: the server never reads a half-written file


def _ckpt_index(path: str):
    m = re.search(r"(\d+)\.pt$", path)
    return int(m.group(1)) if m else None


def watch(run_dir: str, a) -> None:
    """Profile every new checkpoint of ``run_dir`` vs 3x ``--vs``; one result file per run."""
    run_dir = os.path.abspath(run_dir)
    run_id = os.path.basename(run_dir.rstrip("/"))
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, f"watch-{run_id}-vs-{a.vs}.json")
    res = json.load(open(path)) if os.path.exists(path) else {
        "id": f"watch-{run_id}-vs-{a.vs}", "watch_run": run_dir, "vs": a.vs, "games_per_agent": a.games,
        "seed": a.seed, "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "agents": {}}
    done = {p["agent"] for p in res["agents"].values()}
    while True:
        ckpts = sorted(glob.glob(os.path.join(run_dir, "checkpoints", "*.pt")), key=lambda c: _ckpt_index(c) or -1)
        # skip aliases (latest.pt) and files that may still be being written
        todo = [c for c in ckpts if _ckpt_index(c) is not None and f"rl:checkpoint={c}" not in done
                and time.time() - os.path.getmtime(c) > a.settle]
        for c in todo:
            spec = f"rl:checkpoint={c}"
            p = profile(spec, a.vs, a.games, a.seed, a.workers, a.max_turns)
            res["agents"][p["label"]] = p
            done.add(spec)
            res["updated"] = time.strftime("%Y-%m-%dT%H:%M:%S")
            _write(res, path)
            _report(p)
        if a.once:
            return
        time.sleep(a.poll)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--agents", nargs="+")
    ap.add_argument("--labels", nargs="+", help="display labels, one per --agents spec (e.g. registry ids)")
    ap.add_argument("--watch", metavar="RUN_DIR", help="profile each new checkpoint of a training run")
    ap.add_argument("--once", action="store_true", help="with --watch: process existing checkpoints and exit")
    ap.add_argument("--poll", type=float, default=120, help="seconds between checkpoint scans")
    ap.add_argument("--settle", type=float, default=30, help="ignore checkpoints modified in the last N seconds")
    ap.add_argument("--vs", default="basic")
    ap.add_argument("--games", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--max-turns", type=int, default=500)
    ap.add_argument("--out", default="results/behavior")
    ap.add_argument("--id", default=None)
    a = ap.parse_args(argv)
    if not (a.agents or a.watch):
        ap.error("give --agents or --watch")
    for s in (a.agents or []) + [a.vs]:
        parse_spec(s)
    try:  # one intra-op thread per process keeps the CPU budget at --workers cores
        import torch
        torch.set_num_threads(1)
    except ImportError:
        pass
    if a.watch:
        return watch(a.watch, a)
    res = {"id": a.id or time.strftime("%Y%m%d-%H%M%S") + f"-vs-{a.vs}", "vs": a.vs, "games_per_agent": a.games,
           "seed": a.seed, "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "agents": {}}
    if a.labels and len(a.labels) != len(a.agents):
        ap.error("--labels needs one label per --agents spec")
    for i, spec in enumerate(a.agents):
        p = profile(spec, a.vs, a.games, a.seed, a.workers, a.max_turns)
        if a.labels:
            p["label"] = a.labels[i]
        res["agents"][p["label"]] = p
        _report(p)
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, res["id"] + ".json")
    _write(res, path)
    print("wrote", path)


if __name__ == "__main__":
    main()
