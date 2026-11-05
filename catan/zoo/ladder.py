"""Checkpoint ladder: every Nth checkpoint of a run vs fixed baselines.

Each point is ``checkpoint vs 3x opponent`` on held-out seed block 20
(seats rotated), so every checkpoint and every run sees the same boards.
Results go to ``results/tournaments/ladder_<run>.json``; points already
evaluated with the same sha256 and enough games are reused, so re-running
(or ``--watch``) only plays new checkpoints.
"""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

import torch

from catan.eval.tournament import _git_commit, run_tournament

from catan.eval.seeds import LADDER_BLOCK

from .registry import ckpt_rel, sha256_file

BLOCK = LADDER_BLOCK  # 20
OPPONENTS = ("basic", "intermediate", "random")
SCHEMA = 1
_CKPT = re.compile(r"ckpt_(\d+)\.pt$")


def list_checkpoints(run_dir) -> list[Path]:
    """``ckpt_*.pt`` sorted by update number (``latest.pt`` is an alias, skipped)."""
    d = Path(run_dir) / "checkpoints"
    return sorted((p for p in d.glob("ckpt_*.pt") if _CKPT.search(p.name)),
                  key=lambda p: int(_CKPT.search(p.name).group(1)))


def select(ckpts: list, every: int) -> list:
    """Every ``every``-th checkpoint, counting back from the last one (always included)."""
    return ckpts[::-1][::max(1, every)][::-1]


def ladder_path(run_id: str, out="results") -> Path:
    return Path(out) / "tournaments" / f"ladder_{run_id}.json"


def point(spec: str, opp: str, games: int, workers: int, num_players: int = 4) -> dict:
    res = run_tournament([spec] + [opp] * (num_players - 1), games, seed=BLOCK, workers=workers, num_players=num_players, out_dir=None)
    s = res["per_agent"][spec]
    return {"opponent": opp, "games": s["games"], "wins": s["wins"], "win_rate": s["win_rate"],
            "ci95": s["ci95"], "avg_vp": s["avg_vp"]["mean"], "truncation_rate": res["global"]["truncation_rate"],
            "failed_games": len(res["failed_games"]), "illegal_actions": s["illegal_actions"],
            "seeds": [res["seeds"][0], res["seeds"][-1]], "lineups": res["lineups"],
            # compact per-game record [rotation, winner, vp...] so the leaderboard can refit ratings
            "outcomes": [[g["rotation"], g["winner"], *g["vp"]] for g in res["game_summaries"]],
            "elapsed_sec": res["elapsed_sec"]}


def run_ladder(run_dir, every: int = 1, games: int = 100, final_games: int = 200, workers: int = 3,
               opponents=OPPONENTS, out="results", root=".", num_players: int = 4, log=print) -> dict:
    torch.set_num_threads(1)  # forked workers inherit this; torch defaults to one thread per core
    run_dir = Path(run_dir)
    run_id = run_dir.name
    path = ladder_path(run_id, out)
    try:
        lad = json.loads(path.read_text())
    except (OSError, ValueError):
        lad = {}
    if lad.get("schema") != SCHEMA:
        lad = {"schema": SCHEMA, "kind": "ladder", "run": run_id, "checkpoints": []}
    lad.update(run_dir=ckpt_rel(str(run_dir), root), opponents=list(opponents), seed_block=BLOCK,
               num_players=num_players, rotation="cyclic",
               seed_set="heldout 1_000_000 + 20*100_000 + i (block 20, ladder only)")
    old = {c["checkpoint"]: c for c in lad["checkpoints"]}
    ckpts = list_checkpoints(run_dir)
    for p in select(ckpts, every):
        rel = ckpt_rel(str(p), root)
        sha = sha256_file(p)
        prev = old.get(rel)
        if not prev or prev.get("sha256") != sha:
            ck = torch.load(p, map_location="cpu", weights_only=True)
            prev = {"checkpoint": rel, "sha256": sha, "update": ck.get("update"), "step": ck.get("step"),
                    "spec": f"rl:checkpoint={rel}", "results": {}}
        need = final_games if p == ckpts[-1] else games
        for opp in opponents:
            r = prev["results"].get(opp)
            if r and r["games"] >= need:
                continue
            t0 = time.time()
            prev["results"][opp] = point(prev["spec"], opp, need, workers, num_players)
            r = prev["results"][opp]
            log(f"[{run_id}] {p.name} vs 3x{opp}: {r['win_rate']:.1%} "
                f"[{r['ci95'][0]:.1%},{r['ci95'][1]:.1%}] n={r['games']} ({time.time() - t0:.0f}s)")
            prev["evaluated"] = time.strftime("%Y-%m-%dT%H:%M:%S")
            # save after every point: a killed run loses at most one point
            old[rel] = prev
            lad["checkpoints"] = sorted(old.values(), key=lambda c: c["update"] or 0)
            _write(lad, path)
    lad["updated"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    lad["git_commit"] = _git_commit()
    _write(lad, path)
    return lad


def _write(lad: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(lad, indent=1))
    os.replace(tmp, path)


def summary_table(lad: dict) -> str:
    opps = lad["opponents"]
    lines = [f"{'checkpoint':<16}{'update':>7}{'step':>11}" + "".join(f"{'vs 3x' + o:>28}" for o in opps)]
    for c in lad["checkpoints"]:
        cells = []
        for o in opps:
            r = c["results"].get(o)
            cells.append(f"{r['win_rate']:.1%} [{r['ci95'][0]:.1%},{r['ci95'][1]:.1%}] n={r['games']}"
                         if r else "-")
        lines.append(f"{Path(c['checkpoint']).name:<16}{c['update'] or '':>7}{c['step'] or '':>11}"
                     + "".join(f"{x:>28}" for x in cells))
    return "\n".join(lines)
