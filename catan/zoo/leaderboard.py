"""Global leaderboard: one Bradley-Terry fit (``catan.eval.ratings``) over
every game in ``results/tournaments`` (plain tournaments and ladders),
restricted to registry entries. Seats held by unregistered agents are dropped
from each game's ranking; games left with fewer than two distinct entries
don't count. No entry is pinned: every fit (point and each bootstrap resample)
is mean-centred, so the average of all entries is 1500 and every entry,
random included, gets a bootstrap CI.
"""
from __future__ import annotations

import json
import random
import time
from pathlib import Path

from catan.eval.ratings import bradley_terry, pairwise_outcomes
from catan.eval.tournament import copy_spec, rank_scores

from .registry import spec_to_id

MEAN_ELO = 1500.0


def _games(t: dict):
    """Yield (lineup, rank scores) for every game in a tournament or ladder JSON."""
    if t.get("kind") == "ladder":
        for c in t.get("checkpoints", []):
            for r in c.get("results", {}).values():
                for rot, w, *vp in r.get("outcomes", []):
                    yield r["lineups"][rot], rank_scores({"winner": w, "vp": vp})
    else:
        for g in t.get("game_summaries") or []:
            yield g["lineup"], rank_scores(g)


def _gate_owner(filename: str, reg: dict) -> str | None:
    """``gate_<entry id>_<tag>.json`` (catan/zoo/gate.py) -> entry id."""
    for i in sorted((e["id"] for e in reg["entries"]), key=len, reverse=True):
        if filename.startswith(f"gate_{i}_"):
            return i
    return None


def collect(results_dir, reg: dict):
    """Rankings restricted to registry entries. A lineup spec counts for an entry when it equals the
    entry's ``spec`` or one of its ``aliases``, or when it is the checkpoint seat of that entry's own
    gate run: an entry's spec can change after its gate (e.g. the checkpoint was copied to models/),
    and its games must still count (DS2-11)."""
    ids = spec_to_id(reg)
    for e in reg["entries"]:
        for a in e.get("aliases") or []:
            ids.setdefault(a, e["id"])
    for s, i in list(ids.items()):  # the gate's same-seat champion copy counts as the champion
        ids.setdefault(copy_spec(s), i)
    rankings, sources = [], []
    for p in sorted((Path(results_dir) / "tournaments").glob("*.json")):
        try:
            t = json.loads(p.read_text())
        except (OSError, ValueError):
            continue
        owner = _gate_owner(p.name, reg)
        local = dict(ids)
        if owner is not None:
            for a in t.get("agents") or []:
                if a not in local and ":checkpoint=" in a:
                    local[a] = owner
        before = len(rankings)
        for lineup, sc in _games(t):
            rk = [(local[a], s) for a, s in zip(lineup, sc) if a in local]
            if len({a for a, _ in rk}) >= 2:
                rankings.append(rk)
        if len(rankings) > before:
            sources.append({"file": p.name, "games": len(rankings) - before})
    return rankings, sources


def _fit(rankings, labels, iters=500):
    r = bradley_terry(pairwise_outcomes(rankings), labels, iters=iters)
    shift = MEAN_ELO - sum(r.values()) / len(r)  # bradley_terry already centres; keep it explicit
    return {k: v + shift for k, v in r.items()}


def build(reg: dict, results_dir="results", bootstrap: int = 200, seed: int = 0) -> dict:
    rankings, sources = collect(results_dir, reg)
    labels = sorted({a for g in rankings for a, _ in g})
    games = {x: sum(1 for g in rankings if any(a == x for a, _ in g)) for x in labels}
    point = _fit(rankings, labels) if len(labels) >= 2 else {x: MEAN_ELO for x in labels}
    rng = random.Random(seed)
    samples = {x: [] for x in labels}
    for _ in range(bootstrap if len(labels) >= 2 else 0):
        rs = [rankings[rng.randrange(len(rankings))] for _ in rankings]
        for x, v in _fit(rs, labels, iters=200).items():
            samples[x].append(v)
    # D34: Plackett-Luce over the full finishing order, shown next to BT (an option, not the ranking key)
    from catan.eval.plackett_luce import ratings as pl_ratings
    plr = pl_ratings(rankings, labels, bootstrap=bootstrap, seed=seed) if len(labels) >= 2 else {}
    by_id = {e["id"]: e for e in reg["entries"]}
    entries = []
    for x in labels:
        s = sorted(samples[x]) or [point[x]]
        e = by_id[x]
        entries.append({"id": x, "name": e["name"], "kind": e["kind"], "status": e["status"],
                        "benchmark": e.get("anchor", False), "spec": e["spec"],
                        "elo": round(point[x], 1),
                        "ci95": [round(s[int(0.025 * len(s))], 1), round(s[min(len(s) - 1, int(0.975 * len(s)))], 1)],
                        "games": games[x],
                        "pl_elo": plr[x]["elo"] if x in plr else None, "pl_ci95": plr[x]["ci95"] if x in plr else None})
    entries.sort(key=lambda e: -e["elo"])
    for e in reg["entries"]:  # registered but never played: listed as unrated rather than silently missing
        if e["id"] not in games:
            entries.append({"id": e["id"], "name": e["name"], "kind": e["kind"], "status": e["status"],
                            "benchmark": e.get("anchor", False), "spec": e["spec"],
                            "elo": None, "ci95": None, "games": 0, "pl_elo": None, "pl_ci95": None})
    return {"updated": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "method": "Bradley-Terry (MM) on pairwise finishing order (winner, then VP); "
                      f"bootstrap {bootstrap} over games; mean of all entries = {MEAN_ELO:.0f}",
            "pl_method": "Plackett-Luce (MM, Breslow ties) on the full finishing order; same bootstrap and mean "
                         "(catan.eval.plackett_luce)",
            "mean_elo": MEAN_ELO, "games": len(rankings),
            "sources": sources, "entries": entries}


def write(reg: dict, results_dir="results", **kw) -> dict:
    lb = build(reg, results_dir, **kw)
    p = Path(results_dir) / "leaderboard.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(lb, indent=1) + "\n")
    return lb
