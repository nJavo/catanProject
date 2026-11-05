"""Checkpoint ladder for a training run, then refresh the global leaderboard.

    python -m scripts.ladder --run runs/baseline_r1 --every 2 --games 100 --final-games 200 --workers 3
    python -m scripts.ladder --run runs/baseline_r1 --watch --interval 300 --workers 2

Writes results/tournaments/ladder_<run>.json and results/leaderboard.json.
"""
from __future__ import annotations

import argparse
import json
import sys
import time

from catan.zoo import leaderboard, registry
from catan.zoo.gate import provenance_guard
from catan.zoo.ladder import OPPONENTS, ladder_path, list_checkpoints, run_ladder, summary_table


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", nargs="+", required=True, help="run directories, e.g. runs/baseline_r1")
    ap.add_argument("--every", type=int, default=1, help="evaluate every Nth checkpoint (the last is always included)")
    ap.add_argument("--games", type=int, default=100, help="games per point")
    ap.add_argument("--final-games", type=int, default=200, help="games for the run's last checkpoint")
    ap.add_argument("--workers", type=int, default=3, help="tournament processes (CPU budget: 4 total; each is single-threaded)")
    ap.add_argument("--opponents", nargs="+", default=list(OPPONENTS))
    ap.add_argument("--out", default="results")
    ap.add_argument("--registry", default=str(registry.REGISTRY_PATH))
    ap.add_argument("--watch", action="store_true", help="keep polling for new checkpoints")
    ap.add_argument("--interval", type=int, default=300, help="seconds between polls in --watch")
    ap.add_argument("--allow-dirty", action="store_true",
                    help="ladder runs whose meta.json says git_dirty (or from a dirty tree); recorded in the ladder JSON")
    a = ap.parse_args(argv)
    dirty = provenance_guard(a.run)
    if dirty:
        print("ladder results would NOT be reproducible from a clean commit (RL-020):", file=sys.stderr)
        for c in dirty:
            print(f"  - {c}", file=sys.stderr)
        if not a.allow_dirty:
            print("refusing: pass --allow-dirty to record them as non-reproducible", file=sys.stderr)
            return 4
    seen = None
    while True:
        state = {r: [p.name for p in list_checkpoints(r)] for r in a.run}
        if state != seen:
            for r in a.run:
                lad = run_ladder(r, a.every, a.games, a.final_games, a.workers, a.opponents, a.out)
                if dirty:  # --allow-dirty: mark the file, so the leaderboard/UI can tell
                    p = ladder_path(lad["run"], a.out) if "run" in lad else None
                    if p is not None and p.is_file():
                        d = json.loads(p.read_text())
                        d["provenance_warning"] = dirty
                        p.write_text(json.dumps(d, indent=1) + "\n")
                print(summary_table(lad), flush=True)
            lb = leaderboard.write(registry.load(a.registry), a.out)
            print(f"leaderboard: {lb['games']} games, top: "
                  + ", ".join(f"{e['id']} {e['elo']:.0f}" for e in lb["entries"][:5]), flush=True)
            seen = state
        if not a.watch:
            break
        time.sleep(a.interval)  # idle poll: one directory listing per interval


if __name__ == "__main__":
    sys.exit(main())
