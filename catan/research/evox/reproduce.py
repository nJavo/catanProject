"""Rerun D29 evox (island-model evolution of intermediate's evaluator weights) at the recorded seeds.

    uv run python -m catan.research.evox.reproduce                      # the evox_r2-v1 gate, game by game
    uv run python -m catan.research.evox.reproduce --games 8            # a quick prefix of the gate
    uv run python -m catan.research.evox.reproduce --stage evolve       # run r2 again, then pick and test

The gate stage replays the recorded gate (seed blocks 40-42) with models/evox_best.pt and exits 1 if
any game differs. The evolve stage reruns evox_r2 with its recorded arguments and seed (2) into
runs/evox_r2, picks the final genome and runs the paired final test on report blocks 60/61.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from catan.research.reproduce import replay

HERE = Path(__file__).resolve().parent

# runs/evox_r2/meta.json "args" of the recorded run (commit 287eea6); --generations 1300 covers the
# gen-1299 checkpoint, the last one the final pick looked at
R2 = ["--islands", "120", "--members", "4", "--per-table", "2", "--init", "random", "--genome", "base",
      "--sigma0", "0.3", "--tau", "0.3", "--sigma-min", "0.05", "--opp-mode", "same", "--crossover", "0.3",
      "--replace", "1", "--shrink", "4.0", "--migrate", "10", "--hof-frac", "0.2", "--int-frac", "0.5",
      "--every", "50", "--candidates", "4", "--sel-seeds", "25", "--report-seeds", "50", "--seed", "2"]


def sh(*args):
    print("+", " ".join(args), flush=True)
    subprocess.run([sys.executable, *args], check=True)


def evolve(generations: int, procs: int, run_id: str):
    sh("-m", "catan.research.evox.loop", "--run-id", run_id, "--generations", str(generations),
       "--procs", str(procs), *R2)
    if generations < 1300:
        return
    sh("-m", "scripts.evox_pick", "--run", f"runs/{run_id}", "--last", "10", "--seeds", "50",
       "--out", f"runs/{run_id}/evox_best.json")
    sh("-m", "scripts.evox_eval", "--run", f"runs/{run_id}", "--seeds", "75", "--z", f"runs/{run_id}/evox_best.json")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=("gate", "evolve"), default="gate")
    ap.add_argument("--games", type=int, help="gate: replay only the first N games of each test")
    ap.add_argument("--tests", nargs="+", choices=("basic", "intermediate", "table"))
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--generations", type=int, default=1300, help="evolve: stop after this many generations")
    ap.add_argument("--run-id", default="evox_r2")
    a = ap.parse_args(argv)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    if a.stage == "evolve":
        evolve(a.generations, a.workers, a.run_id)
        return
    sys.exit(0 if replay(HERE / "expected.json", a.games, a.workers, a.tests) else 1)


if __name__ == "__main__":
    main()
