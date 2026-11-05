"""Rerun D12 (the value net) and its value_trade agent at the recorded seeds.

    uv run python -m catan.research.value.reproduce                    # the value_trade-v1 gate, game by game
    uv run python -m catan.research.value.reproduce --games 8          # a quick prefix of the gate
    uv run python -m catan.research.value.reproduce --stage train      # generate data, then train a value net

The gate stage replays the recorded promotion gate (seed blocks 40-42) with the shipped weights and
exits 1 if any game differs. The train stage regenerates this branch's share of the training data at its
recorded seeds and trains a new value net under runs/; see README.md for how close that gets.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from catan.research.reproduce import replay

HERE = Path(__file__).resolve().parent


def sh(*args):
    print("+", " ".join(args), flush=True)
    subprocess.run([sys.executable, *args], check=True)


def train(workers: int):
    # D12 data: 6,000 mixed-table games on seeds 70,000,000 + i, 15% of each seat's non-forced decisions
    sh("-m", "catan.research.value.data", "gen", "--games", "6000", "--chunk", "40", "--keep", "0.15",
       "--tables", "base", "--workers", str(workers), "--out", "runs/value_data/gen")
    sh("-m", "catan.research.value.data", "pack", "--src", "runs/value_data/gen:1.0", "--out", "runs/value_data/packed")
    # value head on the D5 GNN; the recorded run kept its step-2,226 (0.5 epoch) checkpoint
    sh("-m", "catan.research.value.train", "--data", "runs/value_data/packed", "--out", "runs/value_v1",
       "--init", "models/arch_best.pt", "--epochs", "1", "--seed", "0")
    print("New value net: runs/value_v1/best.pt. To gate it, copy it over models/value_best.pt and rerun this "
          "module without --stage.")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=("gate", "train"), default="gate")
    ap.add_argument("--games", type=int, help="replay only the first N games of each gate test")
    ap.add_argument("--tests", nargs="+", choices=("basic", "intermediate", "table"))
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args(argv)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    if a.stage == "train":
        train(a.workers)
        return
    sys.exit(0 if replay(HERE / "expected.json", a.games, a.workers, a.tests) else 1)


if __name__ == "__main__":
    main()
