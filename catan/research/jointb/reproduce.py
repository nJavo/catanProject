"""Rerun D18 jointb (the trade head) and D21 (the opponent classifier) at their recorded seeds.

    uv run python -m catan.research.jointb.reproduce                    # the jointb_gated-v1 gate, game by game
    uv run python -m catan.research.jointb.reproduce --games 8          # a quick prefix of the gate
    uv run python -m catan.research.jointb.reproduce --stage train      # labels, head, classifier, then the gate

The gate stage replays the recorded promotion gate (seed blocks 40-42) with the shipped weights and
exits 1 if any game differs. The train stage regenerates every input at its recorded seeds and writes
new weights under runs/; see README.md for how close that gets.
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
    # D18 labels: training seeds TRAIN_SEED_BASE + 18e6 + i (seed 0). The recorded run stopped at 2,040 games.
    sh("-m", "scripts.jointb_labels", "--games", "2040", "--workers", str(workers), "--seed", "0",
       "--out", "runs/jointb_labels")
    # r5, the shipped head: nego init, lr 2e-3, bc 0.1, tau 0.002, 16 epochs; the best epoch on held-out regret is bundled
    sh("-m", "scripts.jointb_train", "--labels", "runs/jointb_labels", "--out", "runs/jointb_r5", "--init", "nego",
       "--lr", "2e-3", "--bc", "0.1", "--tau", "0.002", "--epochs", "16")
    # D21 classifier: train games on TRAIN_SEED_BASE + 21e6 + i, test games on held-out block 55
    sh("-m", "scripts.opp_data", "--games", "1500", "--split", "train", "--out", "runs/opp_data/train.npz",
       "--workers", str(workers))
    sh("-m", "scripts.opp_data", "--games", "400", "--split", "test", "--out", "runs/opp_data/test.npz",
       "--workers", str(workers))
    sh("-m", "scripts.opp_train", "--train", "runs/opp_data/train.npz", "--test", "runs/opp_data/test.npz",
       "--run", "runs/opp_clf", "--save", "runs/opp_clf/opp_best.pt")
    print("New weights: runs/jointb_r5/bundle.pt and runs/opp_clf/opp_best.pt. To gate them, copy them over "
          "models/jointb_best.pt and models/opp_best.pt and rerun this module without --stage.")


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
