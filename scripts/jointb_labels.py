"""Generate counterfactual trade-option labels for D18. See docs/research/JOINTB.md.

    python -m scripts.jointb_labels --games 4500 --workers 3 --out runs/jointb_labels
"""
from __future__ import annotations

import argparse
import os
import sys

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from catan.research.jointb.labels import run  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=4500)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--out", default="runs/jointb_labels")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--frac", type=float, default=0.5)
    ap.add_argument("--k", type=int, default=8)
    a = ap.parse_args()
    run(a.games, a.workers, a.out, a.seed, a.frac, a.k, log=lambda s: print(s, flush=True))
