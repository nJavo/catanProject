"""Distil D18 counterfactual labels into the jointb trade head. See docs/research/JOINTB.md.

    python -m scripts.jointb_train --labels runs/jointb_labels --out runs/jointb_r1 [--bundle-epoch 3]
"""
from __future__ import annotations

import argparse
import os
import sys

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from catan.research.jointb.train import bundle, train  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="runs/jointb_labels")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tau", type=float, default=0.002)
    ap.add_argument("--bc", type=float, default=0.2)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--init", choices=["joint", "nego"], default="joint")
    ap.add_argument("--bundle-epoch", type=int, default=None, help="only bundle this epoch's head (no training)")
    a = ap.parse_args()
    if a.bundle_epoch is None:
        best = train(a.labels, a.out, tau=a.tau, bc=a.bc, epochs=a.epochs, lr=a.lr, init=a.init,
                     log=lambda s: print(s, flush=True))
        ep = int(best["tag"].removeprefix("epoch"))
    else:
        ep = a.bundle_epoch
    bundle(os.path.join(a.out, f"jointb_ep{ep}.pt"), os.path.join(a.out, "bundle.pt"))
    print(f"bundled epoch {ep} -> {os.path.join(a.out, 'bundle.pt')}", flush=True)
