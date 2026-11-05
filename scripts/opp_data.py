"""D21: labelled games for the opponent classifier (catan/research/opp, docs/research/OPP.md).

    python -m scripts.opp_data --games 1500 --split train --out runs/opp_data/train.npz --workers 3
    python -m scripts.opp_data --games 300 --split test --out runs/opp_data/test.npz --workers 3

Every seat draws a class uniformly from catan.research.opp.features.CLASSES, then a spec of that
class. Per game and seat, the public features are taken at several global turns (SNAPSHOTS;
-1 = after the first setup round, before the snake reply). Train games use training seeds
(TRAIN_SEED_BASE + 21e6 + i); test games use held-out block 55 (D21's selection block).
"""
from __future__ import annotations

import argparse
import os
import random
import sys
import time

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from catan.engine import Game, GameConfig  # noqa: E402
from catan.engine.actions import A, Action  # noqa: E402
from catan.eval.seeds import TRAIN_SEED_BASE  # noqa: E402
from catan.eval.tournament import _safe_play, heldout_seeds  # noqa: E402
from catan.research.opp.features import CLASSES, seat_features  # noqa: E402

POOL = {
    "basic": ["basic"],
    "intermediate": ["intermediate"],
    "intermediate_trader": ["intermediate_trader"],
    "search": ["search"],
    "learned": ["value_trade:checkpoint=models/value_best.pt,deterministic=true", "combo",
                "nego:checkpoint=models/nego_best.pt", "joint_vt",
                "arch_rl:checkpoint=models/league_best.pt,deterministic=true"],
}
SNAPSHOTS = (-1, 0, 4, 8, 12, 16, 24, 32, 48)
SEED_OFFSET = 21_000_000  # D21
TEST_BLOCK = 55


def prefixes(log, n):
    """{snapshot: log prefix length} for the SNAPSHOTS this game reached."""
    out = {-1: n}  # first setup round: n settlements + n roads
    turn = 0
    if len(log) >= 4 * n:
        out[0] = 4 * n
    for i, a in enumerate(log[4 * n:], start=4 * n):
        if a.type == A.END_TURN:
            turn += 1
            if turn in SNAPSHOTS:
                out[turn] = i + 1
    return out


def _init():
    import torch
    torch.set_num_threads(1)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=1500)
    ap.add_argument("--split", choices=("train", "test"), default="train")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args(argv)
    seeds = ([TRAIN_SEED_BASE + SEED_OFFSET + i for i in range(a.games)] if a.split == "train"
             else heldout_seeds(TEST_BLOCK, a.games))
    cfg = GameConfig(num_players=4, max_turns=500)
    tasks, labels = [], []
    for s in seeds:
        r = random.Random(s)
        cls = [r.randrange(len(CLASSES)) for _ in range(4)]
        tasks.append({"seed": s, "rotation": 0, "lineup": tuple(r.choice(POOL[CLASSES[c]]) for c in cls),
                      "config": cfg.to_dict(), "save_replay": True, "infoset": False})
        labels.append(cls)
    X, Y, G, N, S, SPEC = [], [], [], [], [], []
    t0 = time.time()
    from multiprocessing import get_context
    with get_context("fork").Pool(a.workers, initializer=_init) as pool:
        try:
            for gi, rec in enumerate(pool.imap(_safe_play, tasks, chunksize=1)):
                if "error" in rec:
                    print("error", rec["seed"], rec["error"][-300:], flush=True)
                    continue
                log = [Action.from_json(x) for x in rec["replay"]["actions"]]
                lay = Game(config=cfg, seed=rec["seed"]).layout
                for snap, k in prefixes(log, 4).items():
                    f = seat_features(lay, cfg, log[:k])
                    for q in range(4):
                        X.append(f[q])
                        Y.append(labels[gi][q])
                        G.append(gi)
                        N.append(snap)
                        S.append(q)
                        SPEC.append(tasks[gi]["lineup"][q])
                if (gi + 1) % 50 == 0:
                    print(f"{gi + 1}/{len(tasks)} games {time.time() - t0:.0f}s", flush=True)
        finally:
            pool.terminate()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    np.savez_compressed(a.out, X=np.array(X, np.float32), y=np.array(Y), game=np.array(G), snap=np.array(N),
                        seat=np.array(S), spec=np.array(SPEC), classes=np.array(CLASSES))
    print(f"wrote {a.out}: {len(X)} rows from {len(set(G))} games in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
