"""Train a win-probability value net on packed outcome data (catan.research.value.data pack).

    python -m catan.research.value.train --data runs/value_data/packed --out runs/value_v1 \
        --init models/arch_best.pt --epochs 3 --threads 3

The model is the D5 GraphPolicy (trunk initialised from ``--init``); only its value head output
is trained, as a logit, with BCE on z. Validation holds out whole games (gid % 10 == 0). Reported
on validation: Brier, log-loss, ECE and a 10-bin reliability table, overall and per source, plus
two baselines: the train base rate, and a logistic fit of intermediate's evaluate margin (only on
rows from our own generator, which record the margin). Best epoch by val log-loss is kept.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time

import numpy as np
import torch

from catan.research.arch.agent import load_checkpoint, save_checkpoint
from catan.research.value.data import ObsFile


def calibration(p, z, bins=10) -> dict:
    p = np.clip(np.asarray(p, np.float64), 1e-6, 1 - 1e-6)
    z = np.asarray(z, np.float64)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    rel, ece = [], 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            rel.append({"bin": [round(edges[b], 2), round(edges[b + 1], 2)], "n": int(m.sum()),
                        "mean_p": float(p[m].mean()), "freq": float(z[m].mean())})
            ece += m.mean() * abs(p[m].mean() - z[m].mean())
    return {"n": int(len(z)), "brier": float(((p - z) ** 2).mean()),
            "logloss": float(-(z * np.log(p) + (1 - z) * np.log(1 - p)).mean()),
            "ece": float(ece), "reliability": rel}


def fit_logistic(x, z, iters=50):
    """1-D logistic regression by Newton's method: returns (a, b) with p = sigmoid(a*x + b)."""
    a, b = 0.0, 0.0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(a * x + b)))
        w = p * (1 - p) + 1e-9
        g = np.array([((p - z) * x).sum(), (p - z).sum()])
        h = np.array([[(w * x * x).sum(), (w * x).sum()], [(w * x).sum(), w.sum()]])
        a, b = np.array([a, b]) - np.linalg.solve(h, g)
    return float(a), float(b)


@torch.no_grad()
def predict(model, obs, idx, batch=2048):
    """idx must be sorted (sequential memmap reads); returns p aligned with idx."""
    assert (np.diff(idx) > 0).all()
    model.eval()
    out = []
    for s in range(0, len(idx), batch):
        _, v = model(torch.from_numpy(obs.rows(idx[s:s + batch]).astype(np.float32)))
        out.append(torch.sigmoid(v).numpy())
    return np.concatenate(out).astype(np.float64)


def validate(model, obs, meta, val, sources):
    p = predict(model, obs, val)
    z = meta["z"][val]
    out = {"all": calibration(p, z)}
    for s, name in enumerate(sources):
        m = meta["source"][val] == s
        if m.any():
            out[os.path.basename(name)] = {k: v for k, v in calibration(p[m], z[m]).items() if k != "reliability"}
    return out, p


def main(argv=None):
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--init", default="models/arch_best.pt")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--threads", type=int, default=3, help="single training process: use the 3-process budget as threads")
    ap.add_argument("--evals-per-epoch", type=int, default=4)
    ap.add_argument("--max-val", type=int, default=60_000, help="cap validation rows (random whole games kept)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    os.makedirs(a.out, exist_ok=True)
    obs = ObsFile(os.path.join(a.data, "obs.npy"))  # pread, not mmap: keeps RSS small
    md = np.load(os.path.join(a.data, "meta.npz"))
    meta = {k: md[k] for k in md.files}
    sources = list(meta.pop("sources"))
    is_val = meta["gid"] % 10 == 0
    val, tr = np.flatnonzero(is_val), np.flatnonzero(~is_val)
    rng = np.random.default_rng(a.seed)
    if len(val) > a.max_val:
        val = np.sort(rng.choice(val, a.max_val, replace=False))
    log = open(os.path.join(a.out, "metrics.jsonl"), "a")

    def emit(row):
        row["time"] = time.time()
        log.write(json.dumps(row) + "\n")
        log.flush()
        print(json.dumps({k: v for k, v in row.items() if k != "val"} |
                         {"val": {k: v for k, v in row.get("val", {}).get("all", {}).items() if k != "reliability"}}),
              flush=True)

    # baselines on validation
    zt, zv = meta["z"][tr], meta["z"][val]
    base = {"base_rate": float(zt.mean()), "constant": calibration(np.full(len(val), zt.mean()), zv)}
    mt, mv = ~np.isnan(meta["margin"][tr]), ~np.isnan(meta["margin"][val])
    if mt.sum() > 100 and mv.sum() > 100:
        ab = fit_logistic(meta["margin"][tr][mt], zt[mt])
        pm = 1 / (1 + np.exp(-(ab[0] * meta["margin"][val][mv] + ab[1])))
        base["heur_margin_logistic"] = {"a": ab[0], "b": ab[1], **calibration(pm, zv[mv])}
    emit({"kind": "baselines", "rows_train": int(len(tr)), "rows_val": int(len(val)),
          "games_train": int(len(np.unique(meta["gid"][tr]))), "games_val": int(len(np.unique(meta["gid"][val]))),
          **base})

    model, ck = load_checkpoint(a.init)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    cfg = ck["config"]
    extra = {"value_logit": True, "value_data": a.data, "value_init": a.init}
    step, best = 0, (float("inf"), None)
    res0, _ = validate(model, obs, meta, val, sources)  # the init model's calibration (e.g. v1 on new data)
    emit({"kind": "eval", "step": 0, "epoch": 0.0, "train_bce": None, "val": res0})
    every = max(1, len(tr) // a.batch // a.evals_per_epoch)
    for ep in range(a.epochs):
        perm = rng.permutation(tr)
        model.train()
        tot, k = 0.0, 0
        for s in range(0, len(perm) - a.batch + 1, a.batch):
            i = np.sort(perm[s:s + a.batch])
            o = torch.from_numpy(obs.rows(i).astype(np.float32))
            _, v = model(o)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(v, torch.from_numpy(meta["z"][i]))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot, k, step = tot + loss.item(), k + 1, step + 1
            if step % every == 0:
                res, p = validate(model, obs, meta, val, sources)
                row = {"kind": "eval", "step": step, "epoch": ep + step * a.batch / len(tr) % 1,
                       "train_bce": tot / k, "val": res}
                # the margin baseline is fit on our generator's rows only: compare on the same rows
                if "heur_margin_logistic" in base:
                    row["val_same_rows_as_margin"] = {kk: vv for kk, vv in calibration(p[mv], zv[mv]).items()
                                                      if kk != "reliability"}
                emit(row)
                tot, k = 0.0, 0
                model.train()
                if res["all"]["logloss"] < best[0]:
                    best = (res["all"]["logloss"], step)
                    save_checkpoint(os.path.join(a.out, "best.pt"), model, cfg, {**extra, "step": step})
    emit({"kind": "done", "best_val_logloss": best[0], "best_step": best[1]})


if __name__ == "__main__":
    main()
