"""Distil counterfactual option values into JointNet (D15).

    python -m scripts.joint_train --labels runs/joint_labels --out runs/joint_r1 [--tau 0.005 --bc 0.5]

Loss per decision = w * CE(pi, softmax(q / tau) over evaluated options)
                  + bc * CE(pi, nego choice)                  (anchor to the BC head)
with w = min(1, spread / spread_ref) and spread = max q - min q. Decisions with
no spread only contribute the anchor term (a fraction is kept). Whole games
are held out (10%) for the offline regret metric:
  regret(choice) = max_evaluated q - q[choice]   (mean over held-out labelled decisions)
reported for the trained head, the nego BC choice and the trader teacher.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import glob
import json
import pickle
import random
import time

import numpy as np
import torch

from catan.research.nego.model import load as load_nego

from .model import JointNet, save


def load_samples(labels_dir, keep_flat=0.2, seed=0):
    rng = random.Random(seed)
    games = []
    for p in sorted(glob.glob(os.path.join(labels_dir, "part_*.pkl"))):
        for r in pickle.load(open(p, "rb")):
            out = []
            for d in r["samples"]:
                q = d.get("q")
                spread = (max(q.values()) - min(q.values())) if q else 0.0
                if spread <= 1e-4 and rng.random() > keep_flat:
                    continue
                out.append({"s": np.asarray(d["s"], np.float32), "o": np.asarray(d["o"], np.float32),
                            "give": np.asarray(d["give"], np.int64), "get": np.asarray(d["get"], np.int64),
                            "a": d["a"], "teacher": d["teacher"], "q": q, "spread": spread,
                            "decision": d["decision"]})
            games.append(out)
    return games


def collate(samples):
    K = max(len(x["o"]) for x in samples)
    B = len(samples)
    F = samples[0]["o"].shape[1]
    s = torch.from_numpy(np.stack([x["s"] for x in samples]))
    o = torch.zeros(B, K, F)
    gv = torch.zeros(B, K, 5, dtype=torch.long)
    gt = torch.zeros(B, K, 5, dtype=torch.long)
    m = torch.zeros(B, K, dtype=torch.bool)
    for i, x in enumerate(samples):
        k = len(x["o"])
        o[i, :k] = torch.from_numpy(x["o"])
        gv[i, :k] = torch.from_numpy(x["give"])
        gt[i, :k] = torch.from_numpy(x["get"])
        m[i, :k] = True
    return s, o, gv, gt, m


def _targets(samples, K, tau, spread_ref):
    T = torch.zeros(len(samples), K)
    w = torch.zeros(len(samples))
    for i, x in enumerate(samples):
        q = x["q"]
        if not q or x["spread"] <= 1e-4:
            continue
        idx = list(q)
        vals = torch.tensor([q[j] for j in idx])
        p = torch.softmax(vals / tau, 0)
        T[i, idx] = p
        w[i] = min(1.0, x["spread"] / spread_ref)
    return T, w


def regret(samples, choose):
    r, n = 0.0, 0
    for x in samples:
        q = x["q"]
        if not q or x["spread"] <= 1e-4:
            continue
        c = choose(x)
        if c not in q:  # chose an option that was not evaluated: score with the nego choice's value
            c = x["a"]
        r += max(q.values()) - q[c]
        n += 1
    return r / max(1, n), n


def train(labels, out, tau=0.005, bc=0.5, spread_ref=0.02, epochs=8, lr=5e-4, minibatch=256, seed=0, log=print):
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    os.makedirs(out, exist_ok=True)
    games = load_samples(labels, seed=seed)
    rng = random.Random(seed)
    rng.shuffle(games)
    nval = max(1, len(games) // 10)
    val = [x for g in games[:nval] for x in g]
    tr = [x for g in games[nval:] for x in g]
    log(f"{len(games)} games, train {len(tr)} decisions, val {len(val)}; "
        f"labelled with spread: train {sum(1 for x in tr if x['spread'] > 1e-4)}, val {sum(1 for x in val if x['spread'] > 1e-4)}")
    nego, _ = load_nego("models/nego_best.pt")
    net = JointNet.from_nego(nego)
    opt = torch.optim.Adam(net.parameters(), lr=lr)

    @torch.no_grad()
    def head_choice(samples):
        out = {}
        for i in range(0, len(samples), 512):
            mb = samples[i:i + 512]
            logits, _ = net(*collate(mb))
            for x, c in zip(mb, logits.argmax(-1).tolist()):
                out[id(x)] = c
        return out

    def evaluate(tag):
        net.eval()
        ch = head_choice(val)
        r_head, n = regret(val, lambda x: ch[id(x)])
        r_nego, _ = regret(val, lambda x: x["a"])
        r_teach, _ = regret(val, lambda x: x["teacher"])
        agree = sum(1 for x in val if ch[id(x)] == x["a"]) / max(1, len(val))
        row = {"tag": tag, "val_regret_head": r_head, "val_regret_nego": r_nego, "val_regret_teacher": r_teach,
               "val_labelled": n, "agree_with_nego": agree}
        log(json.dumps({k: (round(v, 6) if isinstance(v, float) else v) for k, v in row.items()}))
        net.train()
        return row

    hist = [evaluate("init")]
    t0 = time.time()
    for ep in range(epochs):
        rng.shuffle(tr)
        tot = 0.0
        for i in range(0, len(tr), minibatch):
            mb = tr[i:i + minibatch]
            s, o, gv, gt, m = collate(mb)
            logits, _ = net(s, o, gv, gt, m)
            logp = torch.log_softmax(logits, -1).masked_fill(~m, 0.0)
            T, w = _targets(mb, logits.shape[1], tau, spread_ref)
            lq = -(w * (T * logp).sum(-1)).sum() / max(1.0, float((w > 0).sum()))
            a = torch.tensor([x["a"] for x in mb])
            lbc = torch.nn.functional.cross_entropy(logits.masked_fill(~m, -1e9), a)
            loss = lq + bc * lbc
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            tot += float(loss.detach()) * len(mb)
        row = evaluate(f"epoch{ep}")
        row["train_loss"] = tot / len(tr)
        hist.append(row)
        save(net, os.path.join(out, f"joint_ep{ep}.pt"), {"epoch": ep, "tau": tau, "bc": bc, "labels": labels})
    best = min(hist[1:], key=lambda r: r["val_regret_head"])
    json.dump({"history": hist, "best": best, "tau": tau, "bc": bc, "train_s": time.time() - t0,
               "train_decisions": len(tr), "val_decisions": len(val)}, open(os.path.join(out, "summary.json"), "w"),
              indent=1)
    log(f"best {best['tag']}: head regret {best['val_regret_head']:.5f} vs nego {best['val_regret_nego']:.5f}")
    return best
