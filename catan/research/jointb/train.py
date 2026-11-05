"""Distil D18 counterfactual labels into JointNet(extra=EXTRA_B).

    python -m scripts.jointb_train --labels runs/jointb_labels --out runs/jointb_r1 [--init joint|nego]

Loss and regret metric as in catan.research.joint.train. Differences:
  * init ``joint`` (default) warm-starts from the D15 head in models/joint_best.pt
    with zero columns for the opponent-history inputs (initial policy = joint_vt's head);
    ``nego`` starts from the nego BC head as D15 did;
  * held-out regret is reported per opponent table, so the basic-table regret can
    be watched directly; best epoch = lowest mean of the per-table regrets.
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

from catan.research.joint.agent import _bundle_head
from catan.research.joint.model import JointNet, save
from catan.research.joint.train import _targets, collate, regret
from catan.research.nego.model import load as load_nego

from .features import EXTRA_B


def widen(net: JointNet, extra: int = EXTRA_B) -> JointNet:
    """Copy ``net`` into a JointNet with ``extra`` state inputs; new input columns start at zero."""
    out = JointNet(net.hidden, extra)
    sd = net.state_dict()
    w = sd["enc.0.weight"]
    new = torch.zeros(w.shape[0], out.enc[0].in_features)
    new[:, :w.shape[1]] = w
    sd["enc.0.weight"] = new
    out.load_state_dict(sd)
    return out


def init_net(kind: str) -> JointNet:
    if kind == "joint":
        return widen(_bundle_head("models/joint_best.pt"))
    return widen(JointNet.from_nego(load_nego("models/nego_best.pt")[0]))


def load_games(labels_dir, keep_flat=0.2, seed=0):
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
                            "decision": d["decision"], "table": r["table"]})
            games.append(out)
    return games


def train(labels, out, tau=0.002, bc=0.2, spread_ref=0.02, epochs=8, lr=5e-4, minibatch=256, init="joint",
          seed=0, log=print):
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    os.makedirs(out, exist_ok=True)
    games = load_games(labels, seed=seed)
    rng = random.Random(seed)
    rng.shuffle(games)
    nval = max(1, len(games) // 10)
    val = [x for g in games[:nval] for x in g]
    tr = [x for g in games[nval:] for x in g]
    tables = sorted({x["table"] for x in val})
    log(f"{len(games)} games, train {len(tr)} decisions, val {len(val)}")
    net = init_net(init)
    opt = torch.optim.Adam(net.parameters(), lr=lr)

    @torch.no_grad()
    def choices(samples):
        ch = {}
        for i in range(0, len(samples), 512):
            mb = samples[i:i + 512]
            for x, c in zip(mb, net(*collate(mb))[0].argmax(-1).tolist()):
                ch[id(x)] = c
        return ch

    def evaluate(tag):
        net.eval()
        ch = choices(val)
        row = {"tag": tag, "agree_with_nego": sum(ch[id(x)] == x["a"] for x in val) / max(1, len(val))}
        for t in tables:
            sub = [x for x in val if x["table"] == t]
            rh, n = regret(sub, lambda x: ch[id(x)])
            rn, _ = regret(sub, lambda x: x["a"])
            row[t] = {"head": round(rh, 6), "nego": round(rn, 6), "n": n}
        row["val_regret_head"], row["val_labelled"] = regret(val, lambda x: ch[id(x)])
        row["val_regret_nego"] = regret(val, lambda x: x["a"])[0]
        row["val_regret_teacher"] = regret(val, lambda x: x["teacher"])[0]
        row["table_mean_head"] = sum(row[t]["head"] for t in tables) / len(tables)
        log(json.dumps(row))
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
            loss = lq + bc * torch.nn.functional.cross_entropy(logits.masked_fill(~m, -1e9), a)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            tot += float(loss.detach()) * len(mb)
        row = evaluate(f"epoch{ep}")
        row["train_loss"] = tot / len(tr)
        hist.append(row)
        save(net, os.path.join(out, f"jointb_ep{ep}.pt"), {"epoch": ep, "tau": tau, "bc": bc, "init": init,
                                                           "labels": labels})
        with open(os.path.join(out, "metrics.jsonl"), "a") as f:
            f.write(json.dumps({"step": ep, "train_loss": row["train_loss"], "val_regret": row["val_regret_head"],
                                "val_regret_table_mean": row["table_mean_head"]}) + "\n")
    best = min(hist[1:], key=lambda r: r["table_mean_head"])
    json.dump({"history": hist, "best": best, "tau": tau, "bc": bc, "init": init, "train_s": time.time() - t0,
               "train_decisions": len(tr), "val_decisions": len(val)}, open(os.path.join(out, "summary.json"), "w"),
              indent=1)
    log(f"best {best['tag']}: table-mean regret {best['table_mean_head']:.5f}, all {best['val_regret_head']:.5f} "
        f"vs nego {best['val_regret_nego']:.5f}")
    return best


def bundle(head_path, out="models/jointb_best.pt", gnn="models/league_best.pt"):
    """GNN weights + ``joint_head`` in one file, the layout models/joint_best.pt uses."""
    from catan.research.joint.model import load
    net, meta = load(head_path)
    ck = torch.load(gnn, map_location="cpu", weights_only=True)
    ck["joint_head"] = {"state_dict": net.state_dict(), "hidden": net.hidden, "extra": net.extra}
    ck["joint"] = {**meta, "head": head_path, "direction": "D18"}
    torch.save(ck, out)
