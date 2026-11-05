"""D21: train the opponent classifier and report held-out accuracy by turn and seat.

    python -m scripts.opp_train --train runs/opp_data/train.npz --test runs/opp_data/test.npz \
        --run runs/opp_clf --save models/opp_best.pt

Splits the training games 90/10 (by game) for epoch selection; the test file (block 55 games)
is only scored once at the end. Writes metrics.jsonl (per epoch) and eval.json under --run.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from catan.research.opp.classifier import OppNet  # noqa: E402
from catan.research.opp.features import CLASSES  # noqa: E402

THRESHOLDS = (0.5, 0.6, 0.7, 0.8, 0.9)
GROUP = np.array([0, 1, 2, 1, 3])  # rollout policy per class: basic, intermediate, trader, intermediate, learned


def report(m, d):
    p = m.proba(d["X"])
    pred, conf, y = p.argmax(1), p.max(1), d["y"]
    ok = pred == y
    gok = GROUP[pred] == GROUP[y]
    out = {"n": int(len(y)), "acc": float(ok.mean()), "by_turn": {}, "by_seat_turn": {}, "confusion": {},
           "coverage": {}}
    for s in sorted(set(d["snap"].tolist())):
        k = d["snap"] == s
        out["by_turn"][int(s)] = {"acc": round(float(ok[k].mean()), 4), "n": int(k.sum()),
                                  "policy_acc": round(float(gok[k].mean()), 4),
                                  "policy_acc_conf0.6": round(float(gok[k & (conf >= 0.6)].mean()), 4)
                                  if (k & (conf >= 0.6)).any() else None,
                                  "per_class": {c: round(float(ok[k & (y == i)].mean()), 4)
                                                for i, c in enumerate(CLASSES) if (k & (y == i)).any()},
                                  "confident": {str(t): [round(float((conf[k] >= t).mean()), 4),
                                                         round(float(ok[k & (conf >= t)].mean()), 4)
                                                         if (k & (conf >= t)).any() else None]
                                                for t in THRESHOLDS}}
        out["by_seat_turn"][int(s)] = {int(q): round(float(ok[k & (d["seat"] == q)].mean()), 4) for q in range(4)}
    for i, c in enumerate(CLASSES):
        out["confusion"][c] = {c2: int(((y == i) & (pred == j)).sum()) for j, c2 in enumerate(CLASSES)}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--run", default="runs/opp_clf")
    ap.add_argument("--save", default="models/opp_best.pt")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    a = ap.parse_args(argv)
    torch.set_num_threads(1)
    torch.manual_seed(0)
    tr, te = dict(np.load(a.train)), dict(np.load(a.test))
    val_g = tr["game"] % 10 == 0
    va = {k: v[val_g] for k, v in tr.items() if k != "classes"}
    tr = {k: v[~val_g] for k, v in tr.items() if k != "classes"}
    m = OppNet(a.hidden)
    X = torch.from_numpy(tr["X"])
    y = torch.from_numpy(tr["y"]).long()
    m.mean.copy_(X.mean(0))
    m.std.copy_(X.std(0).clamp_min(1e-3))
    w = torch.bincount(y, minlength=len(CLASSES)).float()
    lossf = torch.nn.CrossEntropyLoss(weight=w.sum() / (len(CLASSES) * w))
    opt = torch.optim.AdamW(m.parameters(), lr=a.lr, weight_decay=a.wd)
    os.makedirs(a.run, exist_ok=True)
    best, best_state = -1.0, None
    with open(os.path.join(a.run, "metrics.jsonl"), "w") as fm:
        for ep in range(a.epochs):
            m.train()
            perm = torch.randperm(len(y))
            tot = 0.0
            for i in range(0, len(y), 512):
                b = perm[i:i + 512]
                loss = lossf(m(X[b]), y[b])
                opt.zero_grad()
                loss.backward()
                opt.step()
                tot += loss.item() * len(b)
            m.eval()
            vacc = float((m.proba(va["X"]).argmax(1) == va["y"]).mean())
            fm.write(json.dumps({"epoch": ep, "loss": tot / len(y), "val_acc": vacc}) + "\n")
            if vacc > best:
                best, best_state = vacc, {k: v.clone() for k, v in m.state_dict().items()}
    m.load_state_dict(best_state)
    m.eval()
    res = {"val_acc": best, "val": report(m, va), "test": report(m, te), "args": vars(a)}
    json.dump(res, open(os.path.join(a.run, "eval.json"), "w"), indent=1)
    m.save(a.save, val_acc=best, test_acc=res["test"]["acc"])
    print(f"val {best:.3f} test {res['test']['acc']:.3f}")
    for s, r in res["test"]["by_turn"].items():
        print(f"turn {s:3d}: acc {r['acc']:.3f} policy {r['policy_acc']:.3f} n={r['n']} per class {r['per_class']} "
              f"conf>=0.6 {r['confident']['0.6']}")
    print(json.dumps(res["test"]["confusion"]))


if __name__ == "__main__":
    main()
