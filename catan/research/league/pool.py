"""League state shared by the learner processes (file-locked JSON).

state.json:
  members: id -> {"kind": "agent"|"snapshot", "spec": registry spec (agent),
                  "path": checkpoint (snapshot), "role": ..., "created": ...}
  payoff:  learner_role -> opponent mode -> {"wins": float, "games": float}
           (exponentially decayed counts; "mode" is the opponent agent name, or
           "a+b+c" for a mixed table)
  roles:   role -> {"generation": int, "segments": int, "latest": snapshot id}
"""
from __future__ import annotations

import contextlib
import fcntl
import json
import os
import time


@contextlib.contextmanager
def locked(league_dir):
    os.makedirs(league_dir, exist_ok=True)
    with open(os.path.join(league_dir, ".lock"), "w") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        path = os.path.join(league_dir, "state.json")
        st = json.load(open(path)) if os.path.exists(path) else {"members": {}, "payoff": {}, "roles": {}}
        yield st
        tmp = path + ".tmp"
        json.dump(st, open(tmp, "w"), indent=1)
        os.replace(tmp, path)


def read(league_dir):
    with locked(league_dir) as st:
        return json.loads(json.dumps(st))


def add_member(st, mid, **info):
    st["members"].setdefault(mid, dict(info, created=time.strftime("%Y-%m-%dT%H:%M:%S")))


def record(st, role, results: dict, decay: float):
    """results: mode -> (wins, games) from one segment. Older evidence decays."""
    row = st["payoff"].setdefault(role, {})
    for m in row.values():
        m["wins"] *= decay
        m["games"] *= decay
    for mode, (w, n) in results.items():
        r = row.setdefault(mode, {"wins": 0.0, "games": 0.0})
        r["wins"] += w
        r["games"] += n


def win_rate(st, role, mode, prior=(0.25, 4.0)):
    """Posterior-mean 1v3 win rate (prior: parity 1/4 with weight 4 games)."""
    r = st["payoff"].get(role, {}).get(mode, {"wins": 0.0, "games": 0.0})
    return (r["wins"] + prior[0] * prior[1]) / (r["games"] + prior[1])
