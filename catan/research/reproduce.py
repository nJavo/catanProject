"""Rerun a recorded promotion gate on its recorded seed blocks and compare game by game.

The gate (catan/zoo/gate.py) plays fixed seed blocks: 40 (1 vs 3x basic), 41 (1 vs 3x intermediate),
42 (the mixed champion table, all 12 seatings). Games are seeded, and the agents are deterministic, so a
rerun on the same code plays the same games. ``expected`` holds each test's lineup, seating, seed block and
the recorded per-game result ("1" where the candidate seat won). With ``games`` below the recorded n, the first ``games`` games
are compared (the task order is a prefix of the full run).
"""
from __future__ import annotations

import json
from pathlib import Path

from catan.eval.stats import wilson
from catan.eval.tournament import run_tournament


def replay(expected_path: Path, games: int | None = None, workers: int = 2, tests=None) -> bool:
    exp = json.loads(Path(expected_path).read_text())
    cand = exp["candidate"]
    ok = True
    for test, t in exp["tests"].items():
        if tests and test not in tests:
            continue
        want = t["wins"]
        n = min(games or len(want), len(want))
        block, label = t["block"], " + ".join(t["agents"][1:]) + f" ({t['rotation']} seating)"
        res = run_tournament(t["agents"], n, seed=block, workers=workers, rotation=t["rotation"], out_dir=None,
                             progress=True)
        got = "".join(str(int(g["winner"] >= 0 and g["lineup"][g["winner"]] == cand)) for g in res["game_summaries"])
        k = got.count("1")
        lo, hi = wilson(k, n)
        same = got == want[:n]
        ok &= same
        print(f"{label} (block {block}, n={n}): {k}/{n} = {k / n:.1%} [{lo:.1%}, {hi:.1%}]; "
              f"recorded {want[:n].count('1')}/{n}; {'identical games' if same else 'DIFFERS from the recorded games'}")
    return ok
