"""D29 final pick: re-evaluate the last K checkpoint champions on fresh selection-block seeds (block 56,
offset 60,000+, never used by the loop) and write the best as the model file (Such et al. 2017 elite
re-evaluation; the per-checkpoint pick is the max of noisy estimates).

    python -m scripts.evox_pick --run runs/evox_r2 --last 10 --seeds 50 --out models/evox_best.json
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

from catan.eval.seeds import block_seeds
from catan.research.evox.agent import DIM
from catan.research.evox.loop import FIT_BLOCK, Runner, eval_panel

PICK_OFFSET = 60_000


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--last", type=int, default=10)
    ap.add_argument("--seeds", type=int, default=50, help="x4 seats per opponent")
    ap.add_argument("--out", required=True)
    ap.add_argument("--procs", type=int, default=3)
    a = ap.parse_args(argv)
    run = Path(a.run)
    champs = [json.loads(x) for x in (run / "champions.jsonl").read_text().splitlines() if x.strip()][-a.last:]
    with open(run / "state.pkl", "rb") as f:
        st, _ = pickle.load(f)
    seeds = block_seeds(FIT_BLOCK, PICK_OFFSET + a.seeds)[PICK_OFFSET:]
    runner = Runner(a.procs)
    rows = []
    try:
        for c in champs:
            p = eval_panel(runner, c["z"], ("basic", "intermediate"), seeds)
            score = p["basic"]["score"] + p["intermediate"]["score"]
            rows.append((score, c, p))
            print(c["gen"], c["id"], f"score {score:.3f}", {o: round(p[o]["win_rate"], 3) for o in p}, flush=True)
    finally:
        runner.close()
    score, c, p = max(rows, key=lambda r: r[0])
    meta = json.loads((run / "meta.json").read_text())
    out = {"z": c["z"], "id": c["id"], "gen": c["gen"], "run": run.name, "root": c["root"],
           "root_z": st["roots"][c["root"]], "genes": len(c["z"]), "base_genes": DIM,
           "pick": {"block": FIT_BLOCK, "offset": PICK_OFFSET, "seeds": a.seeds, "score": score,
                    "win_rate": {o: p[o]["win_rate"] for o in p}, "n": {o: p[o]["n"] for o in p},
                    "candidates": [{"gen": r[1]["gen"], "id": r[1]["id"], "score": round(r[0], 4)} for r in rows]},
           "mode": meta.get("mode")}
    Path(a.out).write_text(json.dumps(out, indent=1))
    (run / "pick.json").write_text(json.dumps(out, indent=1))
    print("picked", c["gen"], c["id"], a.out)


if __name__ == "__main__":
    main()
