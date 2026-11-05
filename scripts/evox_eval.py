"""D29 final evaluation: evolved genome vs its generation-0 ancestor (and references), paired on report blocks.

    python -m scripts.evox_eval --run runs/evox_r1 [--seeds 75] [--z models/evox_best.json]

Every genome plays the same (seed, seat) games 1v3 vs 3x random / basic / intermediate on report blocks
60 (seeds 1000+, after the curve's seeds) and 61, so comparisons are paired (exact McNemar on discordant
games). Writes <run>/final_eval.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from catan.eval.seeds import block_seeds
from catan.eval.stats import binom_two_sided_p
from catan.research.evox.agent import INTERMEDIATE_Z
from catan.research.evox.loop import ANCHORS, Runner, eval_panel


def paired(a, b):
    n = len(a)
    only_a = sum(x > y for x, y in zip(a, b))
    only_b = sum(y > x for x, y in zip(a, b))
    return {"n": n, "a": sum(a) / n, "b": sum(b) / n, "diff": (sum(a) - sum(b)) / n,
            "only_a": only_a, "only_b": only_b, "p": binom_two_sided_p(only_a, only_a + only_b)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--z", help="genome json (default <run>/best.json)")
    ap.add_argument("--seeds", type=int, default=75, help="seeds per block; x4 seats x2 blocks games per table")
    ap.add_argument("--procs", type=int, default=3)
    a = ap.parse_args(argv)
    run = Path(a.run)
    best = json.loads(Path(a.z or run / "best.json").read_text())
    genomes = {"evolved": best["z"], "ancestor": best["root_z"], "zero": [0.0] * len(best["z"]),
               "intermediate_genome": list(INTERMEDIATE_Z)}
    seeds = block_seeds(60, 1000 + a.seeds)[1000:] + block_seeds(61, a.seeds)
    runner = Runner(a.procs)
    try:
        res = {k: eval_panel(runner, z, ANCHORS, seeds) for k, z in genomes.items()}
    finally:
        runner.close()
    out = {"seeds": {"60": [seeds[0], seeds[a.seeds - 1]], "61": [seeds[a.seeds], seeds[-1]]},
           "best": {k: best.get(k) for k in ("id", "gen", "root")}, "tables": {}, "paired": {}}
    for k, r in res.items():
        out["tables"][k] = {o: {x: r[o][x] for x in ("win_rate", "ci95", "n", "score")} for o in ANCHORS}
    for b in ("ancestor", "zero", "intermediate_genome"):
        out["paired"][f"evolved_vs_{b}"] = {o: paired(res["evolved"][o]["per_game"], res[b][o]["per_game"]) for o in ANCHORS}
    (run / "final_eval.json").write_text(json.dumps(out, indent=1))
    for k, t in out["tables"].items():
        print(k, {o: f"{t[o]['win_rate']:.3f} [{t[o]['ci95'][0]:.3f},{t[o]['ci95'][1]:.3f}] n={t[o]['n']}" for o in ANCHORS})
    for k, t in out["paired"].items():
        print(k, {o: f"diff {t[o]['diff']:+.3f} p={t[o]['p']:.2g} ({t[o]['only_a']}/{t[o]['only_b']})" for o in ANCHORS})


if __name__ == "__main__":
    main()
