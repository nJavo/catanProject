"""Register a checkpoint as a candidate, run the promotion gate, promote if it passes.

    python -m scripts.promote --anchors                       # register heuristic anchors only
    python -m scripts.promote --checkpoint runs/baseline_r1/checkpoints/ckpt_00200.pt --name baseline_r1
    python -m scripts.promote --checkpoint ... --name ... --games 400 --workers 3

The gate is described in catan/zoo/gate.py.
Exit code 0 = promoted, 2 = stays candidate, 3 = refused: the candidate's
metadata says it was selected on gate seeds (see --allow-gate-selected),
4 = refused: the run or the evaluating code is not a clean, recorded commit
(see --allow-dirty).
"""
from __future__ import annotations

import argparse
import json
import sys

from catan.zoo import leaderboard, registry
from catan.zoo.gate import champion_test_lines, provenance_guard, run_gate, selection_guard


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint")
    ap.add_argument("--name", help="model name; versions are <name>-v<k>")
    ap.add_argument("--agent", default="rl", help="registry agent that loads the checkpoint, e.g. arch_rl")
    ap.add_argument("--table-games", type=int, default=600,
                    help="games for each champion table on block 42: the candidate's and the cached same-seat "
                         "champion copy's (default 600)")
    ap.add_argument("--deterministic", action="store_true",
                    help="register the spec with deterministic=true (argmax play) so the gate and the UI play as selected")
    ap.add_argument("--anchors", action="store_true", help="register the heuristic anchors")
    ap.add_argument("--games", type=int, default=200, help="games per gate test")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--registry", default=str(registry.REGISTRY_PATH))
    ap.add_argument("--out", default="results")
    ap.add_argument("--selection-meta", help='JSON file describing how the checkpoint was selected, '
                    'e.g. {"selection_seeds": [first, last]} or {"selection_block": 52}')
    ap.add_argument("--allow-gate-selected", action="store_true",
                    help="run the gate even if the candidate was selected on gate seeds (recorded in the entry)")
    ap.add_argument("--allow-dirty", action="store_true",
                    help="run the gate even if the run's meta.json says git_dirty (or the current tree is dirty); "
                         "the reasons are recorded in the entry")
    a = ap.parse_args(argv)
    if not a.anchors and not (a.checkpoint and a.name):
        ap.error("need --checkpoint and --name (or --anchors)")
    reg = registry.load(a.registry)
    for n in registry.ANCHORS:
        registry.add_anchor(reg, n)
    registry.save(reg, a.registry)
    if not a.checkpoint:
        print("anchors:", ", ".join(registry.ANCHORS))
        return 0
    cand = registry.add_checkpoint(reg, a.checkpoint, a.name, agent=a.agent, spec_kw="deterministic=true" if a.deterministic else None)
    registry.verify(cand)
    champ = registry.champion(reg)
    registry.verify(champ)
    registry.save(reg, a.registry)
    print(f"candidate {cand['id']} ({cand['sha256'][:12]}), champion {champ['id']}", flush=True)
    if cand["id"] == champ["id"]:
        print("candidate is already the champion")
        return 0
    extra = json.load(open(a.selection_meta)) if a.selection_meta else None
    conflicts = selection_guard(cand, extra=extra)
    if conflicts:
        bar = "!" * 72
        print(bar, file=sys.stderr)
        print(f"{cand['id']} appears to have been SELECTED ON GATE SEEDS (catan/eval/seeds.py):", file=sys.stderr)
        for c in conflicts:
            print(f"  - {c}", file=sys.stderr)
        print(bar, file=sys.stderr)
        if not a.allow_gate_selected:
            print("refusing to run the gate: re-select on a selection block (50-59), or pass "
                  "--allow-gate-selected to record the result as biased", file=sys.stderr)
            return 3
        reg = registry.load(a.registry)
        registry.get(reg, cand["id"])["selection_warning"] = conflicts
        registry.save(reg, a.registry)
    dirty = provenance_guard([f"runs/{cand['source_run']}"] if cand.get("source_run") else [])
    if dirty:
        bar = "!" * 72
        print(bar, file=sys.stderr)
        print(f"{cand['id']}: results would NOT be reproducible from a clean commit (RL-020):", file=sys.stderr)
        for c in dirty:
            print(f"  - {c}", file=sys.stderr)
        print(bar, file=sys.stderr)
        if not a.allow_dirty:
            print("refusing to run the gate: retrain/commit, or pass --allow-dirty to record the result "
                  "as non-reproducible", file=sys.stderr)
            return 4
        reg = registry.load(a.registry)
        registry.get(reg, cand["id"])["provenance_warning"] = dirty
        registry.save(reg, a.registry)
    res = run_gate(cand, champ, games=a.games, workers=a.workers, out_dir=a.out, table_games=a.table_games)
    reg = registry.load(a.registry)  # re-read: the gate takes minutes
    e = registry.get(reg, cand["id"])
    e["eval"] = res
    if res["promote"]:
        if champ["kind"] == "rl":
            registry.set_status(reg, champ["id"], "retired")
        registry.set_status(reg, cand["id"], "promoted")
    registry.save(reg, a.registry)
    leaderboard.write(reg, a.out)
    print(json.dumps({k: res[k] for k in ("promote", "reasons", "champion")}, indent=1))
    for h, r in res["heuristic_tests"].items():
        c = r["candidate"]
        print(f"  vs 3x{h}: {c['win_rate']:.1%} [{c['ci95'][0]:.1%},{c['ci95'][1]:.1%}] n={c['games']}; "
              f"champion {r['champion']['win_rate']:.1%}; worse={r['worse']}")
    for line in champion_test_lines(res["champion_test"]):
        print(line)
    return 0 if res["promote"] else 2


if __name__ == "__main__":
    sys.exit(main())
