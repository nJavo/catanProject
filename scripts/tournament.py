"""Run a seat-rotated tournament between registry agents.

    python scripts/tournament.py --agents random basic intermediate search --games 400 --workers 14 --seed 0
    python scripts/tournament.py --agents basic "rl:checkpoint=runs/x/checkpoints/last.pt" --games 200

Writes results/tournaments/<id>.json.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from catan.eval import seeds as S  # noqa: E402
from catan.eval.tournament import format_table, run_tournament  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--agents", nargs="+", required=True, help="agent specs: name or name:key=val,...")
    ap.add_argument("--games", type=int, default=400)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--seed", type=int, default=0,
                    help="held-out seed block 0..89; purposes are registered in catan/eval/seeds.py")
    ap.add_argument("--players", type=int, default=4)
    ap.add_argument("--max-turns", type=int, default=500)
    ap.add_argument("--rotation", choices=["cyclic", "all", "none"], default="cyclic")
    ap.add_argument("--out", default="results")
    ap.add_argument("--id", default=None)
    ap.add_argument("--save-replays", type=int, default=0, help="save the first K games as replays")
    ap.add_argument("--json", action="store_true", help="print the full result JSON")
    ap.add_argument("--infoset", action="store_true",
                    help="give agents game.observe_for(seat) instead of the full game (hidden state raises)")
    a = ap.parse_args(argv)
    try:
        S.block_seeds(a.seed, 0)
    except ValueError as e:
        ap.error(str(e))
    purpose = S.purpose_of(S.block_seeds(a.seed, 1)[0])
    print(f"seed block {a.seed}: {purpose}", file=sys.stderr)
    if purpose == "gate":
        print(f"WARNING: block {a.seed} is reserved for the promotion gate (catan/zoo/gate.py); "
              "results on it must not be used to select checkpoints", file=sys.stderr)
    res = run_tournament(a.agents, a.games, seed=a.seed, workers=a.workers, num_players=a.players,
                         max_turns=a.max_turns, rotation=a.rotation, out_dir=a.out, tid=a.id,
                         save_replays=a.save_replays, progress=True, infoset=a.infoset)
    print(json.dumps(res, indent=1) if a.json else format_table(res))
    if res["failed_games"]:
        print(f"WARNING: {len(res['failed_games'])} games failed", file=sys.stderr)
    print(f"wrote {os.path.join(a.out, 'tournaments', res['id'] + '.json')}")


if __name__ == "__main__":
    main()
