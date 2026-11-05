"""DS2-11: every registry entry with games is rated, even when its spec changed after its gate run;
entries without games are listed as unrated instead of disappearing."""
import json

from catan.zoo import leaderboard


def entry(i, spec, kind="rl", **kw):
    return {"id": i, "name": i.split("-v")[0], "kind": kind, "status": "candidate", "anchor": kind != "rl",
            "spec": spec, **kw}


def tournament(path, agents, lineups):
    games = [{"seed": s, "rotation": 0, "lineup": lu, "winner": s % 4, "vp": [10 if q == s % 4 else 5 for q in range(4)],
              "truncated": False} for s, lu in enumerate(lineups)]
    path.write_text(json.dumps({"agents": agents, "game_summaries": games}))


def test_gate_games_count_after_spec_change_and_unplayed_entries_are_listed(tmp_path):
    t = tmp_path / "tournaments"
    t.mkdir()
    old = "arch_rl:checkpoint=runs/gnn/bc.pt"
    reg = {"entries": [entry("basic", "basic", kind="heuristic"),
                       entry("gnn-v1", "arch_rl:checkpoint=models/arch_best.pt"),   # spec rewritten after the gate
                       entry("mlp-v1", "arch_rl:checkpoint=runs/mlp/bc.pt", aliases=["rl:checkpoint=runs/mlp/old.pt"]),
                       entry("fresh-v1", "rl:checkpoint=runs/fresh/c.pt")]}           # registered, never played
    tournament(t / "gate_gnn-v1_vs3basic.json", [old, "basic"], [[old, "basic", "basic", "basic"]] * 8)
    tournament(t / "misc.json", ["rl:checkpoint=runs/mlp/old.pt", "basic"],
               [["rl:checkpoint=runs/mlp/old.pt", "basic", "basic", "basic"]] * 8)
    # a gate file for another entry must not claim this one's games
    tournament(t / "gate_other_vs3basic.json", ["rl:checkpoint=runs/x.pt", "basic"],
               [["rl:checkpoint=runs/x.pt", "basic", "basic", "basic"]] * 4)
    lb = leaderboard.build(reg, tmp_path, bootstrap=5)
    by = {e["id"]: e for e in lb["entries"]}
    assert by["gnn-v1"]["games"] == 8 and by["gnn-v1"]["elo"] is not None
    assert by["mlp-v1"]["games"] == 8
    assert by["fresh-v1"]["games"] == 0 and by["fresh-v1"]["elo"] is None
    assert set(by) == {e["id"] for e in reg["entries"]}
    assert lb["entries"][-1]["id"] == "fresh-v1"  # unrated rows come last
