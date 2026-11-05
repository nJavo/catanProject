"""D34: the gate's primary champion test is same-seat paired vs a cached champion copy."""
import json

from catan.eval import tournament as T
from catan.zoo import gate, leaderboard


def test_copy_spec_round_trip_and_agent_ignores_marker():
    assert T.copy_spec("basic") == "basic:gate_copy=1"
    assert T.copy_spec("rl:checkpoint=a.pt") == "rl:checkpoint=a.pt,gate_copy=1"
    assert type(T._agent(T.copy_spec("basic"), 0)) is type(T._agent("basic", 0))


def test_same_seat_check_decides_on_discordant_pairs():
    cand = [1] * 40 + [0] * 60
    copy = [1] * 15 + [0] * 25 + [0] * 55 + [1] * 5   # candidate-only 25, copy-only 5
    r = gate.same_seat_check(cand, copy)
    assert r["discordant"] == {"candidate_only": 25, "copy_only": 5} and r["better"] and r["mcnemar_p"] < 0.01
    assert abs(r["diff"] - 0.20) < 1e-9 and r["diff_ci95"][0] > 0
    same = gate.same_seat_check(cand, cand)
    assert not same["better"] and same["mcnemar_p"] == 1.0 and same["diff_ci95"] == [0.0, 0.0]
    # the within-table result is secondary: a big within-table win does not promote
    even = [1, 0] * 50
    heur = {"basic": gate.heuristic_check(even, even, 0.25)}
    ct = gate.champion_check({"kind": "same_seat", "lineup": ["c", "h", "intermediate", "intermediate"],
                              "cand": cand, "copy": cand,
                              "within": {"cand_wins": 300, "champ_wins": 150, "games": 600}})
    assert ct["secondary"]["better"] and not ct["better"]
    assert not gate.decide(heur, ct)["promote"]
    ct2 = gate.champion_check({"kind": "same_seat", "lineup": ["c", "h"], "cand": cand, "copy": copy,
                               "within": {"cand_wins": 100, "champ_wins": 100, "games": 600}})
    assert gate.decide(heur, ct2)["promote"]
    worse = {"basic": gate.heuristic_check([0] * 100, [1] * 40 + [0] * 60, 0.25)}
    assert not gate.decide(worse, ct2)["promote"]
    assert "cand" not in ct2 and "copy" in ct2  # raw vectors are not stored in the registry
    lines = gate.champion_test_lines(ct2)
    assert "primary, same-seat" in lines[0] and "secondary" in lines[1]


def test_run_gate_same_seat_end_to_end_and_copy_cache(tmp_path, monkeypatch):
    """Real (cheap) games: a candidate identical to the champion gets every pair identical, and the
    copy table is played once per champion, then reused."""
    champ = {"id": "champ-v1", "spec": "basic", "kind": "rl"}
    cand = {"id": "same-v1", "spec": T.copy_spec("basic"), "kind": "rl"}
    out = gate.run_gate(cand, champ, games=4, workers=1, out_dir=tmp_path, table_games=12)
    ct = out["champion_test"]
    assert ct["kind"] == "same_seat" and ct["pairs"] == 12
    assert ct["discordant"] == {"candidate_only": 0, "copy_only": 0} and not ct["better"] and not out["promote"]
    assert ct["control_table"] == "gate_champ_champ-v1_table_copy_n12"
    cf = tmp_path / "tournaments" / "gate_champ_champ-v1_table_copy_n12.json"
    assert json.loads(cf.read_text())["agents"] == [T.copy_spec("basic"), "basic", "intermediate"]
    assert "secondary" in ct and ct["secondary"]["games"] == 12
    played = []
    real = gate.run_tournament
    monkeypatch.setattr(gate, "run_tournament", lambda agents, *a, **k: played.append(list(agents)) or real(agents, *a, **k))
    other = {"id": "rand-v1", "spec": "random", "kind": "rl"}
    out2 = gate.run_gate(other, champ, games=4, workers=1, out_dir=tmp_path, table_games=12)
    assert [T.copy_spec("basic"), "basic", "intermediate", "intermediate"] not in played  # copy table cached
    assert ["random", "basic", "intermediate", "intermediate"] in played
    assert out2["champion_test"]["pairs"] == 12


def test_leaderboard_counts_copy_seat_as_the_champion(tmp_path):
    reg = {"entries": [{"id": "basic", "name": "basic", "kind": "heuristic", "status": "promoted", "spec": "basic"},
                       {"id": "intermediate", "name": "intermediate", "kind": "heuristic", "status": "promoted",
                        "spec": "intermediate"}]}
    t = {"agents": [T.copy_spec("basic"), "basic", "intermediate"], "game_summaries": [
        {"lineup": [T.copy_spec("basic"), "basic", "intermediate", "intermediate"], "winner": 0, "vp": [10, 5, 4, 3]}]}
    (tmp_path / "tournaments").mkdir()
    (tmp_path / "tournaments" / "gate_champ_x_table_copy_n12.json").write_text(json.dumps(t))
    rk, _ = leaderboard.collect(tmp_path, reg)
    assert rk and rk[0][0][0] == "basic"
