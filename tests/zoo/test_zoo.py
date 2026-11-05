import json

import pytest
from fastapi.testclient import TestClient

from catan.rl.model import PolicyNet, save_checkpoint
from catan.rl.spaces import ACT_DIM, OBS_DIM
from catan.server.app import create_app
from catan.zoo import gate, leaderboard, registry
from catan.zoo.ladder import ladder_path, run_ladder, select


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory):
    root = tmp_path_factory.mktemp("repo")
    ck = root / "runs" / "r1" / "checkpoints"
    ck.mkdir(parents=True)
    net = PolicyNet(OBS_DIM, ACT_DIM, hidden=16, blocks=1)
    for u in (1, 2, 3):
        save_checkpoint(ck / f"ckpt_{u:05d}.pt", net, {"model": {"hidden": 16, "blocks": 1}},
                        extra={"update": u, "step": 100 * u})
    return root


# ---------------------------------------------------------------- registry
def test_registry_round_trip_and_sha(run_dir, tmp_path):
    path = tmp_path / "models" / "registry.json"
    reg = registry.load(path)
    for a in registry.ANCHORS:
        registry.add_anchor(reg, a)
    e = registry.add_checkpoint(reg, "runs/r1/checkpoints/ckpt_00002.pt", "r1", root=run_dir)
    assert e["id"] == "r1-v1" and e["status"] == "candidate" and e["source_run"] == "r1"
    assert e["update"] == 2 and e["step"] == 200 and len(e["sha256"]) == 64
    assert e["spec"] == "rl:checkpoint=runs/r1/checkpoints/ckpt_00002.pt"
    # same bytes -> same entry; different checkpoint -> next version
    assert registry.add_checkpoint(reg, "runs/r1/checkpoints/ckpt_00002.pt", "r1", root=run_dir) is e
    registry.save(reg, path)
    reg2 = registry.load(path)
    assert reg2 == json.loads(json.dumps(reg))
    assert registry.champion(reg2)["id"] == registry.DEFAULT_CHAMPION
    registry.set_status(reg2, "r1-v1", "promoted")
    assert registry.champion(reg2)["id"] == "r1-v1"
    with pytest.raises(ValueError):
        registry.set_status(reg2, "basic", "retired")
    registry.verify(registry.get(reg2, "r1-v1"), root=run_dir)
    bad = dict(registry.get(reg2, "r1-v1"), sha256="0" * 64)
    with pytest.raises(ValueError, match="sha256 mismatch"):
        registry.verify(bad, root=run_dir)


# ---------------------------------------------------------------- gate
def test_gate_promotes_only_on_significant_champion_win():
    even = [1, 0] * 50
    heur = {"basic": gate.heuristic_check(even, even, 0.25)}
    win = gate.champion_check({"kind": "mixed", "cand_wins": 60, "champ_wins": 30, "games": 200})
    tie = gate.champion_check({"kind": "mixed", "cand_wins": 35, "champ_wins": 30, "games": 200})
    assert win["better"] and not tie["better"]
    assert gate.decide(heur, win)["promote"]
    assert not gate.decide(heur, tie)["promote"]


def test_gate_blocks_regression_vs_heuristics():
    cand = [1] * 10 + [0] * 90
    champ = [1] * 40 + [0] * 60
    r = gate.heuristic_check(cand, champ, 0.25)
    assert r["worse"] and r["discordant"] == {"candidate_only": 0, "champion_only": 30}
    win = gate.champion_check({"kind": "mixed", "cand_wins": 60, "champ_wins": 30, "games": 200})
    d = gate.decide({"basic": r}, win)
    assert not d["promote"] and "3xbasic" in d["reasons"][0]
    # small noisy difference is not "significantly worse"
    assert not gate.heuristic_check([1] * 30 + [0] * 70, [1] * 33 + [0] * 67, 0.25)["worse"]


def test_gate_vs_heuristic_champion_uses_wilson_bound():
    assert gate.champion_check({"kind": "anchor", "wins": 80, "games": 200, "null": 0.25})["better"]
    assert not gate.champion_check({"kind": "anchor", "wins": 55, "games": 200, "null": 0.25})["better"]
    assert gate.heuristic_check([0] * 200, None, 0.25)["worse"]
    assert not gate.heuristic_check([1] * 50 + [0] * 150, None, 0.25)["worse"]


# ---------------------------------------------------------------- ladder + leaderboard
def test_select_keeps_last():
    assert select([1, 2, 3, 4, 5], 2) == [1, 3, 5]
    assert select([1, 2, 3, 4], 3) == [1, 4]


def test_ladder_schema_incremental_and_leaderboard(run_dir, monkeypatch):
    monkeypatch.chdir(run_dir)
    lad = run_ladder("runs/r1", every=2, games=4, final_games=8, workers=1, opponents=("random",), log=lambda *_: None)
    import torch
    assert torch.get_num_threads() == 1  # workers fork from here; one thread each
    assert lad["schema"] == 1 and lad["kind"] == "ladder" and lad["run"] == "r1" and lad["seed_block"] == 20
    assert [c["update"] for c in lad["checkpoints"]] == [1, 3]
    for c in lad["checkpoints"]:
        assert set(c) >= {"checkpoint", "sha256", "update", "step", "spec", "results"}
        r = c["results"]["random"]
        assert set(r) >= {"games", "wins", "win_rate", "ci95", "avg_vp", "outcomes", "lineups"}
        assert r["ci95"][0] <= r["win_rate"] <= r["ci95"][1]
        assert len(r["outcomes"]) == r["games"]
    assert lad["checkpoints"][-1]["results"]["random"]["games"] == 8
    assert ladder_path("r1").is_file()
    # second call plays nothing new (would need a tournament otherwise)
    monkeypatch.setattr("catan.zoo.ladder.point", lambda *a, **k: pytest.fail("re-evaluated"))
    run_ladder("runs/r1", every=2, games=4, final_games=8, workers=1, opponents=("random",))

    reg = {"entries": []}
    registry.add_anchor(reg, "random")
    registry.add_checkpoint(reg, "runs/r1/checkpoints/ckpt_00003.pt", "r1")
    lb = leaderboard.build(reg, "results", bootstrap=20)
    ids = {e["id"]: e for e in lb["entries"]}
    assert set(ids) == {"random", "r1-v1"} and lb["games"] == 8
    # mean-centred, nothing pinned: the average is 1500 and random has a real interval
    assert abs(sum(e["elo"] for e in lb["entries"]) / len(lb["entries"]) - 1500) < 0.5
    assert ids["random"]["ci95"][0] < ids["random"]["elo"] < ids["random"]["ci95"][1]
    assert ids["r1-v1"]["ci95"][0] <= ids["r1-v1"]["elo"] <= ids["r1-v1"]["ci95"][1]


# ---------------------------------------------------------------- endpoints
def test_endpoints(run_dir):
    c = TestClient(create_app(data_dir=run_dir, static_dir=run_dir / "nodist"))
    assert c.get("/api/registry").json() == {"entries": [], "champion": None}  # no registry yet
    (run_dir / "models").mkdir(exist_ok=True)
    (run_dir / "models" / "registry.json").write_text(json.dumps({"version": 1, "entries": [{"id": "basic"}]}))
    assert c.get("/api/registry").json()["entries"][0]["id"] == "basic"
    assert "entries" in c.get("/api/leaderboard").json()
    lads = c.get("/api/ladders").json()["ladders"]
    assert [x["run"] for x in lads] == ["r1"]
    assert "outcomes" not in lads[0]["checkpoints"][0]["results"]["random"]
    assert c.get("/api/ladders/r1").json()["run"] == "r1"
    assert c.get("/api/ladders/nope").status_code == 404
    for bad in ("..", "..%2Fsecret", "a b"):
        assert c.get(f"/api/ladders/{bad}").status_code in (400, 404)


# ---------------------------------------------------------------- champion agent
def test_champion_agent_and_api(run_dir, tmp_path):
    from catan.rl.policy_agent import PolicyAgent
    from catan.zoo import champion
    root = tmp_path
    (root / "runs").symlink_to(run_dir / "runs")
    path = root / registry.REGISTRY_PATH
    reg = registry.load(path)
    registry.add_anchor(reg, "basic")
    e = registry.add_checkpoint(reg, "runs/r1/checkpoints/ckpt_00003.pt", "r1", root=root)
    registry.save(reg, path)
    with pytest.raises(ValueError, match="no champion"):
        champion.make_champion(root=root)
    e["eval"] = {"champion": "intermediate",
                 "heuristic_tests": {"basic": {"candidate": {"wins": 30, "games": 200, "win_rate": 0.15,
                                                             "ci95": [0.107, 0.207]}}},
                 "champion_test": {"kind": "anchor", "candidate": {"wins": 80, "games": 200, "win_rate": 0.4,
                                                                   "ci95": [0.33, 0.47]}}}
    registry.set_status(reg, e["id"], "promoted")
    registry.save(reg, path)
    assert isinstance(champion.make_champion(root=root, seed=1), PolicyAgent)
    s = champion.summary(champion.current(root))
    assert s["id"] == "r1-v1" and s["version"] == "1" and s["step"] == 300
    assert s["label"] == "Champion — r1 (vs 3×basic 15% [11%, 21%], n=200)"
    api = TestClient(create_app(data_dir=root, static_dir=root / "nodist")).get("/api/registry").json()
    assert api["champion"]["label"] == s["label"]
    reg["entries"][-1]["sha256"] = "0" * 64
    registry.save(reg, path)
    with pytest.raises(ValueError, match="sha256"):
        champion.make_champion(root=root)


def test_run_gate_end_to_end_reuses_champion_baseline(run_dir, tmp_path, monkeypatch):
    reg = {"entries": []}
    champ = registry.add_anchor(reg, "basic")
    cand = registry.add_checkpoint(reg, "runs/r1/checkpoints/ckpt_00001.pt", "r1", root=run_dir)
    cand["spec"] = f"rl:checkpoint={run_dir}/{cand['checkpoint']}"
    out = gate.run_gate(cand, champ, games=4, workers=1, out_dir=tmp_path)
    assert out["promote"] is False and out["champion"] == "basic"
    assert set(out["heuristic_tests"]) == {"basic", "intermediate"}
    assert (tmp_path / "tournaments" / "gate_champ_basic_1v3intermediate_n4.json").is_file()
    played = []
    real = gate.run_tournament
    monkeypatch.setattr(gate, "run_tournament", lambda agents, *a, **k: played.append(agents) or real(agents, *a, **k))
    gate.run_gate(cand, champ, games=4, workers=1, out_dir=tmp_path)
    assert played and ["basic", "intermediate"] not in played  # champion baseline came from cache


def test_gate_and_ladder_play_one_candidate_seat_vs_three(monkeypatch):
    """Regression: [cand, h] was filled cyclically into cand,h,cand,h (a 2v2 team game)."""
    import catan.zoo.gate as gate
    import catan.zoo.ladder as ladder
    seen = []

    def fake(agents, games, **kw):
        seen.append(list(agents))
        raise RuntimeError("stop")

    monkeypatch.setattr(gate, "run_tournament", fake)
    monkeypatch.setattr(ladder, "run_tournament", fake)
    import pytest
    cand = {"id": "c", "spec": "random", "kind": "rl"}
    champ = {"id": "intermediate", "spec": "intermediate", "kind": "anchor"}
    with pytest.raises(RuntimeError):
        gate.run_gate(cand, champ, games=4, workers=1, out_dir=None)
    with pytest.raises(RuntimeError):
        ladder.point("random", "basic", 4, 1)
    assert seen and all(a.count(a[0]) == 1 and len(a) == 4 for a in seen), seen


def test_mixed_champion_table_plays_every_seating(monkeypatch):
    """Regression: cyclic rotation always seated the candidate right before the champion (seat-order bias)."""
    seen = {}

    def fake(agents, games, **kw):
        seen[tuple(agents)] = kw.get("rotation", "cyclic")
        n = kw.get("num_players", 4)
        return {"games": games, "per_agent": {a: {"wins": 0} for a in agents}, "results": [],
                "agents": list(dict.fromkeys(agents)), "lineups": [], "num_players": n}

    monkeypatch.setattr(gate, "run_tournament", fake)
    monkeypatch.setattr(gate, "_wins", lambda t, spec: {})
    cand = {"id": "c", "spec": "random", "kind": "rl"}
    champ = {"id": "k", "spec": "basic", "kind": "rl"}
    try:
        gate.run_gate(cand, champ, games=4, workers=1, out_dir=None)
    except Exception:
        pass
    assert seen.get(("random", "basic", "intermediate", "intermediate")) == "all", seen


def test_add_checkpoint_same_weights_other_agent_is_new_entry(run_dir):
    """Regression (D11): dedup by sha256 alone reused the arch_rl entry when registering the same weights as combo."""
    reg = {"entries": []}
    a = registry.add_checkpoint(reg, "runs/r1/checkpoints/ckpt_00001.pt", "r1", root=run_dir)
    assert registry.add_checkpoint(reg, "runs/r1/checkpoints/ckpt_00001.pt", "r1", root=run_dir) is a
    b = registry.add_checkpoint(reg, "runs/r1/checkpoints/ckpt_00001.pt", "r1c", root=run_dir, agent="combo")
    assert b is not a and b["spec"].startswith("combo:")
