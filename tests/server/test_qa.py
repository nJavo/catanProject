"""QA regression tests: save/resume, races, eviction, bad payloads."""
import json
import threading

from fastapi.testclient import TestClient

from catan.engine import Game
from catan.server import app as app_mod
from catan.server.app import create_app

from test_app import HUMAN, RANDOM, act, client, data, new_game, play_to_main  # noqa: F401 (fixtures)


def test_rl_default_checkpoint_falls_back_to_latest_run(tmp_path, monkeypatch):
    """The rl agent's built-in default (models/ppo_baseline.pt) is absent on a
    fresh clone, so "Default checkpoint" in Setup/Lab failed to create the game."""
    import os
    import time as _t
    from catan.rl.policy_agent import DEFAULT_CHECKPOINT
    if os.path.isfile(DEFAULT_CHECKPOINT):
        return  # the built-in default exists here: nothing to fall back from
    for i, run in enumerate(("old_run", "new_run")):
        d = tmp_path / "runs" / run / "checkpoints"
        d.mkdir(parents=True)
        (d / "latest.pt").write_bytes(b"not a real checkpoint")
        os.utime(d / "latest.pt", (_t.time() + i, _t.time() + i))
    seen, real = {}, app_mod.make_agent

    def fake_make(name, **kw):  # the fake file can't load: record what the server asked for
        seen.update(kw)
        return real("random")
    monkeypatch.setattr(app_mod, "make_agent", fake_make)
    c = TestClient(create_app(data_dir=tmp_path, static_dir=tmp_path / "none"))
    assert c.get("/api/checkpoints").json()["default"] == "runs/new_run/checkpoints/latest.pt"
    r = c.post("/api/games", json={"players": [HUMAN, {"type": "agent", "agent": "rl"}]})
    assert r.status_code == 200 and seen["checkpoint"].endswith("new_run/checkpoints/latest.pt")


def test_save_and_resume_roundtrip(client, data):
    gid = new_game(client, [HUMAN, RANDOM, RANDOM], seed=21)
    for _ in range(4):  # through setup: human takes the first legal spot, AIs advance
        v = client.post(f"/api/games/{gid}/advance?steps=200&viewer=0").json()["view"]
        if v["phase"] in ("SETUP_SETTLEMENT", "SETUP_ROAD"):
            act(client, gid, v["legal_actions"][0])
    act(client, gid, {"player": 0, "type": "ROLL"})
    before = client.get(f"/api/games/{gid}?viewer=0").json()
    r = client.post(f"/api/games/{gid}/save", json={"name": "qa-1"})
    assert r.status_code == 200 and r.json() == {"name": "qa-1"}
    assert (data / "results" / "saves" / "qa-1.json").is_file()
    lst = client.get("/api/saves").json()["saves"]
    assert lst[0]["name"] == "qa-1" and lst[0]["turn"] == before["turn"]
    assert "game" not in lst[0]  # listing never exposes hands or the deck
    new = client.post("/api/saves/qa-1/resume").json()
    assert new["id"] != gid and [s["type"] for s in new["seats"]] == ["human", "agent", "agent"]
    after = client.get(f"/api/games/{new['id']}?viewer=0").json()
    for k in ("phase", "turn", "bank", "vertex_owner", "edge_owner", "robber", "legal_actions", "actor"):
        assert after[k] == before[k], k
    assert after["players"][0]["hand"] == before["players"][0]["hand"]
    # same hidden-info rules as a live game
    assert "hand" not in after["players"][1]
    assert client.get(f"/api/games/{new['id']}?viewer=all").status_code == 403
    assert client.get(f"/api/games/{new['id']}?viewer=1").status_code == 403
    assert any(e["type"] == "game_resumed" for e in after["events"] + client.get(
        f"/api/games/{new['id']}?viewer=0").json()["events"])
    # the resumed game plays on: AI seats advance
    assert client.post(f"/api/games/{new['id']}/advance?steps=5&viewer=0").status_code == 200


def test_resumed_game_is_deterministic(client):
    """Same RNG state: the next roll after resume matches the original game."""
    gid = new_game(client, [HUMAN, HUMAN], seed=8)
    play_to_main(client, gid, 2)
    client.post(f"/api/games/{gid}/save", json={"name": "det"})
    new = client.post("/api/saves/det/resume").json()["id"]
    r1 = act(client, gid, {"player": 0, "type": "ROLL"}).json()["view"]
    r2 = act(client, new, {"player": 0, "type": "ROLL"}).json()["view"]
    assert r1["last_roll"] == r2["last_roll"] and r1["players"][0]["hand"] == r2["players"][0]["hand"]


def test_save_names_are_traversal_safe(client, data):
    gid = new_game(client, [HUMAN, RANDOM])
    for bad in ("../x", "a/b", "..", "", "x.json", "/etc/passwd", "a b", 5):
        r = client.post(f"/api/games/{gid}/save", json={"name": bad})
        assert r.status_code in (400, 200), bad
        if bad == "":
            assert r.status_code == 200  # empty -> default timestamped name
        else:
            assert r.status_code == 400, bad
    (data / "secret.json").write_text(json.dumps({"format": "catan-save"}))
    for bad in ("..%2Fsecret", "%2e%2e%2fsecret", "..", "nope"):
        r = client.post(f"/api/saves/{bad}/resume")
        assert r.status_code in (400, 404), (bad, r.status_code)
    assert not list(data.glob("*.tmp")) and not (data / "x.json").exists()


def test_corrupt_save_rejected(client, data):
    d = data / "results" / "saves"
    d.mkdir(parents=True, exist_ok=True)
    (d / "broken.json").write_text(json.dumps({"format": "catan-save", "seats": [], "game": {"seed": 1}}))
    (d / "junk.json").write_text("{not json")
    assert client.post("/api/saves/broken/resume").status_code == 422
    assert client.post("/api/saves/junk/resume").status_code == 404
    names = [s["name"] for s in client.get("/api/saves").json()["saves"]]
    assert "junk" not in names


def test_concurrent_actions_are_serialized(client):
    """Double-submit: many threads post the same ROLL; exactly one succeeds."""
    gid = new_game(client, [HUMAN, HUMAN], seed=3)
    play_to_main(client, gid, 2)
    codes = []

    def roll():
        codes.append(act(client, gid, {"player": 0, "type": "ROLL"}).status_code)
    ts = [threading.Thread(target=roll) for _ in range(12)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert codes.count(200) == 1 and codes.count(400) == 11
    v = client.get(f"/api/games/{gid}?viewer=0").json()
    assert sum(1 for e in v["events"] if e["type"] == "dice_rolled") == 1


def test_concurrent_advance_and_actions_stay_consistent(client, data):
    gid = new_game(client, [HUMAN, RANDOM, RANDOM, RANDOM], seed=17)
    stop = threading.Event()
    errors = []

    def spam_advance():
        while not stop.is_set():
            r = client.post(f"/api/games/{gid}/advance?steps=3&viewer=0")
            if r.status_code != 200:
                errors.append(r.text)

    ts = [threading.Thread(target=spam_advance) for _ in range(4)]
    for t in ts:
        t.start()
    try:
        for _ in range(400):
            v = client.get(f"/api/games/{gid}?viewer=0").json()
            if v["phase"] == "GAME_OVER" or v["turn"] > 15:
                break
            if v["actor"] == 0 and v["legal_actions"]:
                a = next((x for x in v["legal_actions"] if x["type"] in ("END_TURN", "ROLL")), v["legal_actions"][0])
                act(client, gid, a)  # may race with nothing; 400 is fine, 500 is not
    finally:
        stop.set()
        for t in ts:
            t.join()
    assert not errors
    assert client.post(f"/api/games/{gid}/save", json={"name": "race"}).status_code == 200
    d = json.loads((data / "results" / "saves" / "race.json").read_text())
    # whatever interleaving happened, the action log re-simulates to the same state
    g = Game.from_dict(d["game"])
    assert len(g.action_log) > 20
    h = Game.replay(g.seed, g.config, g.action_log)
    assert (h.vertex_owner, h.bank, h.phase, h.turn) == (g.vertex_owner, g.bank, g.phase, g.turn)


def test_eviction_keeps_live_human_games(tmp_path, monkeypatch):
    monkeypatch.setattr(app_mod, "MAX_SESSIONS", 5)
    c = TestClient(create_app(data_dir=tmp_path, static_dir=tmp_path / "none"))
    live = new_game(c, [HUMAN, RANDOM], seed=1)
    done = [c.post("/api/lab/matches", json={"agents": ["random"] * 2, "seed": i, "max_turns": 50,
                                               "run_to_end": True}).json()["id"] for i in range(4)]
    for i in range(6):  # more lab matches than slots
        c.post("/api/lab/matches", json={"agents": ["random"] * 2, "seed": 100 + i})
        c.get(f"/api/games/{live}")  # the human keeps playing
    ids = {g["id"] for g in c.get("/api/games").json()["games"]}
    assert live in ids and len(ids) == 5
    assert not ids & set(done)  # finished games went first


def test_finished_sessions_are_capped(tmp_path, monkeypatch):
    monkeypatch.setattr(app_mod, "MAX_FINISHED", 3)
    c = TestClient(create_app(data_dir=tmp_path, static_dir=tmp_path / "none"))
    ids = [c.post("/api/lab/matches", json={"agents": ["random"] * 2, "seed": i, "max_turns": 30,
                                              "run_to_end": True}).json()["id"] for i in range(6)]
    live = c.post("/api/lab/matches", json={"agents": ["random"] * 2, "seed": 99}).json()["id"]
    kept = {g["id"] for g in c.get("/api/games").json()["games"]}
    # room is made before each insert, so at most MAX_FINISHED - 1 finished games survive it
    assert kept == set(ids[-2:]) | {live}


def test_wrong_phase_and_bad_payloads(client):
    gid = new_game(client, [HUMAN, HUMAN], seed=5)
    play_to_main(client, gid, 2)
    bad = [
        {"player": 0, "type": "END_TURN"},                      # before rolling
        {"player": 0, "type": "BUILD_ROAD", "value": 0},        # PRE_ROLL
        {"player": 0, "type": "OFFER_TRADE", "value": [[1, 0, 0, 0, 0], [0, 1, 0, 0, 0]]},  # pre-roll trade
        {"player": 1, "type": "ROLL"},                          # wrong player
        {"player": 0, "type": "DISCARD", "value": 0},
        {"player": 0, "type": "MOVE_ROBBER", "value": 3},
        {"player": 0, "type": "PLAY_MONOPOLY", "value": 9},
        {"player": 0, "type": "PLAY_YEAR_OF_PLENTY", "value": [0]},
        {"player": 0, "type": "ROLL", "value": [[[[1]]]]},
        {"player": -1, "type": "ROLL"}, {"player": 99, "type": "ROLL"},
        {"player": 0.0, "type": "ROLL"}, {"player": True, "type": "ROLL"},
        {"player": 0, "type": None}, {"type": "ROLL"}, [], None, 7,
    ]
    for a in bad:
        r = client.post(f"/api/games/{gid}/actions", json={"action": a})
        assert r.status_code == 400, (a, r.status_code, r.text)
    assert client.post(f"/api/games/{gid}/actions", content=b"not json",
                       headers={"Content-Type": "application/json"}).status_code in (400, 422)
    v = client.get(f"/api/games/{gid}?viewer=0").json()
    assert v["phase"] == "PRE_ROLL" and v["action_count"] == 8
