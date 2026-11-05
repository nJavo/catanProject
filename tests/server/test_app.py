import json

import pytest
from fastapi.testclient import TestClient

from catan.engine import Game, GameConfig
from catan.server.app import create_app


@pytest.fixture
def data(tmp_path):
    (tmp_path / "results" / "replays").mkdir(parents=True)
    (tmp_path / "results" / "tournaments").mkdir(parents=True)
    run = tmp_path / "runs" / "run1"
    (run / "checkpoints").mkdir(parents=True)
    (run / "meta.json").write_text(json.dumps({"run_id": "run1"}))
    (run / "metrics.jsonl").write_text("\n".join(json.dumps({"step": i, "time": i, "win_rate": i / 10})
                                                 for i in range(5)) + "\nnot json\n")
    (run / "checkpoints" / "c1.pt").write_bytes(b"x")
    (tmp_path / "results" / "tournaments" / "t1.json").write_text(json.dumps({"id": "t1", "per_agent": {}}))
    (tmp_path / "secret.json").write_text(json.dumps({"seed": 1, "actions": []}))
    return tmp_path


@pytest.fixture
def client(data):
    return TestClient(create_app(data_dir=data, static_dir=data / "nodist"))


def new_game(client, players, seed=5):
    r = client.post("/api/games", json={"players": players, "seed": seed})
    assert r.status_code == 200, r.text
    return r.json()["id"]


HUMAN = {"type": "human"}
RANDOM = {"type": "agent", "agent": "random"}


def act(client, gid, a):
    return client.post(f"/api/games/{gid}/actions", json={"action": a})


def test_create_and_legal_flow(client):
    gid = new_game(client, [HUMAN, RANDOM, RANDOM])
    v = client.get(f"/api/games/{gid}?viewer=0").json()
    assert v["phase"] == "SETUP_SETTLEMENT" and v["actor"] == 0
    assert v["legal_actions"] and all(a["type"] == "BUILD_SETTLEMENT" for a in v["legal_actions"])
    r = act(client, gid, v["legal_actions"][0])
    assert r.status_code == 200
    v = r.json()["view"]
    assert v["phase"] == "SETUP_ROAD"
    r = act(client, gid, v["legal_actions"][0])
    assert r.json()["view"]["actor"] == 1
    # AI seats advance; stop when human must act again (second settlement, snake order)
    r = client.post(f"/api/games/{gid}/advance?steps=100&viewer=0").json()
    assert len(r["steps"]) == 8 and r["view"]["actor"] == 0 and r["view"]["legal_actions"]


def test_illegal_and_malformed_rejected(client):
    gid = new_game(client, [HUMAN, RANDOM])
    assert act(client, gid, {"player": 0, "type": "ROLL"}).status_code == 400
    assert act(client, gid, {"player": 1, "type": "BUILD_SETTLEMENT", "value": 0}).status_code == 400
    assert act(client, gid, {"player": 0, "type": "NOPE"}).status_code == 400
    assert act(client, gid, {"player": "0", "type": "ROLL"}).status_code == 400
    assert act(client, gid, {"player": 0, "type": "BUILD_SETTLEMENT", "value": 999}).status_code == 400
    assert act(client, gid, {"player": 0, "type": "BUILD_SETTLEMENT", "value": [1, [2]]}).status_code == 400
    assert act(client, gid, "x").status_code == 400
    assert client.post(f"/api/games/{gid}/actions", json={}).status_code == 400
    # an AI seat cannot be driven by the client
    v = client.get(f"/api/games/{gid}?viewer=0").json()
    act(client, gid, v["legal_actions"][0])
    v = client.get(f"/api/games/{gid}?viewer=0").json()
    act(client, gid, v["legal_actions"][0])
    g_view = client.get(f"/api/games/{gid}").json()
    assert g_view["actor"] == 1
    assert act(client, gid, {"player": 1, "type": "BUILD_SETTLEMENT", "value": 0}).status_code == 400


def test_bad_create(client):
    assert client.post("/api/games", json={"players": [HUMAN]}).status_code == 400
    assert client.post("/api/games", json={"players": [HUMAN, {"type": "agent", "agent": "nope"}]}).status_code == 400
    assert client.post("/api/games", json={"players": [HUMAN, {"type": "alien"}]}).status_code == 400
    assert client.post("/api/games", json={"players": [HUMAN, HUMAN], "seed": "x"}).status_code == 400


def play_to_main(client, gid, n):
    """Drive a hot-seat game through setup with legal actions."""
    for _ in range(4 * n):
        v = client.get(f"/api/games/{gid}").json()
        if v["phase"] not in ("SETUP_SETTLEMENT", "SETUP_ROAD"):
            return
        a = client.get(f"/api/games/{gid}?viewer={v['actor']}").json()["legal_actions"][0]
        assert act(client, gid, a).status_code == 200


def test_hidden_info_not_leaked(client):
    gid = new_game(client, [HUMAN, HUMAN, HUMAN], seed=3)
    play_to_main(client, gid, 3)
    v0 = client.get(f"/api/games/{gid}?viewer=0").json()
    v1 = client.get(f"/api/games/{gid}?viewer=1").json()
    assert "hand" in v0["players"][0] and "hand" not in v0["players"][1] and "dev_cards" not in v0["players"][2]
    assert "hand" in v1["players"][1] and "hand" not in v1["players"][0]
    assert v1["legal_actions"] == [] and v0["legal_actions"]
    pub = client.get(f"/api/games/{gid}").json()
    assert all("hand" not in p for p in pub["players"])
    # full reveal refused while humans play
    assert client.get(f"/api/games/{gid}?viewer=all").status_code == 403
    # replay of a live human game refused (it would reveal the dev deck via re-simulation)
    assert client.get(f"/api/games/{gid}/replay").status_code == 403


def test_private_events_filtered():
    from catan.server.app import Session, Seat, _view
    g = Game(seed=1, num_players=2)
    sess = Session("x", g, [Seat("human", None, {}), Seat("human", None, {})], "play")
    sess.events.append({"seq": 1, "i": 0, "type": "dev_bought_detail", "player": 0, "card": "knight",
                        "visible_to": [0]})
    sess.events.append({"seq": 2, "i": 0, "type": "dev_bought", "player": 0})
    assert any(e["type"] == "dev_bought_detail" for e in _view(sess, 0)["events"])
    assert not any(e["type"] == "dev_bought_detail" for e in _view(sess, 1)["events"])
    assert not any(e["type"] == "dev_bought_detail" for e in _view(sess, None)["events"])


def test_viewer_restrictions(client):
    gid = new_game(client, [HUMAN, RANDOM])
    assert client.get(f"/api/games/{gid}?viewer=1").status_code == 403
    assert client.get(f"/api/games/{gid}?viewer=9").status_code == 400
    assert client.get(f"/api/games/{gid}?viewer=abc").status_code == 400
    assert client.get("/api/games/nope").status_code == 404


def test_ai_match_to_completion_and_replay_determinism(client, data):
    r = client.post("/api/lab/matches", json={"agents": ["random"] * 4, "seed": 9, "max_turns": 300,
                                              "run_to_end": True}).json()
    assert r["result"] is not None
    gid = r["id"]
    v = client.get(f"/api/games/{gid}?viewer=all").json()
    assert v["phase"] == "GAME_OVER" and all("hand" in p for p in v["players"])
    rep = client.get(f"/api/games/{gid}/replay").json()
    g = Game.replay(rep["seed"], GameConfig(**rep["config"]), rep["actions"])
    assert g.winner == v["winner"] and g.turn == v["turn"]
    assert [p["vp"] for p in v["players"]] == [g.victory_points(p) for p in range(4)]
    # decisions recorded for every AI action
    assert len(rep["meta"]["decisions"]) == len(rep["actions"])
    # save to file, list, step through
    name = client.post(f"/api/games/{gid}/replay/save").json()["name"]
    lst = client.get("/api/replays").json()["replays"]
    assert any(x["name"] == name for x in lst)
    st = client.get(f"/api/replay-state?source=file&ref={name}&index=20").json()
    assert st["index"] == 20 and st["total_actions"] == len(rep["actions"])
    h = Game.replay(rep["seed"], GameConfig(**rep["config"]), rep["actions"][:20])
    assert st["vertex_owner"] == h.vertex_owner and st["edge_owner"] == h.edge_owner
    end = client.get(f"/api/replay-state?source=game&ref={gid}&index=999999").json()
    assert end["phase"] == "GAME_OVER"
    # cached forward stepping and backward scrubbing match a fresh re-simulation
    for k in (30, 45, 10, 46):
        st = client.get(f"/api/replay-state?source=file&ref={name}&index={k}").json()
        h = Game.replay(rep["seed"], GameConfig(**rep["config"]), rep["actions"][:k])
        assert (st["vertex_owner"], st["bank"], st["phase"]) == (h.vertex_owner, h.bank, h.phase)
        assert [p["hand"] for p in st["players"]] == [pl.hand for pl in h.players]


def test_advance_live_lab(client):
    r = client.post("/api/lab/matches", json={"agents": ["random", "random"], "seed": 2}).json()
    v = client.post(f"/api/games/{r['id']}/advance?steps=5&viewer=all").json()
    assert len(v["steps"]) == 5 and v["view"]["action_count"] == 5
    assert "info" in v["view"]["last_decision"]


def test_path_traversal_rejected(client):
    for bad in ("..", "../secret.json", "..%2Fsecret.json", "%2e%2e%2fsecret.json", "a/b.json", ".hidden/../x"):
        r = client.get(f"/api/replays/{bad}")
        assert r.status_code in (400, 404), (bad, r.status_code)
        assert "actions" not in r.text
    r = client.get("/api/replay-state", params={"source": "file", "ref": "../secret.json"})
    assert r.status_code == 400
    r = client.post("/api/games", json={"players": [HUMAN, {"type": "agent", "agent": "random",
                                                            "kwargs": {"checkpoint": "../secret.json"}}]})
    assert r.status_code == 400
    r = client.post("/api/games", json={"players": [HUMAN, {"type": "agent", "agent": "random",
                                                            "kwargs": {"checkpoint": "/etc/passwd"}}]})
    assert r.status_code == 400


def test_lab_artifacts(client, data):
    runs = client.get("/api/runs").json()["runs"]
    assert runs[0]["id"] == "run1" and runs[0]["metrics_total"] == 6 and runs[0]["checkpoints"] == 1
    run = client.get("/api/runs/run1").json()  # rows live in the per-run detail (tests/server/test_runs_api.py)
    assert len(run["metrics"]) == 5 and run["checkpoints"] == ["c1.pt"]
    assert client.get("/api/tournaments").json()["tournaments"][0]["id"] == "t1"
    # ladders rendered as empty tournament cards: they belong to the Leaderboard tab only
    (data / "results" / "tournaments" / "ladder_x.json").write_text(json.dumps({"kind": "ladder", "run": "x"}))
    assert [t["id"] for t in client.get("/api/tournaments").json()["tournaments"]] == ["t1"]
    assert "random" in client.get("/api/agents").json()["agents"]
    assert client.get("/api/checkpoints").json()["checkpoints"] == ["runs/run1/checkpoints/c1.pt"]
    assert len(client.get("/api/topology").json()["vertices"]) == 54


def test_missing_dirs(tmp_path):
    c = TestClient(create_app(data_dir=tmp_path, static_dir=tmp_path / "none"))
    assert c.get("/api/runs").json() == {"runs": []}
    assert c.get("/api/tournaments").json() == {"tournaments": []}
    assert c.get("/api/replays").json() == {"replays": []}


def test_domestic_trade_with_ai_responses(client):
    gid = new_game(client, [HUMAN, RANDOM, RANDOM], seed=11)
    # finish setup
    for _ in range(40):
        v = client.post(f"/api/games/{gid}/advance?steps=50&viewer=0").json()["view"]
        if v["phase"] not in ("SETUP_SETTLEMENT", "SETUP_ROAD"):
            break
        act(client, gid, v["legal_actions"][0])
    v = client.get(f"/api/games/{gid}?viewer=0").json()
    assert v["phase"] == "PRE_ROLL" and v["actor"] == 0
    v = act(client, gid, {"player": 0, "type": "ROLL"}).json()["view"]
    while v["actor"] != 0 or v["phase"] != "MAIN":
        if v["actor"] == 0:
            v = act(client, gid, v["legal_actions"][0]).json()["view"]
        else:
            v = client.post(f"/api/games/{gid}/advance?viewer=0").json()["view"]
    hand = v["players"][0]["hand"]
    give_r = max(range(5), key=lambda r: hand[r])
    give = [0] * 5
    give[give_r] = 1
    get = [0] * 5
    get[(give_r + 1) % 5] = 1
    # arbitrary (non-template) amounts allowed if well-formed; gifts rejected
    assert act(client, gid, {"player": 0, "type": "OFFER_TRADE", "value": [give, [0] * 5]}).status_code == 400
    r = act(client, gid, {"player": 0, "type": "OFFER_TRADE", "value": [give, get]})
    assert r.status_code == 200 and r.json()["view"]["phase"] == "TRADE_RESPONSE"
    # a human responder never receives enumerated counters (they encode the proposer's hand)
    gid2 = new_game(client, [HUMAN, HUMAN], seed=11)
    play_to_main(client, gid2, 2)
    act(client, gid2, {"player": 0, "type": "ROLL"})
    v = client.get(f"/api/games/{gid2}?viewer=0").json()
    assert v["phase"] == "MAIN"
    offer = next(a for a in v["legal_actions"] if a["type"] == "OFFER_TRADE")
    act(client, gid2, offer)
    v1 = client.get(f"/api/games/{gid2}?viewer=1").json()
    assert v1["phase"] == "TRADE_RESPONSE"
    assert not any(a["type"] == "COUNTER_OFFER" for a in v1["legal_actions"])
    v = client.post(f"/api/games/{gid}/advance?steps=10&viewer=0").json()["view"]
    assert v["phase"] in ("TRADE_CONFIRM", "MAIN") and v["actor"] == 0
