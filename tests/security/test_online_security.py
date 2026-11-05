"""--online mode: room/seat tokens, admin key, no private info across seats (docs/ONLINE.md)."""
import json
import random

import pytest
from fastapi.testclient import TestClient

from catan.server import online as online_mod
from catan.server.app import create_app

HUMAN = {"type": "human"}
RANDOM = {"type": "agent", "agent": "random"}
KEY = "admin-key-for-tests-0123456789"


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("CATAN_ADMIN_KEY", KEY)
    return TestClient(create_app(data_dir=tmp_path, static_dir=tmp_path / "nodist", online=True))


def room(c, players=(HUMAN, HUMAN, RANDOM), seed=7):
    r = c.post("/api/rooms", json={"players": list(players), "seed": seed}, headers={"X-Catan-Admin": KEY})
    assert r.status_code == 200, r.text
    return r.json()


def tok(t):
    return {"X-Room-Token": t}


def test_missing_or_wrong_token_is_403(client):
    r = room(client)
    gid = r["id"]
    wrong = "x" * 32
    for h in ({}, tok(wrong), tok(""), {"X-Catan-Admin": "nope"}):
        assert client.get(f"/api/games/{gid}?viewer=0", headers=h).status_code == 403
        assert client.get(f"/api/rooms/{gid}", headers=h).status_code == 403
        assert client.post(f"/api/games/{gid}/advance", headers=h).status_code == 403
        assert client.post(f"/api/games/{gid}/actions", json={"action": {"player": 0, "type": "ROLL"}},
                           headers=h).status_code == 403
    # a token of another room does not open this one
    other = room(client)
    assert client.get(f"/api/games/{gid}", headers=tok(other["tokens"][0])).status_code == 403


def test_admin_routes_need_admin_key(client):
    for method, path in [("get", "/api/games"), ("get", "/api/runs"), ("get", "/api/saves"), ("get", "/api/agents"),
                         ("get", "/api/replays"), ("get", "/api/tournaments"),
                         ("post", "/api/rooms"), ("post", "/api/games"), ("post", "/api/lab/matches"),
                         ("post", "/api/analysis/validate"), ("get", "/api/replay-state?source=file&ref=x")]:
        kw = {"json": {}} if method == "post" else {}
        assert getattr(client, method)(path, **kw).status_code == 403, path
    # a seat token is not an admin key
    t = room(client)["tokens"][0]
    assert client.get("/api/runs", headers=tok(t)).status_code == 403
    assert client.get("/api/runs", headers={"X-Catan-Admin": KEY}).status_code == 200
    assert client.get("/api/health").json() == {"ok": True, "online": True}


def test_seat_token_cannot_act_or_look_for_another_seat(client):
    r = room(client)
    gid, t0, t1 = r["id"], r["tokens"][0], r["tokens"][1]
    v = client.get(f"/api/games/{gid}?viewer=1", headers=tok(t0)).json()   # asks for seat 1, gets seat 0
    assert v["viewer"] == 0 and v["actor"] == 0 and v["legal_actions"]
    assert client.get(f"/api/games/{gid}?viewer=all", headers=tok(t0)).json()["viewer"] == 0
    a = v["legal_actions"][0]
    # seat 1 tries to play seat 0's move
    assert client.post(f"/api/games/{gid}/actions", json={"action": a}, headers=tok(t1)).status_code == 403
    assert client.post(f"/api/games/{gid}/actions", json={"action": a}, headers=tok(r["host_token"])).status_code == 403
    assert client.post(f"/api/games/{gid}/actions", json={"action": a}, headers=tok(t0)).status_code == 200
    # host-only and admin sub-routes are closed to a seat
    assert client.post(f"/api/games/{gid}/save", json={}, headers=tok(t0)).status_code == 403
    assert client.post(f"/api/games/{gid}/replay/save", headers=tok(t0)).status_code == 403
    # only the host link lists the seat links
    assert "tokens" not in client.get(f"/api/rooms/{gid}", headers=tok(t0)).json()
    assert client.get(f"/api/rooms/{gid}", headers=tok(r["host_token"])).json()["tokens"] == r["tokens"]


def _play(client, r, rng, actions=400):
    """Random legal play through every seat's own link; yields (seat, response json) for each call."""
    gid = r["id"]
    toks = r["tokens"]
    for _ in range(actions):
        pub = client.get(f"/api/games/{gid}", headers=tok(r["host_token"])).json()
        if pub["phase"] == "GAME_OVER":
            return
        if pub["awaiting"] == "agent":
            seat = rng.choice([i for i, t in enumerate(toks) if t])
            res = client.post(f"/api/games/{gid}/advance?steps=3&viewer=all", headers=tok(toks[seat]))
            assert res.status_code == 200
            yield seat, res.json()
            continue
        seat = pub["actor"]
        v = client.get(f"/api/games/{gid}", headers=tok(toks[seat])).json()
        yield seat, v
        res = client.post(f"/api/games/{gid}/actions", json={"action": rng.choice(v["legal_actions"])},
                          headers=tok(toks[seat]))
        assert res.status_code == 200, res.text
        yield seat, res.json()


def _assert_private(seat, doc):
    views = [doc.get("view", doc)] if "players" in doc.get("view", doc) else []
    for v in views:
        assert v["viewer"] == seat
        for q, p in enumerate(v["players"]):
            if q != seat:
                assert not {"hand", "dev_cards", "dev_new", "vp"} & set(p), (seat, q, p)
        assert v["last_decision"] is None
    evs = list(doc.get("events", [])) + [e for v in views for e in v["events"]]
    evs += [e for st in doc.get("steps", []) for e in st["events"]]
    for e in evs:
        assert e.get("visible_to") is None or seat in e["visible_to"], (seat, e)


def test_no_private_info_reaches_another_seat(client, monkeypatch):
    monkeypatch.setattr(online_mod, "RATE_LIMIT", 10**6)  # the test client is far faster than a browser
    r = room(client, seed=11)
    seen_private = 0
    for seat, doc in _play(client, r, random.Random(3)):
        _assert_private(seat, doc)
        seen_private += sum(1 for st in doc.get("steps", []) for e in st["events"] if e.get("visible_to"))
        seen_private += sum(1 for e in doc.get("view", doc).get("events", []) if e.get("visible_to"))
    # the game got far enough to produce private events, which then reached only their owners
    assert seen_private > 0
    # and the host link is a spectator: no hand at all
    v = client.get(f"/api/games/{r['id']}", headers=tok(r["host_token"])).json()
    assert v["viewer"] is None and all("hand" not in p for p in v["players"])


def test_room_rate_and_size_limits(client, monkeypatch):
    r = room(client)
    gid, t0 = r["id"], r["tokens"][0]
    big = {"action": {"player": 0, "type": "ROLL"}, "pad": "x" * (online_mod.MAX_ROOM_BODY + 1)}
    assert client.post(f"/api/games/{gid}/actions", json=big, headers=tok(t0)).status_code == 413
    # chunked body without Content-Length gets the same cap
    gen = (b for b in [json.dumps(big).encode()])
    assert client.post(f"/api/games/{gid}/actions", content=gen,
                       headers={**tok(t0), "Content-Type": "application/json"}).status_code == 413
    assert client.post(f"/api/rooms/{gid}/name", json={"name": "x" * 25}, headers=tok(t0)).status_code == 400
    assert client.post(f"/api/rooms/{gid}/name", json={"name": "a\nb"}, headers=tok(t0)).status_code == 400
    monkeypatch.setattr(online_mod, "RATE_LIMIT", 5)
    codes = [client.get(f"/api/games/{gid}", headers=tok(r["tokens"][1])).status_code for _ in range(8)]
    assert codes == [200] * 5 + [429] * 3
    # one seat hitting the limit does not lock out another
    assert client.get(f"/api/games/{gid}", headers=tok(r["host_token"])).status_code == 200


def test_offline_mode_unchanged(tmp_path):
    c = TestClient(create_app(data_dir=tmp_path, static_dir=tmp_path / "nodist", online=False))
    assert c.get("/api/health").json() == {"ok": True, "online": False}
    assert c.post("/api/rooms", json={"players": [HUMAN, RANDOM]}).status_code in (404, 405)
    gid = c.post("/api/games", json={"players": [HUMAN, RANDOM], "seed": 1}).json()["id"]
    assert c.get(f"/api/games/{gid}?viewer=0").json()["viewer"] == 0
    assert c.get("/api/runs").status_code == 200
