"""Online rooms: creation, lobby, names, restart persistence (catan/server/online.py)."""
import pytest
from fastapi.testclient import TestClient

from catan.server.app import create_app
from catan.server.online import Room, admin_key

HUMAN = {"type": "human"}
RANDOM = {"type": "agent", "agent": "random"}
KEY = "admin-key-for-tests-0123456789"
ADMIN = {"X-Catan-Admin": KEY}


@pytest.fixture
def make(tmp_path, monkeypatch):
    monkeypatch.setenv("CATAN_ADMIN_KEY", KEY)
    return lambda: TestClient(create_app(data_dir=tmp_path, static_dir=tmp_path / "nodist", online=True))


def test_room_tokens_are_long_and_distinct(make):
    c = make()
    r = c.post("/api/rooms", json={"players": [HUMAN, RANDOM, HUMAN], "seed": 1}, headers=ADMIN).json()
    assert r["tokens"][1] is None
    toks = [r["host_token"], r["tokens"][0], r["tokens"][2]]
    assert len(set(toks)) == 3 and all(len(t) >= 32 for t in toks)   # 24 random bytes = 192 bits
    assert c.post("/api/rooms", json={"players": [RANDOM, RANDOM]}, headers=ADMIN).status_code == 400


def test_lobby_names_and_connection(make):
    c = make()
    r = c.post("/api/rooms", json={"players": [HUMAN, HUMAN, RANDOM], "seed": 1}, headers=ADMIN).json()
    gid, t0 = r["id"], r["tokens"][0]
    lob = c.get(f"/api/rooms/{gid}", headers={"X-Room-Token": t0}).json()
    assert lob["you"] == 0
    assert [s["connected"] for s in lob["seats"]] == [True, False, True]   # seat 1 never opened its link
    assert c.post(f"/api/rooms/{gid}/name", json={"name": "  Ana "}, headers={"X-Room-Token": t0}).json() == {"name": "Ana"}
    assert c.post(f"/api/rooms/{gid}/name", json={"name": "Host"},
                  headers={"X-Room-Token": r["host_token"]}).status_code == 400
    v = c.get(f"/api/games/{gid}", headers={"X-Room-Token": r["tokens"][1]}).json()
    assert v["seats"][0]["name"] == "Ana" and "name" not in v["seats"][1]


def test_room_survives_restart(make, tmp_path):
    c = make()
    r = c.post("/api/rooms", json={"players": [HUMAN, RANDOM], "seed": 5}, headers=ADMIN).json()
    gid, t0 = r["id"], {"X-Room-Token": r["tokens"][0]}
    c.post(f"/api/rooms/{gid}/name", json={"name": "Ana"}, headers=t0)
    v = c.get(f"/api/games/{gid}", headers=t0).json()
    c.post(f"/api/games/{gid}/actions", json={"action": v["legal_actions"][0]}, headers=t0)
    before = c.get(f"/api/games/{gid}", headers=t0).json()
    assert (tmp_path / "results" / "saves" / f"room-{gid}.json").is_file()
    (tmp_path / "results" / "saves" / "room-torn.json").write_text("{")   # ignored, not fatal

    c2 = make()   # a fresh process over the same data dir
    after = c2.get(f"/api/games/{gid}", headers=t0).json()
    assert after["action_count"] == before["action_count"] == 1
    assert after["players"][0]["hand"] == before["players"][0]["hand"]
    assert after["seats"][0]["name"] == "Ana"
    assert c2.get(f"/api/games/{gid}").status_code == 403


def test_room_unit_rate_window_and_admin_key_file(tmp_path, monkeypatch):
    room = Room("h" * 32, ["a" * 32, None])
    assert room.who("h" * 32) == "host" and room.who("a" * 32) == 0 and room.who(None) is None and room.who("") is None
    assert all(room.allow("t", 100.0) for _ in range(60)) and not room.allow("t", 100.1)
    assert room.allow("t", 105.0)   # new window
    monkeypatch.delenv("CATAN_ADMIN_KEY", raising=False)
    k = admin_key(tmp_path)
    assert len(k) >= 32 and admin_key(tmp_path) == k
    assert (tmp_path / "results" / "online_admin_key").stat().st_mode & 0o077 == 0
