"""EN-006: the server writes versioned replays and rejects unknown versions with 422."""
import json

from catan.engine.replay import REPLAY_FORMAT

from test_app import RANDOM, client, data, new_game  # noqa: F401 (fixtures)


def _put(data, name, doc):
    (data / "results" / "replays" / name).write_text(json.dumps(doc))


def test_saved_replay_is_versioned(client, data):
    gid = new_game(client, [RANDOM, RANDOM, RANDOM], seed=3)
    client.post(f"/api/games/{gid}/advance?steps=50")
    r = client.get(f"/api/games/{gid}/replay").json()
    assert r["format"] == REPLAY_FORMAT and r["version"] == 1 and r["rules_digest"]
    name = client.post(f"/api/games/{gid}/replay/save").json()["name"]
    assert client.get(f"/api/replays/{name}").json()["version"] == 1


def test_legacy_replay_loads_and_unknown_version_is_422(client, data):
    legacy = {"seed": 7, "config": {"num_players": 4}, "agents": ["random"] * 4, "actions": [], "meta": {}}
    _put(data, "old.json", legacy)
    _put(data, "future.json", dict(legacy, format=REPLAY_FORMAT, version=99))
    r = client.get("/api/replays/old.json")
    assert r.status_code == 200 and r.json()["version"] == 1
    assert client.get("/api/replay-state?source=file&ref=old.json&index=0").status_code == 200
    assert client.get("/api/replays/future.json").status_code == 422
    assert client.get("/api/replay-state?source=file&ref=future.json&index=0").status_code == 422
    r = client.post("/api/analysis/game", json={"source": {"kind": "replay", "ref": "future.json"}, "viewer": 0})
    assert r.status_code == 422, r.text
