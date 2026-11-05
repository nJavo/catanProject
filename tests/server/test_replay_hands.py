"""Full-hand reveal in replays / AI-only matches, and refusal for live human games."""
from fastapi.testclient import TestClient
import pytest

from catan.engine import Game, GameConfig
from catan.server.app import create_app


@pytest.fixture
def client(tmp_path):
    (tmp_path / "results" / "replays").mkdir(parents=True)
    return TestClient(create_app(data_dir=tmp_path, static_dir=tmp_path / "nodist"))


def finished_ai_game(client):
    r = client.post("/api/lab/matches", json={"agents": ["random"] * 3, "seed": 4, "max_turns": 200,
                                              "run_to_end": True}).json()
    return r["id"], client.get(f"/api/games/{r['id']}/replay").json()


def test_replay_state_reveals_every_seat_and_previous_hands(client):
    gid, rep = finished_ai_game(client)
    name = client.post(f"/api/games/{gid}/replay/save").json()["name"]
    cfg = GameConfig(**rep["config"])
    for source, ref in (("file", name), ("game", gid)):
        for k in (0, 1, 30, 31, 90, len(rep["actions"])):
            st = client.get("/api/replay-state", params={"source": source, "ref": ref, "index": k}).json()
            g = Game.replay(rep["seed"], cfg, rep["actions"][:k])
            for p, pl in zip(st["players"], g.players):
                assert p["hand"] == pl.hand and p["dev_cards"] == pl.dev and p["dev_new"] == pl.dev_new
                assert p["vp"] == g.victory_points(p["id"])
            if k == 0:
                assert st["prev_players"] is None
            else:
                before = Game.replay(rep["seed"], cfg, rep["actions"][:k - 1])
                assert [q["hand"] for q in st["prev_players"]] == [pl.hand for pl in before.players]
                assert [q["dev_cards"] for q in st["prev_players"]] == [pl.dev for pl in before.players]
    # backwards scrub (cache invalidation) still matches
    st = client.get("/api/replay-state", params={"source": "file", "ref": name, "index": 5}).json()
    assert [p["hand"] for p in st["players"]] == [pl.hand for pl in Game.replay(rep["seed"], cfg, rep["actions"][:5]).players]


def test_ai_only_live_match_full_view(client):
    gid = client.post("/api/lab/matches", json={"agents": ["random", "random"], "seed": 2}).json()["id"]
    v = client.post(f"/api/games/{gid}/advance?steps=40&viewer=all").json()["view"]
    assert v["phase"] != "GAME_OVER"
    assert all({"hand", "dev_cards", "dev_new", "vp"} <= p.keys() for p in v["players"])


def test_live_human_game_refuses_reveal_and_replay_state(client):
    gid = client.post("/api/games", json={"players": [{"type": "human"}, {"type": "agent", "agent": "random"}],
                                          "seed": 5}).json()["id"]
    assert client.get(f"/api/games/{gid}?viewer=all").status_code == 403
    assert client.post(f"/api/games/{gid}/advance?viewer=all").status_code == 403
    r = client.get("/api/replay-state", params={"source": "game", "ref": gid, "index": 0})
    assert r.status_code == 403
    assert "hand" not in r.text
    assert client.post(f"/api/games/{gid}/replay/save").status_code == 403
