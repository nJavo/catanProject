"""EN-008: create_app(infoset_agents=True) hands AI seats Game.observe_for(seat)."""
from fastapi.testclient import TestClient

from catan.server.app import create_app

from test_app import data  # noqa: F401 (fixture)


def _play(data, infoset):
    c = TestClient(create_app(data_dir=data, static_dir=data / "nodist", infoset_agents=infoset))
    seats = [{"type": "agent", "agent": a} for a in ("basic", "intermediate", "random")]
    gid = c.post("/api/games", json={"players": seats, "seed": 9}).json()["id"]
    c.post(f"/api/games/{gid}/advance?steps=300")
    return c.get(f"/api/games/{gid}/replay").json()


def test_infoset_mode_plays_the_same_game_without_agent_errors(data):
    full, view = _play(data, False), _play(data, True)
    assert view["actions"] == full["actions"] and len(view["actions"]) > 100
    infos = [d.get("info") or {} for d in view["meta"]["decisions"].values()]
    assert not any("agent_error" in i or "explain_error" in i for i in infos)
    assert sum("explain" in i for i in infos) > 50  # explanations also run on the view
