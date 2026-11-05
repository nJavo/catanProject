"""Security regression tests for catan/server (docs/SECURITY.md)."""
import json
import threading
import time

import pytest
from fastapi.testclient import TestClient

from catan.agents.base import Agent
from catan.agents.registry import register
from catan.engine import Game, GameConfig
from catan.server import analysis as analysis_mod
from catan.server import app as app_mod
from catan.server.app import create_app

HUMAN = {"type": "human"}
RANDOM = {"type": "agent", "agent": "random"}


@pytest.fixture
def data(tmp_path):
    for d in ("results/replays", "results/saves", "results/behavior", "results/tournaments", "runs/r1/checkpoints",
              "models"):
        (tmp_path / d).mkdir(parents=True)
    (tmp_path / "secret.json").write_text(json.dumps({"seed": 1, "actions": [], "format": "catan-save", "agents": {}}))
    return tmp_path


@pytest.fixture
def client(data):
    return TestClient(create_app(data_dir=data, static_dir=data / "nodist"))


def new_game(c, players, seed=1):
    r = c.post("/api/games", json={"players": players, "seed": seed})
    assert r.status_code == 200, r.text
    return r.json()["id"]


# ------------------------------------------------------------ host / origin / CORS
def test_dns_rebinding_host_refused(client):
    assert client.get("/api/health", headers={"Host": "evil.example"}).status_code == 400
    assert client.post("/api/games", json={"players": [HUMAN, RANDOM]}, headers={"Host": "evil.example:8000"}).status_code == 400
    for ok in ("127.0.0.1:8000", "localhost", "[::1]:8000", "LOCALHOST:1"):
        assert client.get("/api/health", headers={"Host": ok}).status_code == 200, ok


def test_allowed_hosts_env(data, monkeypatch):
    monkeypatch.setenv("CATAN_ALLOWED_HOSTS", "catan.lan")
    c = TestClient(create_app(data_dir=data, static_dir=data / "x"))
    assert c.get("/api/health", headers={"Host": "catan.lan:8000"}).status_code == 200
    assert c.get("/api/health", headers={"Host": "other.lan"}).status_code == 400
    monkeypatch.setenv("CATAN_ALLOWED_HOSTS", "*")
    c = TestClient(create_app(data_dir=data, static_dir=data / "x"))
    assert c.get("/api/health", headers={"Host": "anything"}).status_code == 200


def test_cross_origin_writes_refused(client):
    body = {"players": [HUMAN, RANDOM]}
    evil = {"Origin": "https://evil.example"}
    assert client.post("/api/games", json=body, headers=evil).status_code == 403
    assert client.post("/api/games", json=body, headers={"Origin": "null"}).status_code == 403
    # another local app on a different port is a different origin too
    assert client.post("/api/games", json=body, headers={"Origin": "http://testserver:3000"}).status_code == 403
    assert client.post("/api/games", json=body, headers={"Origin": "http://testserver"}).status_code == 200
    assert client.post("/api/games", json=body).status_code == 200  # non-browser clients send no Origin
    # a form/text "simple" request (no preflight) is not parsed as JSON even without Origin
    r = client.post("/api/games", content=json.dumps(body), headers={"Content-Type": "text/plain"})
    assert r.status_code == 422


def test_no_cors_headers(client):
    r = client.get("/api/games", headers={"Origin": "https://evil.example"})
    assert "access-control-allow-origin" not in {k.lower() for k in r.headers}
    r = client.options("/api/games", headers={"Origin": "https://evil.example", "Access-Control-Request-Method": "POST"})
    assert "access-control-allow-origin" not in {k.lower() for k in r.headers}


def test_advisor_extension_origin_only_on_import_image(client, data, monkeypatch):
    from catan.server.guard import ADVISOR_ORIGIN
    ext = {"Origin": ADVISOR_ORIGIN}
    files = {"image": ("a.gif", b"GIF89a", "image/gif")}
    r = client.post("/api/analysis/import-image", files=files, headers=ext)
    assert r.status_code == 415 and r.headers["access-control-allow-origin"] == ADVISOR_ORIGIN  # passed the guard
    r = client.post("/api/analysis/import-image", files=files, headers={"Origin": "chrome-extension://" + "a" * 32})
    assert r.status_code == 403 and "access-control-allow-origin" not in r.headers  # another extension
    r = client.post("/api/games", json={"players": [HUMAN, RANDOM]}, headers=ext)
    assert r.status_code == 403 and "access-control-allow-origin" not in r.headers  # any other path
    assert "access-control-allow-origin" not in client.get("/api/games", headers=ext).headers
    monkeypatch.setenv("CATAN_EXTENSION_ORIGINS", "chrome-extension://" + "b" * 32 + ",https://evil.example")
    c = TestClient(create_app(data_dir=data, static_dir=data / "x"))
    assert c.post("/api/analysis/import-image", files=files, headers={"Origin": "chrome-extension://" + "b" * 32}).status_code == 415
    assert c.post("/api/analysis/import-image", files=files, headers={"Origin": "https://evil.example"}).status_code == 403


# ------------------------------------------------------------ body limits
def test_body_size_cap(client):
    big = b'{"players": [' + b'{"type": "human"},' * 70000 + b'{"type": "human"}]}'
    assert len(big) > app_mod.MAX_BODY
    r = client.post("/api/games", content=big, headers={"Content-Type": "application/json"})
    assert r.status_code == 413

    def chunks():  # no Content-Length: chunked transfer
        for _ in range(30):
            yield b" " * 50_000
    r = client.post("/api/games", content=chunks(), headers={"Content-Type": "application/json"})
    assert r.status_code == 413
    assert client.get("/api/health").status_code == 200


def test_deeply_nested_json_does_not_kill_server(client):
    deep = b"[" * 200_000 + b"]" * 200_000
    r = client.post("/api/games", content=deep, headers={"Content-Type": "application/json"})
    assert r.status_code >= 400
    assert client.get("/api/health").status_code == 200


# ------------------------------------------------------------ agent kwargs / checkpoints
def test_agent_kwargs_allowlisted(client):
    bad = [{"rollouts": 10**9}, {"rollouts": True}, {"time_limit": 1e9}, {"device": "cuda:0"}, {"weights": "x"},
           {"deterministic": "yes"}, {"checkpoint": 5}]
    for kw in bad:
        r = client.post("/api/games", json={"players": [HUMAN, {"type": "agent", "agent": "search", "kwargs": kw}]})
        assert r.status_code == 400, kw
        r = client.post("/api/lab/matches", json={"agents": [{"agent": "search", "kwargs": kw}, "random"]})
        assert r.status_code == 400, kw
    ok = client.post("/api/games", json={"players": [HUMAN, {"type": "agent", "agent": "search", "kwargs": {"rollouts": 4}}]})
    assert ok.status_code == 200


def test_checkpoint_paths_confined(client, data, monkeypatch, tmp_path_factory):
    outside = tmp_path_factory.mktemp("outside") / "evil.pt"
    outside.write_bytes(b"x")
    (data / "runs" / "r1" / "checkpoints" / "link.pt").symlink_to(outside)
    (data / "models" / "m.pt").write_bytes(b"x")
    seen = []
    real = app_mod.make_agent
    monkeypatch.setattr(app_mod, "make_agent", lambda name, **kw: (seen.append(kw), real("random"))[1])
    for bad in ("../secret.json", str(outside), "runs/../secret.json", "runs/r1/checkpoints/link.pt",
                "runs/r1/checkpoints/missing.pt", "/etc/passwd", "models/../secret.json"):
        r = client.post("/api/games", json={"players": [HUMAN, {"type": "agent", "agent": "rl", "kwargs": {"checkpoint": bad}}]})
        assert r.status_code == 400, bad
        r = client.post("/api/analysis/evaluate", json={"position": {}, "policy": f"rl:checkpoint={bad}"})
        assert r.status_code == 400, bad
    r = client.post("/api/games", json={"players": [HUMAN, {"type": "agent", "agent": "rl", "kwargs": {"checkpoint": "models/m.pt"}}]})
    assert r.status_code == 200 and seen[-1]["checkpoint"] == str((data / "models" / "m.pt").resolve())


def test_named_file_endpoints_confined(client):
    bad_names = ["..", "../secret.json", "..%2Fsecret.json", "%2e%2e%2fsecret.json", ".hidden", "a/b",
                 "%00", "x" * 300]
    for n in bad_names:
        for url in (f"/api/replays/{n}", f"/api/behavior/{n}", f"/api/saves/{n}/resume"):
            r = client.post(url) if url.endswith("resume") else client.get(url)
            assert r.status_code in (400, 404, 405, 422), (url, r.status_code)
            assert '"seed"' not in r.text
        r = client.get("/api/replay-state", params={"source": "file", "ref": n})
        assert r.status_code in (400, 404, 422), n
        r = client.post("/api/analysis/game", json={"source": {"kind": "replay", "ref": n}, "viewer": 0})
        assert r.status_code in (400, 404), n
    for n in ("../x", "a/b", ".x", "x" * 65, "", "a b"):
        gid = new_game(client, [HUMAN, RANDOM])
        r = client.post(f"/api/games/{gid}/save", json={"name": n})
        assert r.status_code == (200 if n == "" else 400), n


def test_save_count_capped(client, monkeypatch):
    monkeypatch.setattr(app_mod, "MAX_SAVES", 2)
    gid = new_game(client, [HUMAN, RANDOM])
    assert client.post(f"/api/games/{gid}/save", json={"name": "a"}).status_code == 200
    assert client.post(f"/api/games/{gid}/save", json={"name": "b"}).status_code == 200
    assert client.post(f"/api/games/{gid}/save", json={"name": "c"}).status_code == 507
    assert client.post(f"/api/games/{gid}/save", json={"name": "a"}).status_code == 200  # overwrite is fine


# ------------------------------------------------------------ hidden information
class Buyer(Agent):
    """Buys a dev card whenever it can, otherwise rolls or ends the turn."""
    def act(self, game, legal):
        for t in ("BUY_DEV", "ROLL", "END_TURN"):
            for a in legal:
                if a.type == t:
                    return a
        return legal[0]


register("qa_buyer", lambda **kw: Buyer())


def _resume_crafted(client, data, seats):
    g = Game(config=GameConfig(num_players=2), seed=7)
    while g.phase in ("SETUP_SETTLEMENT", "SETUP_ROAD"):
        g.step(g.legal_actions()[0])
    for r in (2, 3, 4):
        g.players[0].hand[r] += 3
        g.bank[r] -= 3
    doc = {"format": "catan-save", "version": 1, "name": "x", "saved": 0, "source": "play", "seats": seats,
           "game": g.to_dict(), "events": [{"seq": 0, "i": -1, "type": "game_started"}], "decisions": {}}
    (data / "results" / "saves" / "crafted.json").write_text(json.dumps(doc))
    return client.post("/api/saves/crafted/resume").json()["id"]


def test_advance_and_view_do_not_leak_ai_private_info(client, data):
    gid = _resume_crafted(client, data, [{"type": "agent", "agent": "qa_buyer", "kwargs": {}},
                                         {"type": "human", "agent": None, "kwargs": {}}])
    r = client.post(f"/api/games/{gid}/advance?steps=50&viewer=1").json()
    evs = [e for s in r["steps"] for e in s["events"]]
    assert any(e["type"] == "dev_bought" for e in evs), "precondition: the AI bought a card"
    assert not any(e.get("visible_to") and 1 not in e["visible_to"] for e in evs), "AI's private events leaked"
    assert r["view"]["last_decision"] is None  # agent diagnostics rank its own (hidden) options
    spect = client.post(f"/api/games/{gid}/advance?steps=1").json()
    assert all(not e.get("visible_to") for s in spect["steps"] for e in s["events"])
    assert client.get(f"/api/games/{gid}?viewer=1").json()["last_decision"] is None
    # AI-only games keep the inspector
    lab = client.post("/api/lab/matches", json={"agents": ["basic", "basic"], "seed": 2}).json()["id"]
    assert client.post(f"/api/games/{lab}/advance?steps=3&viewer=all").json()["view"]["last_decision"]


def test_live_human_game_not_revealed_anywhere(client):
    gid = new_game(client, [HUMAN, RANDOM, RANDOM])
    client.post(f"/api/games/{gid}/advance?steps=10&viewer=0")
    assert client.get(f"/api/games/{gid}/replay").status_code == 403
    assert client.post(f"/api/games/{gid}/replay/save").status_code == 403
    assert client.get("/api/replay-state", params={"source": "game", "ref": gid}).status_code == 403
    assert client.get(f"/api/games/{gid}?viewer=all").status_code == 403
    assert client.get(f"/api/games/{gid}?viewer=1").status_code == 403
    assert client.post(f"/api/games/{gid}/advance?viewer=all").status_code == 403
    r = client.post("/api/analysis/evaluate", json={"source": {"kind": "game", "ref": gid}, "viewer": 1})
    assert r.status_code == 403
    lst = json.dumps(client.get("/api/games").json())
    assert '"hand"' not in lst and '"dev_cards"' not in lst
    assert '"game"' not in json.dumps(client.get("/api/saves").json())


# ------------------------------------------------------------ analysis limits
def test_analysis_concurrency_capped(client, monkeypatch):
    gate = threading.Event()
    started = threading.Semaphore(0)

    def slow(*a, **kw):
        started.release()
        gate.wait(10)
        return {"ok": True}
    monkeypatch.setattr(analysis_mod, "analyze", slow)
    codes = []
    ts = [threading.Thread(target=lambda: codes.append(client.post("/api/analysis/evaluate", json={"position": {}}).status_code))
          for _ in range(2)]
    for t in ts:
        t.start()
    assert started.acquire(timeout=5)
    time.sleep(0.3)
    assert client.post("/api/analysis/evaluate", json={"position": {}}).status_code == 429
    gate.set()
    for t in ts:
        t.join()
    assert codes == [200, 200]
    assert client.get("/api/health").status_code == 200


def test_analysis_bounds(client):
    gid = client.post("/api/lab/matches", json={"agents": ["random", "random"], "seed": 1, "max_turns": 30,
                                                "run_to_end": True}).json()["id"]
    r = client.post("/api/analysis/game", json={"source": {"kind": "game", "ref": gid}, "viewer": 0, "depth": "deep"})
    assert r.status_code == 400
    for body in ({"time_limit_ms": 10**9}, {"top_k": 10**6}, {"policy": "search:rollouts=100000"},
                 {"policy": "search:time_limit=99"}, {"depth": "ultra"}, {"seed": -1}):
        r = client.post("/api/analysis/evaluate", json={"position": {}, **body})
        assert r.status_code == 400, body
    r = client.post("/api/analysis/continue", json={"position": {}, "action": {"player": 0, "type": "ROLL"},
                                                    "steps": 10**7})
    assert r.status_code == 400


def test_server_cli_warns_on_public_bind(monkeypatch, capsys):
    import catan.server.__main__ as m
    calls = []
    monkeypatch.setattr(m.uvicorn, "run", lambda *a, **kw: calls.append(kw))
    monkeypatch.setattr("sys.argv", ["x", "--host", "0.0.0.0"])
    monkeypatch.delenv("CATAN_ALLOWED_HOSTS", raising=False)
    m.main()
    assert "no authentication" in capsys.readouterr().err.lower() and calls
    monkeypatch.setattr("sys.argv", ["x"])
    m.main()
    assert capsys.readouterr().err == ""


def test_one_checkpoint_rule_for_seats_lab_and_analysis(client, data, monkeypatch, tmp_path_factory):
    """REG-004: a models/ checkpoint works for a game seat, a lab match and an analysis
    policy; outside paths and symlink escapes are refused by all three."""
    (data / "models" / "x.pt").write_bytes(b"x")
    outside = tmp_path_factory.mktemp("elsewhere") / "evil.pt"
    outside.write_bytes(b"x")
    (data / "models" / "link.pt").symlink_to(outside)
    real = app_mod.make_agent
    monkeypatch.setattr(app_mod, "make_agent", lambda name, **kw: real("random"))  # fake bytes can't load
    seen = []
    monkeypatch.setattr(analysis_mod, "analyze", lambda spec, policy, *a, **k: seen.append(policy) or {"ok": True})
    rl = lambda ck: {"agent": "rl", "kwargs": {"checkpoint": ck}}  # noqa: E731
    for ck, ok in (("models/x.pt", True), ("models/link.pt", False), (str(outside), False),
                   ("models/../secret.json", False)):
        want = 200 if ok else 400
        assert client.post("/api/games", json={"players": [HUMAN, {"type": "agent", **rl(ck)}]}).status_code == want, ck
        assert client.post("/api/lab/matches", json={"agents": [rl(ck), "random"]}).status_code == want, ck
        r = client.post("/api/analysis/evaluate", json={"position": {}, "policy": f"rl:checkpoint={ck}"})
        assert r.status_code == want, (ck, r.text)
    assert seen and seen[0].endswith("models/x.pt")
