import json
import multiprocessing as mp
import shutil
import threading
import time

import pytest
from fastapi.testclient import TestClient

from catan.analysis import engine, position_from_game
from catan.engine import Phase
from catan.server.app import create_app

from .helpers import CKPT, play_until


@pytest.fixture(scope="module")
def mid():
    g = play_until(3, lambda g: g.turn >= 30 and g.phase == Phase.MAIN and g.actor == 0)
    return g, position_from_game(g, 0)


@pytest.fixture
def data(tmp_path, mid):
    (tmp_path / "results" / "replays").mkdir(parents=True)
    g = mid[0]
    (tmp_path / "results" / "replays" / "r1.json").write_text(json.dumps(
        {"seed": g.seed, "config": g.config.to_dict(), "agents": ["intermediate"] * 4,
         "actions": [a.to_json() for a in g.action_log]}))
    (tmp_path / "runs" / "x" / "checkpoints").mkdir(parents=True)
    (tmp_path / "outside.pt").write_bytes(b"x")
    if CKPT is not None:
        shutil.copy(CKPT, tmp_path / "runs" / "x" / "checkpoints" / "c.pt")
    return tmp_path


@pytest.fixture
def client(data):
    return TestClient(create_app(data_dir=data, static_dir=data / "nodist"))


def test_validate_and_legal(client, mid):
    g, spec = mid
    r = client.post("/api/analysis/validate", json={"position": spec}).json()
    assert r["ok"] and r["errors"] == [] and r["derived"]["actor"] == 0
    bad = json.loads(json.dumps(spec))
    bad["bank"][0] += 1
    r = client.post("/api/analysis/validate", json={"position": bad}).json()
    assert not r["ok"] and any("conservation" in e["message"] for e in r["errors"]) and r["derived"] is None
    assert all(set(e) == {"message", "vertices", "edges", "hexes"} for e in r["errors"])
    from catan.engine import TOPOLOGY as T
    v = next(v for v in range(54) if spec["vertex_owner"][v] == 0)
    u = T.vertex_neighbors[v][0]
    bad2 = json.loads(json.dumps(spec))
    bad2["vertex_owner"][u], bad2["vertex_level"][u] = 1, 1
    errs = client.post("/api/analysis/validate", json={"position": bad2}).json()["errors"]
    hit = next(e for e in errs if "distance rule" in e["message"])
    assert {u, v} <= set(hit["vertices"])
    r = client.post("/api/analysis/legal", json={"position": spec})
    assert r.status_code == 200
    legal = r.json()["legal"]
    assert len(legal) == len(g.legal_actions()) and all(isinstance(a["label"], str) for a in legal)
    assert client.post("/api/analysis/legal", json={"position": bad}).status_code == 400


def test_evaluate_position(client, mid):
    r = client.post("/api/analysis/evaluate", json={"position": mid[1], "policy": "intermediate", "depth": "quick"})
    assert r.status_code == 200, r.text
    body = r.json()
    for k in ("position", "actor", "phase", "legal_count", "policy", "rollout_policy", "depth",
              "position_eval", "candidates", "determinization", "timing_ms"):
        assert k in body
    assert body["determinization"]["assumptions"]
    assert client.post("/api/analysis/evaluate", json={"position": mid[1], "depth": "insane"}).status_code == 400
    assert client.post("/api/analysis/evaluate", json={"position": mid[1], "policy": "nope"}).status_code == 400
    assert client.post("/api/analysis/evaluate",
                       json={"position": mid[1], "policy": "search:rollouts=100000"}).status_code == 400


def test_evaluate_replay_source(client, mid):
    g = mid[0]
    r = client.post("/api/analysis/evaluate", json={"source": {"kind": "replay", "ref": "r1.json"},
                                                    "viewer": 0, "depth": "quick"})
    assert r.status_code == 200, r.text
    assert r.json()["position"] == mid[1]
    # an earlier index; viewer who is not to move gets a clear 400
    r = client.post("/api/analysis/evaluate", json={"source": {"kind": "replay", "ref": "r1.json", "index": len(g.action_log)},
                                                    "viewer": 2, "depth": "quick"})
    assert r.status_code == 400 and "viewer" in r.json()["detail"]
    assert client.post("/api/analysis/evaluate", json={"source": {"kind": "replay", "ref": "../x.json"},
                                                       "viewer": 0}).status_code == 400
    assert client.post("/api/analysis/evaluate", json={"source": {"kind": "replay", "ref": "missing.json"},
                                                       "viewer": 0}).status_code == 404


def test_live_game_hidden_info_rules(client):
    gid = client.post("/api/games", json={"players": [{"type": "human"}] + [{"type": "agent", "agent": "basic"}] * 3,
                                          "seed": 5}).json()["id"]
    # seat 0 (human) must place a settlement: analysable from seat 0 only
    r = client.post("/api/analysis/evaluate", json={"source": {"kind": "game", "ref": gid}, "viewer": 0, "depth": "quick"})
    assert r.status_code == 200, r.text
    assert r.json()["phase"] == "SETUP_SETTLEMENT"
    r = client.post("/api/analysis/evaluate", json={"source": {"kind": "game", "ref": gid}, "viewer": 1, "depth": "quick"})
    assert r.status_code == 403
    r = client.post("/api/analysis/game", json={"source": {"kind": "game", "ref": gid}, "viewer": 2})
    assert r.status_code == 403
    assert client.post("/api/analysis/evaluate", json={"source": {"kind": "game", "ref": "nope"},
                                                       "viewer": 0}).status_code == 404


def test_ai_game_source_and_timeline(client):
    m = client.post("/api/lab/matches", json={"agents": ["basic"] * 4, "seed": 3, "max_turns": 20,
                                              "run_to_end": True}).json()
    r = client.post("/api/analysis/game", json={"source": {"kind": "game", "ref": m["id"]}, "viewer": 2,
                                                "policy": "intermediate", "depth": "quick"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["summary"]["decisions"] == len(body["timeline"]) > 0
    t = body["timeline"][0]
    assert set(t["eval_before"]) >= {"win_rate", "ci95", "n", "value_estimate"}
    for k in ("index", "actor", "played", "played_label", "best", "best_label", "eval_before", "eval_after",
              "delta", "verdict"):
        assert k in t


def test_checkpoint_restrictions(client, data, mid):
    r = client.post("/api/analysis/evaluate", json={"position": mid[1], "policy": f"rl:checkpoint={data / 'outside.pt'}",
                                                    "depth": "quick"})
    assert r.status_code == 400 and "runs/" in r.json()["detail"]
    r = client.post("/api/analysis/evaluate", json={"position": mid[1], "policy": "rl:checkpoint=runs/../outside.pt",
                                                    "depth": "quick"})
    assert r.status_code == 400
    if CKPT is not None:
        r = client.post("/api/analysis/evaluate", json={"position": mid[1], "depth": "quick",
                                                        "policy": "rl:checkpoint=runs/x/checkpoints/c.pt"})
        assert r.status_code == 200, r.text
        assert r.json()["candidates"][0]["policy_prob"] is not None


def test_continue(client, mid):
    r = client.post("/api/analysis/continue", json={"position": mid[1], "action": {"player": 0, "type": "END_TURN"},
                                                    "policy": "basic", "until": "turn_end", "seed": 2})
    assert r.status_code == 200, r.text
    body = r.json()
    assert set(body) >= {"position", "events", "done"} and body["position"]["turn"] == mid[1]["turn"] + 1
    assert client.post("/api/analysis/continue", json={"position": mid[1], "action": {"player": 0, "type": "ROLL"}}
                       ).status_code == 400
    assert client.post("/api/analysis/continue", json={"position": mid[1], "action": "x"}).status_code == 400


def test_time_limit_returns_partial(client, mid):
    r = client.post("/api/analysis/evaluate", json={"position": mid[1], "depth": "deep", "time_limit_ms": 100})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["timed_out"] is True
    assert all(c["rollout"] is None or c["rollout"]["n"] < 200 for c in body["candidates"])


def test_checkpoints_lists_models(client, data):
    (data / "models").mkdir()
    (data / "models" / "m.pt").write_bytes(b"x")
    assert "models/m.pt" in client.get("/api/checkpoints").json()["checkpoints"]


# ---------------------------------------------------------------- AN-007 cancellation / concurrency
def _deep_in_background(client, spec, rid, out):
    t = threading.Thread(target=lambda: out.append(client.post("/api/analysis/evaluate", json={
        "position": spec, "depth": "deep", "request_id": rid, "time_limit_ms": 60000})))
    t.start()
    return t


def _wait_pool(timeout=30):
    """True once a deep analysis has reached its rollout phase (pool created)."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        if engine._POOL is not None:
            return True
        time.sleep(0.05)
    return False


@pytest.fixture
def two_workers(monkeypatch):
    engine.shutdown_pool()  # a pool left by an earlier test would fool _wait_pool
    # 2 workers (CPU budget in tests); 400 rollouts per candidate so a deep run outlasts any test step
    monkeypatch.setitem(engine.DEPTHS, "deep", {**engine.DEPTHS["deep"], "workers": 2, "rollouts": 400})
    yield
    engine.shutdown_pool()


def test_cancel_frees_capacity_and_workers(client, mid, two_workers):
    _, spec = mid
    out = []
    t = _deep_in_background(client, spec, "deep-1", out)
    assert _wait_pool(), "deep analysis never started its pool"
    t0 = time.time()
    assert client.post("/api/analysis/cancel", json={"request_id": "deep-1"}).json() == {"cancelled": True}
    t.join(10)
    assert out and out[0].status_code == 409 and "cancelled" in out[0].text
    # AN-007 target is ~1 s on an idle machine; the budget covers thread scheduling under heavy load
    assert time.time() - t0 < 5, "cancel took too long"
    # workers are gone (no orphans) and the slot is free again
    for _ in range(40):
        if not mp.active_children():
            break
        time.sleep(0.05)
    assert not mp.active_children() and engine._POOL is None
    r = client.post("/api/analysis/evaluate", json={"position": spec, "depth": "standard", "time_limit_ms": 300})
    assert r.status_code == 200
    # unknown / malformed ids
    assert client.post("/api/analysis/cancel", json={"request_id": "nope"}).json() == {"cancelled": False}
    # a cancel that overtakes its request still stops it
    r = client.post("/api/analysis/evaluate", json={"position": spec, "depth": "standard", "request_id": "nope"})
    assert r.status_code == 409
    assert client.post("/api/analysis/cancel", json={"request_id": "../x"}).status_code == 400
    assert client.post("/api/analysis/evaluate", json={"position": spec, "request_id": 5}).status_code == 400


class _HeldLock:
    """Hold an engine lock from another thread, as a running analysis does."""
    def __init__(self, lock):
        self.lock, self.go, self.release = lock, threading.Event(), threading.Event()
        self.t = threading.Thread(target=self._hold, daemon=True)

    def _hold(self):
        with self.lock:
            self.go.set()
            self.release.wait(120)

    def __enter__(self):
        self.t.start()
        assert self.go.wait(30)
        return self

    def __exit__(self, *exc):
        self.release.set()
        self.t.join(30)


def test_quick_does_not_wait_for_deep(client, mid):
    """Deterministic: the rollout lock is held for the whole test; quick must still answer."""
    _, spec = mid
    out = []
    with _HeldLock(engine._LOCK) as held:
        t = threading.Thread(target=lambda: out.append(client.post("/api/analysis/evaluate", json={
            "position": spec, "depth": "quick"})))
        t.start()
        t.join(120)  # generous: a loaded machine is slow, but quick must not need the lock at all
        assert not held.release.is_set() and out, "quick waited for the rollout lock"
        assert out[0].status_code == 200 and out[0].json()["candidates"]


def test_cancel_while_waiting_for_the_lock(client, mid, monkeypatch):
    """A request queued behind a running analysis is cancelled before it runs.
    Synchronised on the engine's lock-wait hook, not on sleeps."""
    _, spec = mid
    waiting = threading.Event()
    monkeypatch.setattr(engine, "_on_lock_wait", waiting.set)
    out = []
    with _HeldLock(engine._LOCK):
        t = threading.Thread(target=lambda: out.append(client.post("/api/analysis/evaluate", json={
            "position": spec, "depth": "standard", "request_id": "queued"})))
        t.start()
        assert waiting.wait(60), "request never reached the lock"
        assert client.post("/api/analysis/cancel", json={"request_id": "queued"}).json()["cancelled"]
        t.join(30)  # it gives up within CANCEL_POLL_S while the lock is still held
        assert out and out[0].status_code == 409 and "cancelled" in out[0].text


# ---------------------------------------------------------------- AN-006 uncertainty fields
def test_uncertainty_fields(mid):
    _, spec = mid
    r = engine.analyze(spec, depth="standard", samples=3, rollouts=4, top_k=2, workers=1)
    for c in r["candidates"]:
        ro = c["rollout"]
        assert ro["samples"] == 3 and ro["between_sample_sd"] is not None and 0 <= ro["between_sample_sd"] <= 1
        assert c["value_estimate"] is None and c["value_estimate_calibrated"] is None  # heuristic policy: no value
    assert r["position_eval"]["rollout"]["between_sample_sd"] is not None
    one = engine.analyze(spec, depth="standard", samples=1, rollouts=4, top_k=1, workers=1)
    ro = one["candidates"][0]["rollout"]
    assert ro["between_sample_sd"] is None and "fewer than 2" in ro["between_sample_sd_reason"]
    quick = engine.analyze(spec, depth="quick")
    assert all(c["rollout"] is None for c in quick["candidates"])  # no rollouts: fields absent with the block


def test_between_sample_sd_formula():
    rows = lambda wins, n: [(1, 0, 10)] * wins + [(0, 0, 5)] * (n - wins)  # noqa: E731
    b = [rows(4, 4), rows(0, 4)]  # determinizations disagree completely
    st = engine._rollout_stats(b[0] + b[1], b)
    assert st["win_rate"] == 0.5 and abs(st["between_sample_sd"] - 0.7071) < 1e-3
    same = [rows(2, 4), rows(2, 4)]
    assert engine._rollout_stats(same[0] + same[1], same)["between_sample_sd"] == 0


def test_search_agents_refused_as_rollout_policy(client, mid):
    # a search per move of every simulated game: minutes per batch (the champion hung the UI)
    for p in ("jointb_gated", "search"):
        r = client.post("/api/analysis/evaluate", json={"position": mid[1], "rollout_policy": p, "depth": "standard"})
        assert r.status_code == 400 and "too slow" in r.json()["detail"], r.text


def test_deadline_stops_running_workers(client, mid, two_workers):
    # at the deadline the in-flight rollouts are killed, not left running behind the next request
    r = client.post("/api/analysis/evaluate", json={"position": mid[1], "depth": "deep", "time_limit_ms": 500})
    assert r.status_code == 200 and r.json()["timed_out"] is True
    for _ in range(40):
        if not mp.active_children():
            break
        time.sleep(0.05)
    assert engine._POOL is None and not mp.active_children()
