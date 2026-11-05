import json

import pytest
from fastapi.testclient import TestClient

from catan.server.app import create_app


@pytest.fixture
def client(tmp_path):
    d = tmp_path / "results" / "behavior"
    d.mkdir(parents=True)
    (d / "p1.json").write_text(json.dumps({"id": "p1", "vs": "basic", "agents": {"basic": {"games": 2}}}))
    (d / "broken.json").write_text("{not json")
    (d / "notprofile.json").write_text(json.dumps([1, 2]))
    (tmp_path / "results" / "secret.json").write_text(json.dumps({"agents": {}}))
    return TestClient(create_app(data_dir=tmp_path, static_dir=tmp_path / "nodist"))


def test_list_skips_unreadable(client):
    r = client.get("/api/behavior")
    assert r.status_code == 200
    profs = r.json()["profiles"]
    assert [p["file"] for p in profs] == ["p1.json"]
    assert profs[0]["agents"] == ["basic"] and profs[0]["vs"] == "basic"


def test_get_profile(client):
    assert client.get("/api/behavior/p1.json").json()["agents"]["basic"]["games"] == 2
    assert client.get("/api/behavior/missing.json").status_code == 404
    assert client.get("/api/behavior/broken.json").status_code == 422


@pytest.mark.parametrize("name", ["..%2Fsecret.json", "%2E%2E%2Fsecret.json", "secret", ".hidden.json",
                                  "a%00.json", "..%5Csecret.json"])
def test_path_safety(client, name):
    assert client.get(f"/api/behavior/{name}").status_code in (400, 404)


def test_empty_dir(tmp_path):
    c = TestClient(create_app(data_dir=tmp_path, static_dir=tmp_path / "nodist"))
    assert c.get("/api/behavior").json() == {"profiles": []}


def test_titles_name_what_a_profile_compares(client):
    from catan.server.behavior import _title
    assert client.get("/api/behavior").json()["profiles"][0]["title"] == "basic vs basic"
    spec = "rl:checkpoint=runs/klbc_b1_long/checkpoints/ckpt_00183.pt"
    assert _title({"vs": "basic", "agents": {"a": {"agent": spec}, "b": {"agent": spec + ",deterministic=True"}}}) == "klbc_b1_long #183 vs basic"
    assert _title({"vs": "basic", "agents": {"x": {"agent": "arch_rl:checkpoint=runs/arch_mlp_v1_bc/bc.pt"},
                                             "y": {"agent": "exit_search:sims=16"}}}) == "arch_mlp_v1_bc, exit_search vs basic"
    assert _title({"vs": "random", "agents": {k: {"agent": k} for k in "abcde"}}) == "a, b, c +2 more vs random"
    assert _title({"vs": "basic", "watch_run": "/x/runs/sparse_r1", "agents": {"c1": {}, "c2": {}}}) == "sparse_r1: 2 checkpoints vs basic"
