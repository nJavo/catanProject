"""DS4-14/15: light /api/runs list, per-run detail with a byte cursor and ETags,
path safety, and behaviour-cloning folders."""
import json

from fastapi.testclient import TestClient

from catan.server.app import create_app


def _client(tmp_path):
    return TestClient(create_app(data_dir=tmp_path, static_dir=tmp_path / "nodist"))


def _run(tmp_path, name, rows=5, eval_rows=0):
    d = tmp_path / "runs" / name
    (d / "checkpoints").mkdir(parents=True)
    (d / "meta.json").write_text(json.dumps({"run_id": name, "started": "2026-11-04T10:00:00"}))
    (d / "config.yaml").write_text("lr: 0.0003\n")
    with open(d / "metrics.jsonl", "w") as f:
        for i in range(rows):
            f.write(json.dumps({"step": i, "entropy": 1.0 - i / 100}) + "\n")
    with open(d / "eval.jsonl", "w") as f:
        for i in range(eval_rows):
            f.write(json.dumps({"step": i, "per_opponent": {"basic": {"win_rate": 0.1 * i, "ci95": [0, 1],
                                                                     "games": 50, "avg_vp": 5}}}) + "\n")
    (d / "checkpoints" / "ckpt_00001.pt").write_bytes(b"x")
    return d


def test_list_is_light_and_summarises(tmp_path):
    _run(tmp_path, "big", rows=3000, eval_rows=2)
    c = _client(tmp_path)
    r = c.get("/api/runs")
    item = r.json()["runs"][0]
    assert "metrics" not in item and "config" not in item and len(r.content) < 2000
    assert item["metrics_total"] == 3000 and item["eval_total"] == 2 and item["checkpoints"] == 1
    assert item["kind"] == "rl" and item["meta"]["started"].startswith("2026")
    assert item["last_eval"] == {"step": 1, "checkpoint": None,
                                 "per_opponent": {"basic": {"win_rate": 0.1, "ci95": [0, 1], "games": 50}}}


def test_detail_cursor_returns_only_new_rows(tmp_path):
    d = _run(tmp_path, "live", rows=5)
    c = _client(tmp_path)
    full = c.get("/api/runs/live").json()
    assert [m["step"] for m in full["metrics"]] == list(range(5)) and full["config"] == "lr: 0.0003\n"
    assert full["checkpoints"] == ["ckpt_00001.pt"] and full["since"] == 0
    cur = full["cursor"]
    assert c.get(f"/api/runs/live?since={cur}").json()["metrics"] == []
    with open(d / "metrics.jsonl", "a") as f:
        f.write(json.dumps({"step": 5}) + "\n" + '{"step": 6, "entr')  # a writer mid-append
    inc = c.get(f"/api/runs/live?since={cur}").json()
    assert [m["step"] for m in inc["metrics"]] == [5] and not inc["reset"] and "config" not in inc
    with open(d / "metrics.jsonl", "a") as f:
        f.write('opy": 0.5}\n')  # the partial line completes
    inc2 = c.get(f"/api/runs/live?since={inc['cursor']}").json()
    assert inc2["metrics"] == [{"step": 6, "entropy": 0.5}]


def test_detail_resets_when_the_log_is_rewritten(tmp_path):
    d = _run(tmp_path, "resumed", rows=10)
    c = _client(tmp_path)
    cur = c.get("/api/runs/resumed").json()["cursor"]
    (d / "metrics.jsonl").write_text(json.dumps({"step": 0}) + "\n")  # trimmed on resume
    r = c.get(f"/api/runs/resumed?since={cur}").json()
    assert r["reset"] and r["metrics"] == [{"step": 0}] and r["config"] is not None
    (d / "metrics.jsonl").write_text("".join(json.dumps({"step": i, "pad": "x" * 30}) + "\n" for i in range(10)))
    r = c.get(f"/api/runs/resumed?since={cur - 3}").json()  # cursor not on a line boundary
    assert r["reset"] and len(r["metrics"]) == 10


def test_large_logs_are_downsampled_only_on_full_fetch(tmp_path):
    d = _run(tmp_path, "long", rows=5000)
    c = _client(tmp_path)
    full = c.get("/api/runs/long").json()
    assert full["downsampled"] and len(full["metrics"]) <= 2001 and full["metrics"][-1]["step"] == 4999
    with open(d / "metrics.jsonl", "a") as f:
        f.write("".join(json.dumps({"step": 5000 + i}) + "\n" for i in range(3)))
    assert [m["step"] for m in c.get(f"/api/runs/long?since={full['cursor']}").json()["metrics"]] == [5000, 5001, 5002]


def test_etags_give_304_until_something_changes(tmp_path):
    d = _run(tmp_path, "e", rows=3)
    c = _client(tmp_path)
    for url in ("/api/runs", "/api/runs/e", "/api/runs/e?since=10"):
        r = c.get(url)
        tag = r.headers["etag"]
        assert r.headers["cache-control"] == "no-cache"
        assert c.get(url, headers={"If-None-Match": tag}).status_code == 304
        with open(d / "metrics.jsonl", "a") as f:
            f.write(json.dumps({"step": 99}) + "\n")
        r2 = c.get(url, headers={"If-None-Match": tag})
        assert r2.status_code == 200 and r2.headers["etag"] != tag


def test_detail_is_path_safe(tmp_path):
    _run(tmp_path, "ok")
    (tmp_path / "secret").mkdir()
    (tmp_path / "secret" / "metrics.jsonl").write_text('{"step": 1}\n')
    (tmp_path / "runs" / "link").symlink_to(tmp_path / "secret")
    c = _client(tmp_path)
    assert c.get("/api/runs/ok").status_code == 200
    for bad in ("..", "%2e%2e", "..%2Fsecret", "link", ".hidden", "missing", "a b", "ok%00"):
        assert c.get(f"/api/runs/{bad}").status_code in (400, 404), bad
    assert c.get("/api/runs/ok?since=-1").status_code == 422


def test_bc_only_folders_are_summarised(tmp_path):
    d = tmp_path / "runs" / "bc_teacher"
    d.mkdir(parents=True)
    hist = [{"epoch": e, "train_loss": 1.0 / e, "heldout_acc": 0.6 + 0.02 * e} for e in range(1, 9)]
    (d / "bc_summary.json").write_text(json.dumps({"teacher": "intermediate", "decisions": 1234, "games": 10,
                                                   "history": hist}))
    (d / "eval.jsonl").write_text(json.dumps({"checkpoint": "runs/bc_teacher/bc.pt", "per_opponent":
                                              {"basic": {"win_rate": 0.04, "ci95": [0.02, 0.08], "games": 200}}}) + "\n")
    c = _client(tmp_path)
    item = c.get("/api/runs").json()["runs"][0]
    assert item["kind"] == "bc" and item["metrics_total"] == 0 and item["eval_total"] == 1
    assert item["bc"]["teacher"] == "intermediate" and item["bc"]["epochs"] == 8
    assert abs(item["bc"]["heldout_acc"] - 0.76) < 1e-9
    assert item["last_eval"]["per_opponent"]["basic"]["win_rate"] == 0.04
    det = c.get("/api/runs/bc_teacher").json()
    assert det["bc_summary"]["history"] == hist and len(det["eval"]) == 1 and det["metrics"] == []
