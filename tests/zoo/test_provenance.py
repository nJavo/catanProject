"""RL-020: no promotion or ladder report from non-reproducible code."""
import json
import subprocess

import pytest

from catan.rl.model import PolicyNet, save_checkpoint
from catan.rl.spaces import ACT_DIM, OBS_DIM
from catan.zoo import gate


def _run(root, name, meta):
    d = root / "runs" / name
    (d / "checkpoints").mkdir(parents=True)
    net = PolicyNet(OBS_DIM, ACT_DIM, hidden=16, blocks=1)
    save_checkpoint(d / "checkpoints" / "ckpt_00001.pt", net, {"model": {"hidden": 16, "blocks": 1}},
                    extra={"update": 1, "step": 100})
    if meta is not None:
        (d / "meta.json").write_text(json.dumps(meta))
    return d


def test_run_provenance(tmp_path):
    clean = _run(tmp_path, "clean", {"git_commit": "abc123", "git_dirty": False})
    dirty = _run(tmp_path, "dirty", {"git_commit": "abc123", "git_dirty": True})
    nometa = _run(tmp_path, "nometa", None)
    noflag = _run(tmp_path, "noflag", {"git_commit": "abc123"})
    nocommit = _run(tmp_path, "nocommit", {"git_commit": "unknown", "git_dirty": False})
    assert gate.run_provenance(clean) == []
    assert "dirty git tree" in gate.run_provenance(dirty)[0]
    assert "no meta.json" in gate.run_provenance(nometa)[0]
    assert "no git_dirty flag" in gate.run_provenance(noflag)[0]
    assert "no git_commit" in gate.run_provenance(nocommit)[0]


def _git(root, *args):
    subprocess.run(["git", *args], cwd=root, check=True, capture_output=True,
                   env={"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t",
                        "GIT_COMMITTER_EMAIL": "t@t", "HOME": str(root), "PATH": "/usr/bin:/bin:/usr/local/bin"})


def test_tree_provenance(tmp_path):
    assert "cannot check" in gate.tree_provenance(tmp_path)[0]  # not a repo: unknown, so refuse
    _git(tmp_path, "init", "-q")
    (tmp_path / "a.py").write_text("x = 1\n")
    _git(tmp_path, "add", "a.py")
    _git(tmp_path, "commit", "-qm", "init")
    assert gate.tree_provenance(tmp_path) == []
    (tmp_path / "new_results.json").write_text("{}")  # untracked output files don't count
    assert gate.tree_provenance(tmp_path) == []
    (tmp_path / "a.py").write_text("x = 2\n")
    assert "uncommitted changes: a.py" in gate.tree_provenance(tmp_path)[0]


@pytest.fixture
def repo(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("scripts.promote.provenance_guard",
                        lambda runs: [p for r in runs for p in gate.run_provenance(r)])  # tree: tmp isn't a repo
    return tmp_path


def test_promote_refuses_dirty_run_unless_allowed(repo, monkeypatch):
    from scripts import promote
    _run(repo, "r1", {"git_commit": "abc", "git_dirty": True})
    played = []
    monkeypatch.setattr(promote, "run_gate", lambda *a, **k: played.append(1) or pytest.skip("gate not under test"))
    argv = ["--checkpoint", "runs/r1/checkpoints/ckpt_00001.pt", "--name", "r1",
            "--registry", str(repo / "models" / "registry.json"), "--out", str(repo / "results")]
    assert promote.main(argv) == 4 and not played
    reg = json.loads((repo / "models" / "registry.json").read_text())
    assert "provenance_warning" not in next(e for e in reg["entries"] if e["id"] == "r1-v1")
    with pytest.raises(pytest.skip.Exception):
        promote.main(argv + ["--allow-dirty"])
    assert played
    reg = json.loads((repo / "models" / "registry.json").read_text())
    assert "dirty git tree" in next(e for e in reg["entries"] if e["id"] == "r1-v1")["provenance_warning"][0]


def test_promote_clean_run_reaches_gate(repo, monkeypatch):
    from scripts import promote
    _run(repo, "r2", {"git_commit": "abc", "git_dirty": False})
    monkeypatch.setattr(promote, "run_gate", lambda *a, **k: pytest.skip("reached the gate"))
    with pytest.raises(pytest.skip.Exception):
        promote.main(["--checkpoint", "runs/r2/checkpoints/ckpt_00001.pt", "--name", "r2",
                      "--registry", str(repo / "models" / "registry.json"), "--out", str(repo / "results")])


def test_ladder_refuses_dirty_run_unless_allowed(tmp_path, monkeypatch):
    from scripts import ladder
    monkeypatch.chdir(tmp_path)
    d = _run(tmp_path, "r3", {"git_commit": "abc", "git_dirty": True})
    monkeypatch.setattr(ladder, "provenance_guard", lambda runs: [p for r in runs for p in gate.run_provenance(r)])
    ran = []

    def fake_ladder(run, *a, **k):
        ran.append(run)
        p = tmp_path / "results" / "tournaments" / "ladder_r3.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"kind": "ladder", "run": "r3", "checkpoints": []}))
        return {"kind": "ladder", "run": "r3", "checkpoints": []}
    monkeypatch.setattr(ladder, "run_ladder", fake_ladder)
    monkeypatch.setattr(ladder, "summary_table", lambda lad: "")
    monkeypatch.setattr(ladder.leaderboard, "write", lambda *a, **k: {"games": 0, "entries": []})
    base = ["--run", str(d), "--out", str(tmp_path / "results"), "--registry", str(tmp_path / "reg.json")]
    assert ladder.main(base) == 4 and not ran
    assert ladder.main(base + ["--allow-dirty"]) is None and ran
    out = json.loads((tmp_path / "results" / "tournaments" / "ladder_r3.json").read_text())
    assert "dirty git tree" in out["provenance_warning"][0]
