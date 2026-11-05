import json
import os

from scripts import behavior_report as br


def test_watch_profiles_new_checkpoints_once(tmp_path, monkeypatch):
    ck = tmp_path / "run7" / "checkpoints"
    ck.mkdir(parents=True)
    for name in ("ckpt_00010.pt", "ckpt_00020.pt", "latest.pt"):
        (ck / name).write_bytes(b"x")
        os.utime(ck / name, (1, 1))  # old enough to be settled
    calls = []

    def fake_profile(spec, vs, games, seed, workers, max_turns):
        calls.append(spec)
        return {"agent": spec, "label": br.label(spec), "checkpoint_index": br._ckpt_index(spec),
                "elapsed_sec": 0, "outcome": {"win_rate": 0, "ci95": [0, 1], "avg_vp": {"mean": 1}},
                "behavior": {"metrics": {k: {"mean": 0} for k in ("open_pips_total", "cities_built",
                                                                  "settlements_built")},
                             "ratios": {"robber_self_block_rate": {"rate": 0}}}}

    monkeypatch.setattr(br, "profile", fake_profile)
    out = tmp_path / "out"
    args = ["--watch", str(tmp_path / "run7"), "--once", "--out", str(out)]
    br.main(args)
    assert [c.rsplit("/", 1)[1] for c in calls] == ["ckpt_00010.pt", "ckpt_00020.pt"]  # latest.pt skipped
    d = json.loads((out / "watch-run7-vs-basic.json").read_text())
    assert sorted(p["checkpoint_index"] for p in d["agents"].values()) == [10, 20]
    br.main(args)  # nothing new: no re-profiling
    assert len(calls) == 2
    (ck / "ckpt_00030.pt").write_bytes(b"x")  # fresh file: still being written, skipped
    br.main(args)
    assert len(calls) == 2
