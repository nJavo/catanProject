import json

import pytest

from catan.bots import BasicHeuristicAgent, IntermediateAgent
from catan.engine import Game
from catan.research.evox.agent import DIM, INTERMEDIATE_Z, EvoxAgent, weights
from catan.research.evox.loop import anchor_elo, main


def _log(agents, seed):
    g = Game(seed=seed)
    for i, a in enumerate(agents):
        a.reset(i, seed)
    while not g.is_over:
        g.step(agents[g.actor].act(g, g.legal_actions()))
    return g.action_log


def test_intermediate_genome_reproduces_intermediate():
    a = _log([EvoxAgent(z=INTERMEDIATE_Z)] + [BasicHeuristicAgent() for _ in range(3)], 3)
    b = _log([IntermediateAgent()] + [BasicHeuristicAgent() for _ in range(3)], 3)
    assert a == b


def test_extended_genome_reproduces_intermediate():
    from catan.research.evox.agent import make
    from catan.research.evox.agent_x import DIM_X, INTERMEDIATE_ZX, EvoxXAgent
    ag = make(z=list(INTERMEDIATE_ZX))
    assert isinstance(ag, EvoxXAgent) and len(INTERMEDIATE_ZX) == DIM_X > DIM
    for seed in (3, 5):
        a = _log([make(z=list(INTERMEDIATE_ZX)), IntermediateAgent(), BasicHeuristicAgent(), BasicHeuristicAgent()], seed)
        b = _log([IntermediateAgent(), IntermediateAgent(), BasicHeuristicAgent(), BasicHeuristicAgent()], seed)
        assert a == b


def test_weights_and_checkpoint(tmp_path):
    with pytest.raises(AssertionError):
        weights([0.0] * (DIM - 1))
    p = tmp_path / "g.json"
    p.write_text(json.dumps({"z": [0.0] * DIM}))
    ag = EvoxAgent(checkpoint=str(p))
    assert ag.g["prod"] == 0.0 and ag.g["setup_pips"] == 0.0
    import torch
    torch.save({"z": [1.0] * DIM}, tmp_path / "g.pt")
    assert EvoxAgent(checkpoint=str(tmp_path / "g.pt")).g["hand"] == 1.0
    assert EvoxAgent(z=INTERMEDIATE_Z).g["prod"] == pytest.approx(0.045) and len(INTERMEDIATE_Z) == DIM


def test_anchor_elo():
    anchors = {"a": 1400.0, "b": 1600.0}
    r_even, _ = anchor_elo({"a": (50, 100), "b": (50, 100)}, anchors)
    assert abs(r_even - 1500) < 1
    r_hi, se = anchor_elo({"a": (90, 100), "b": (70, 100)}, anchors)
    assert r_hi > 1600 and 0 < se < 100


def test_loop_smoke(tmp_path):
    main(["--run-id", "t", "--runs-dir", str(tmp_path), "--islands", "2", "--members", "3", "--generations", "2",
          "--every", "1", "--candidates", "1", "--sel-seeds", "1", "--report-seeds", "1", "--procs", "1"])
    d = tmp_path / "t"
    pop = [json.loads(x) for x in (d / "population.jsonl").read_text().splitlines()]
    ch = [json.loads(x) for x in (d / "champions.jsonl").read_text().splitlines()]
    assert [p["gen"] for p in pop] == [0, 1] and len(ch) == 2
    m = pop[-1]["members"][0]
    assert {"id", "parent", "born", "elo", "hparams", "shaping", "matches", "win_rate"} <= set(m)
    assert set(ch[-1]["vs"]) == {"random", "basic", "intermediate"}
    best = json.loads((d / "best.json").read_text())
    assert len(best["z"]) == DIM and len(best["root_z"]) == DIM
    main(["--run-id", "t", "--runs-dir", str(tmp_path), "--resume", "--generations", "3"])
    assert len((d / "generations.jsonl").read_text().splitlines()) == 3
