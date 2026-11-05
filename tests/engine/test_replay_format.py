"""EN-006: versioned replay format with a migration shim for unversioned files."""
import json
import random
from pathlib import Path

import pytest

from catan.engine import Game, GameConfig
from catan.engine.replay import (REPLAY_FORMAT, REPLAY_VERSION, ReplayFormatError, check_rules, load_replay,
                                 make_replay, rules_digest)

ROOT = Path(__file__).resolve().parents[2]


def _game(seed=4):
    g = Game(config=GameConfig(max_turns=60), seed=seed)
    rng = random.Random(seed)
    while not g.is_over:
        g.step(rng.choice(g.legal_actions()))
    return g


def test_written_replays_are_versioned_and_resimulate():
    g = _game()
    d = json.loads(json.dumps(make_replay(g.seed, g.config.to_dict(), ["random"] * 4, g.action_log, {"x": 1})))
    assert d["format"] == REPLAY_FORMAT and d["version"] == REPLAY_VERSION == 1
    assert d["rules_digest"] == rules_digest() and check_rules(d) is None
    r = load_replay(d)
    assert Game.replay(r["seed"], GameConfig(**r["config"]), r["actions"]).to_dict() == g.to_dict()


def test_unversioned_files_load_as_v1():
    g = _game(5)
    legacy = {"seed": g.seed, "config": g.config.to_dict(), "agents": ["random"] * 4,
              "actions": [a.to_json() for a in g.action_log], "meta": {}}
    r = load_replay(legacy)
    assert r["version"] == 1 and r["format"] == REPLAY_FORMAT and r["rules_digest"] is None
    assert "version" not in legacy  # input not mutated
    assert Game.replay(r["seed"], GameConfig(**r["config"]), r["actions"]).to_dict() == g.to_dict()


@pytest.mark.parametrize("bad", [{"version": 99}, {"version": "1"}, {"version": True}, {"format": "catan-save"}])
def test_unknown_versions_and_formats_rejected(bad):
    with pytest.raises(ReplayFormatError):
        load_replay({"seed": 1, "actions": [], **bad})


def test_rules_digest_mismatch_is_reported():
    assert "diverge" in check_rules({"rules_digest": "0000000000000000"})
