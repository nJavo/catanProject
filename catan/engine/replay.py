"""Replay file format (EN-006).

A replay is ``{format: "catan-replay", version: 1, rules_digest, seed, config,
agents, actions, meta}``. ``seed + config + actions`` re-simulate the game
exactly (``Game.replay``). ``rules_digest`` fingerprints the engine rules that
wrote the file; a different digest on load means the rules changed since, so
re-simulation may diverge (loaders warn via ``check_rules`` but still load).

Files written before versioning have no ``format``/``version``: ``load_replay``
treats them as version 1 (same layout). Unknown versions or formats raise
``ReplayFormatError`` (the server answers 422).
"""
from __future__ import annotations

import hashlib
import json
import random
from functools import lru_cache

REPLAY_FORMAT = "catan-replay"
REPLAY_VERSION = 1
SUPPORTED_VERSIONS = (1,)


class ReplayFormatError(ValueError):
    pass


@lru_cache(maxsize=1)
def rules_digest() -> str:
    """Fingerprint of the rules: final states of a few fixed random games.
    Any change to rules or RNG use changes it."""
    from .game import Game, GameConfig
    h = hashlib.sha256()
    for s in range(3):
        g = Game(config=GameConfig(max_turns=150), seed=s)
        rng = random.Random(s)
        while not g.is_over:
            g.step(rng.choice(g.legal_actions()))
        d = g.to_dict()
        d.pop("action_log")
        h.update(json.dumps(d, sort_keys=True).encode())
    return h.hexdigest()[:16]


def make_replay(seed, config: dict, agents, actions, meta: dict | None = None) -> dict:
    """A versioned replay document. ``actions`` are Action objects or their JSON."""
    return {"format": REPLAY_FORMAT, "version": REPLAY_VERSION, "rules_digest": rules_digest(),
            "seed": seed, "config": config, "agents": list(agents),
            "actions": [a.to_json() if hasattr(a, "to_json") else a for a in actions],
            "meta": meta if meta is not None else {}}


def load_replay(d) -> dict:
    """Validate and migrate a replay document to the current version (returns a new dict)."""
    if not isinstance(d, dict) or "actions" not in d or "seed" not in d:
        raise ReplayFormatError("not a replay: needs seed and actions")
    fmt = d.get("format", REPLAY_FORMAT)
    if fmt != REPLAY_FORMAT:
        raise ReplayFormatError(f"unknown replay format {fmt!r}")
    ver = d.get("version", 1)  # files without a version predate EN-006 and are v1
    if type(ver) is not int or ver not in SUPPORTED_VERSIONS:
        raise ReplayFormatError(f"unsupported replay version {ver!r} (supported: {list(SUPPORTED_VERSIONS)})")
    out = dict(d)
    out["format"], out["version"] = REPLAY_FORMAT, REPLAY_VERSION
    out.setdefault("rules_digest", None)  # unknown for pre-versioning files
    return out


def check_rules(d: dict) -> str | None:
    """Warning text if ``d`` was written by different rules than this engine's."""
    rd = d.get("rules_digest")
    if rd and rd != rules_digest():
        return f"replay written with rules {rd}, engine has {rules_digest()}: re-simulation may diverge"
    return None
