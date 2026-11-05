"""Name -> agent factory registry used by server, tournaments and CLIs."""
from __future__ import annotations

import threading
from typing import Callable

from .base import Agent, RandomAgent

_REGISTRY: dict[str, Callable[..., Agent]] = {}
_UNTRUSTED: set[str] = set()


def register(name: str, factory: Callable[..., Agent], trusted: bool = True) -> None:
    """``trusted=False``: the tournament runner and server always give this
    agent ``game.observe_for(seat)`` instead of the full game (catan.engine.infoset)."""
    _REGISTRY[name] = factory
    (_UNTRUSTED.discard if trusted else _UNTRUSTED.add)(name)


def is_trusted(name: str) -> bool:
    _load_plugins()
    return name not in _UNTRUSTED


def make_agent(name: str, **kwargs) -> Agent:
    _load_plugins()
    if name not in _REGISTRY:
        raise KeyError(f"unknown agent {name!r}; known: {sorted(_REGISTRY)}")
    return _REGISTRY[name](**kwargs)


def available_agents() -> list[str]:
    _load_plugins()
    return sorted(_REGISTRY)


_loaded = False
_load_lock = threading.RLock()


def _load_plugins():
    """Import optional agent modules so they self-register. Thread-safe: server threads used to race
    here, and a thread that arrived while another was still importing torch saw a partial registry
    (or, if that import failed, the plugins never loaded for the life of the process)."""
    global _loaded
    if _loaded:
        return
    with _load_lock:
        if _loaded:
            return
        import importlib
        for mod in ("catan.bots", "catan.rl.policy_agent", "catan.zoo.champion", "catan.research.jointb.gated",
                    "catan.research.arch.agent", "catan.research.nego.agent", "catan.research.exit", "catan.research.combo.agent", "catan.research.value"):
            try:
                importlib.import_module(mod)
            except ModuleNotFoundError as e:
                if e.name is None or not mod.startswith(e.name):
                    raise
        _loaded = True


register("random", lambda seed=None, **_: RandomAgent(seed))
