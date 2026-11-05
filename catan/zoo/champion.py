"""``champion`` agent: the RL entry most recently promoted by the gate.

Registered in ``catan.agents.registry`` only when ``models/registry.json`` has
a promoted RL entry, so it shows up in ``available_agents()`` and every agent
dropdown. Construction re-reads the registry and verifies the sha256.
shortcut: listing is decided at import, so a long-running server shows a newly
promoted champion after restart; resolve per request if that matters.
"""
from __future__ import annotations

import os
from pathlib import Path

from catan.agents.registry import register

from . import registry

ROOT = Path(__file__).resolve().parents[2]


def data_root() -> Path:
    return Path(os.environ.get("CATAN_DATA_DIR") or ROOT)


def current(root=None) -> dict | None:
    root = Path(root or data_root())
    return registry.rl_champion(registry.load(root / registry.REGISTRY_PATH))


def summary(entry: dict | None) -> dict | None:
    """Champion card for the API/UI: identity plus the gate evidence."""
    if not entry:
        return None
    g = entry.get("eval") or {}
    tests = {f"3x{h}": {k: r["candidate"][k] for k in ("win_rate", "ci95", "games")}
             for h, r in (g.get("heuristic_tests") or {}).items()}
    ct = g.get("champion_test")
    if ct:
        tests["champion_test"] = {"kind": ct["kind"], "previous": g.get("champion"),
                                  **{k: ct["candidate"][k] for k in ("win_rate", "ci95", "games")}}
        if ct["kind"] == "same_seat":  # D34 primary test: vs a same-seat copy of the previous champion
            tests["champion_test"].update(copy_win_rate=ct["copy"]["win_rate"], mcnemar_p=ct["mcnemar_p"],
                                          diff=ct["diff"], diff_ci95=ct["diff_ci95"])
    b = tests.get("3xbasic")
    label = f"Champion — {entry['name']}" + (
        f" (vs 3×basic {b['win_rate']:.0%} [{b['ci95'][0]:.0%}, {b['ci95'][1]:.0%}], n={b['games']})" if b else "")
    return {"id": entry["id"], "name": entry["name"], "kind": entry.get("kind"), "version": entry["id"].rsplit("-v", 1)[-1],
            "source_run": entry["source_run"], "update": entry["update"], "step": entry["step"],
            "checkpoint": entry["checkpoint"], "sha256": entry["sha256"],
            "promoted": entry.get("status_changed"), "gate": tests, "label": label}


def make_champion(root=None, **kw):
    from catan.agents.registry import make_agent
    from catan.eval.tournament import parse_spec
    root = Path(root or data_root())
    e = current(root)
    if e is None:
        raise ValueError("no champion: no RL model has passed the promotion gate yet "
                         "(python -m scripts.promote --checkpoint ... --name ...)")
    registry.verify(e, root)
    kw.pop("checkpoint", None)  # the champion's checkpoint is fixed by the registry
    # Build from the entry's agent spec so non-`rl` agents (e.g. arch_rl GNNs) work.
    name, spec_kw = parse_spec(e.get("spec") or f"rl:checkpoint={e['checkpoint']}")
    spec_kw["checkpoint"] = str(root / e["checkpoint"])
    return make_agent(name, **{**spec_kw, **kw})


if current() is not None:
    register("champion", make_champion)
