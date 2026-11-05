"""Versioned model registry stored in ``models/registry.json``.

Entry: {id, name, kind (heuristic|rl), spec, checkpoint, sha256, source_run,
update, step, created, git_commit, status, anchor, eval}. Heuristic bots are
fixed anchors. ``status`` is candidate | promoted | retired.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

from catan.eval.tournament import _git_commit

REGISTRY_PATH = Path("models/registry.json")
STATUSES = ("candidate", "promoted", "retired")
ANCHORS = ("random", "basic", "intermediate", "search")
DEFAULT_CHAMPION = "intermediate"  # champion until an RL entry is promoted


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load(path=REGISTRY_PATH) -> dict:
    try:
        with open(path) as f:
            reg = json.load(f)
    except FileNotFoundError:
        reg = {"version": 1, "entries": []}
    reg.setdefault("entries", [])
    return reg


def save(reg: dict, path=REGISTRY_PATH) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(reg, indent=1) + "\n")
    os.replace(tmp, path)  # atomic: a crash never leaves a half-written registry


def get(reg: dict, eid: str) -> dict:
    for e in reg["entries"]:
        if e["id"] == eid:
            return e
    raise KeyError(f"no registry entry {eid!r}")


def spec_to_id(reg: dict) -> dict:
    return {e["spec"]: e["id"] for e in reg["entries"]}


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def add_anchor(reg: dict, name: str) -> dict:
    for e in reg["entries"]:
        if e["id"] == name:
            return e
    e = {"id": name, "name": name, "kind": "heuristic", "spec": name, "checkpoint": None,
         "sha256": None, "source_run": None, "update": None, "step": None, "created": _now(),
         "git_commit": _git_commit(), "status": "promoted", "anchor": True, "eval": None}
    reg["entries"].append(e)
    return e


def ckpt_rel(path, root=".") -> str:
    """Repo-relative path when the checkpoint lives under ``root`` (symlinks kept)."""
    rel = os.path.relpath(os.path.abspath(path), os.path.abspath(root))
    return path if rel.startswith("..") else rel


def add_checkpoint(reg: dict, checkpoint: str, name: str, root=".", agent: str = "rl", spec_kw: str | None = None) -> dict:
    """Register ``checkpoint`` as a candidate. Same sha256, agent and extra spec kwargs already
    registered -> that entry (the same weights under another agent, e.g. arch_rl vs combo, is a new entry)."""
    import torch  # lazy: registry reads must not need torch
    full = Path(root) / checkpoint if not os.path.isabs(checkpoint) else Path(checkpoint)
    if not full.is_file():
        raise FileNotFoundError(checkpoint)
    sha = sha256_file(full)
    suffix = f",{spec_kw}" if spec_kw else ""
    for e in reg["entries"]:
        spec = e.get("spec") or ""
        if e.get("sha256") == sha and spec.split(":", 1)[0] == agent and spec.endswith(suffix):
            return e
    ck = torch.load(full, map_location="cpu", weights_only=True)
    rel = ckpt_rel(str(full), root)
    parts = Path(rel).parts
    run = parts[parts.index("runs") + 1] if "runs" in parts[:-2] else None
    version = 1 + sum(1 for e in reg["entries"] if e["name"] == name)
    e = {"id": f"{name}-v{version}", "name": name, "kind": "rl", "spec": f"{agent}:checkpoint={rel}" + (f",{spec_kw}" if spec_kw else ""),
         "checkpoint": rel, "sha256": sha, "source_run": run, "update": ck.get("update"),
         "step": ck.get("step"), "created": _now(), "git_commit": _git_commit(),
         "status": "candidate", "anchor": False, "eval": None}
    reg["entries"].append(e)
    return e


def verify(entry: dict, root=".") -> None:
    """Refuse to evaluate a checkpoint whose bytes changed since registration."""
    if not entry.get("checkpoint"):
        return
    p = Path(root) / entry["checkpoint"]
    if not p.is_file():
        raise FileNotFoundError(f"{entry['id']}: checkpoint {entry['checkpoint']} missing")
    got = sha256_file(p)
    if got != entry["sha256"]:
        raise ValueError(f"{entry['id']}: sha256 mismatch ({got[:12]} != {entry['sha256'][:12]})")


def set_status(reg: dict, eid: str, status: str) -> dict:
    if status not in STATUSES:
        raise ValueError(f"status must be one of {STATUSES}")
    e = get(reg, eid)
    if e.get("anchor") and status != "promoted":
        raise ValueError("anchors are fixed")
    e["status"] = status
    e["status_changed"] = _now()
    return e


def rl_champion(reg: dict) -> dict | None:
    """Most recently promoted RL entry, or None."""
    # any learned (non-anchor) promoted entry: rl, arch_rl, nego, ...
    rl = [e for e in reg["entries"] if not e.get("anchor") and e.get("checkpoint") and e.get("status") == "promoted"]
    return max(rl, key=lambda e: e.get("status_changed") or e["created"]) if rl else None


def champion(reg: dict) -> dict:
    """Most recently promoted RL entry, else the default heuristic anchor."""
    rl = rl_champion(reg)
    if rl:
        return rl
    try:
        return get(reg, DEFAULT_CHAMPION)
    except KeyError:
        return add_anchor(reg, DEFAULT_CHAMPION)
