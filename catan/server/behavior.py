"""Behavior profiles API: ``results/behavior/*.json`` written by scripts/behavior_report.py.

    GET /api/behavior          -> {"profiles": [{file, id, title, vs, created, agents: [label...]}]}
    GET /api/behavior/{file}   -> full profile JSON
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from fastapi import APIRouter, HTTPException

_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.\-]*\.json$")


def _agent_name(spec: str) -> str:
    """'rl:checkpoint=runs/klbc_b1_long/checkpoints/ckpt_00183.pt,deterministic=True' -> 'klbc_b1_long #183'."""
    if "checkpoint=" not in spec:
        return spec.split(":")[0]
    p = Path(spec.split("checkpoint=", 1)[1].split(",")[0])
    run = p.parts[p.parts.index("runs") + 1] if "runs" in p.parts[:-1] else None
    m = re.fullmatch(r"ckpt_0*(\d+)", p.stem)
    if run is None:
        return p.stem
    return f"{run} #{m.group(1)}" if m else run if p.stem == "bc" else f"{run} {p.stem}"


def _title(d: dict) -> str:
    """Human chip label for a profile: what it compares, against whom (DS6-08)."""
    vs = d.get("vs") or "?"
    if d.get("watch_run"):
        return f"{Path(d['watch_run']).name}: {len(d['agents'])} checkpoints vs {vs}"
    names = list(dict.fromkeys(_agent_name(str(a.get("agent") or k)) if isinstance(a, dict) else k
                               for k, a in d["agents"].items()))
    shown = ", ".join(names[:3]) + (f" +{len(names) - 3} more" if len(names) > 3 else "")
    return f"{shown} vs {vs}"


def behavior_router(data: Path) -> APIRouter:
    base = (Path(data) / "results" / "behavior").resolve()
    r = APIRouter()

    def load(p: Path):
        try:
            return json.loads(p.read_text())
        except (OSError, ValueError):
            return None

    @r.get("/api/behavior")
    def list_profiles():
        out = []
        if base.is_dir():
            for p in sorted(base.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
                d = load(p)
                if isinstance(d, dict) and isinstance(d.get("agents"), dict):
                    out.append({"file": p.name, "id": d.get("id", p.stem), "vs": d.get("vs"),
                                "created": d.get("created"), "watch_run": d.get("watch_run"), "title": _title(d),
                                "agents": list(d["agents"])})
        # DS35-01: newest first by the file's own `created`; a fresh checkout gives every file the same mtime.
        # Stable sort, so files without `created` stay in mtime order after the dated ones.
        out.sort(key=lambda d: str(d["created"] or ""), reverse=True)
        return {"profiles": out}

    @r.get("/api/behavior/{name}")
    def get_profile(name: str):
        if not _NAME.match(name):
            raise HTTPException(400, "invalid name")
        p = (base / name).resolve()
        if p.parent != base:
            raise HTTPException(400, "invalid name")
        if not p.is_file():
            raise HTTPException(404, "no such profile")
        d = load(p)
        if not isinstance(d, dict):
            raise HTTPException(422, "unreadable profile")
        # DS36-09: readable column names, the same ones the chip title uses
        if isinstance(d.get("agents"), dict):
            d["agent_names"] = {k: _agent_name(str(a.get("agent") or k)) if isinstance(a, dict) else k
                                for k, a in d["agents"].items()}
        return d

    return r
