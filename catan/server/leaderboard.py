"""Read-only API for the model zoo: leaderboard, registry and checkpoint ladders."""
from __future__ import annotations

import json
import re
from pathlib import Path

from fastapi import APIRouter, HTTPException

from catan.zoo.champion import summary
from catan.zoo.registry import rl_champion

RUN_NAME = re.compile(r"^[A-Za-z0-9_.\-]{1,128}$")


def _read(p: Path, default):
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return default


def _slim(lad: dict) -> dict:
    """Drop per-game outcomes (only the leaderboard needs them)."""
    out = {k: v for k, v in lad.items() if k != "checkpoints"}
    out["checkpoints"] = [{**c, "results": {o: {k: v for k, v in r.items() if k not in ("outcomes", "lineups")}
                                            for o, r in c.get("results", {}).items()}}
                          for c in lad.get("checkpoints", [])]
    return out


def make_router(data: Path) -> APIRouter:
    r = APIRouter()
    tdir = data / "results" / "tournaments"

    @r.get("/api/leaderboard")
    def leaderboard():
        return _read(data / "results" / "leaderboard.json", {"entries": []})

    @r.get("/api/registry")
    def registry():
        reg = _read(data / "models" / "registry.json", {"entries": []})
        reg["champion"] = summary(rl_champion(reg)) if isinstance(reg.get("entries"), list) else None
        for e in reg.get("entries") or []:  # the UI offers Play only for checkpoints this server can load
            if isinstance(e, dict) and isinstance(e.get("checkpoint"), str):
                e["available"] = (data / e["checkpoint"]).is_file()
        return reg

    @r.get("/api/ladders")
    def ladders():
        out = []
        for p in sorted(tdir.glob("ladder_*.json")) if tdir.is_dir() else []:
            lad = _read(p, None)
            if isinstance(lad, dict) and lad.get("kind") == "ladder":
                out.append({**_slim(lad), "file": p.name})
        return {"ladders": out}

    @r.get("/api/ladders/{run}")
    def ladder(run: str):
        if not RUN_NAME.match(run) or run in (".", ".."):
            raise HTTPException(400, "invalid run name")
        p = (tdir / f"ladder_{run}.json").resolve()
        if p.parent != tdir.resolve() or not p.is_file():
            raise HTTPException(404, "no ladder for that run")
        lad = _read(p, None)
        if not isinstance(lad, dict):
            raise HTTPException(404, "no ladder for that run")
        return _slim(lad)

    return r
