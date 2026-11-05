"""The one checkpoint path rule for every route (REG-004): game seats, lab
matches and analysis policies. Checkpoints are loaded with
``torch.load(weights_only=True)``, and only files under ``runs/`` or ``models/``
of the data directory are allowed (after resolving symlinks)."""
from __future__ import annotations

import os
from pathlib import Path

from fastapi import HTTPException

CHECKPOINT_DIRS = ("runs", "models")


def resolve_checkpoint(data: Path, value: str) -> str:
    """Absolute path of an allowed checkpoint, or HTTP 400."""
    p = (Path(value) if os.path.isabs(value) else Path(data) / value).resolve()
    if not any((Path(data) / d).resolve() in p.parents for d in CHECKPOINT_DIRS) or not p.is_file():
        raise HTTPException(400, "checkpoint must be an existing file under runs/ or models/")
    return str(p)
