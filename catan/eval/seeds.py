"""Seed registry: the one source of truth for which game seeds serve which purpose.

A game seed is the ``seed`` passed to ``Game``. Purposes must never share seeds:
picking a checkpoint and reporting it on the same games biases the report.

Raw seed ranges (half-open, non-overlapping; see ``RANGES``):

    design_qa    [0, 1_000)                  unit tests, smoke runs, ad-hoc dev games
    train_eval   [1_000, 1_000_000)          RL in-training eval (checkpoint selection)
    heldout      [1_000_000, 10_000_000)     tournament blocks, see below
    train        [10_000_000, 10_000_000 + 2**30)  RL training games (also BC data)

Held-out blocks: block ``b`` is ``HELDOUT_BASE + b * BLOCK_SIZE + i`` for
``0 <= i < BLOCK_SIZE`` and ``0 <= b < NUM_BLOCKS`` (90). Block 90 would start at
10_000_000, inside the training range, so ``block_seeds`` refuses it. Named blocks:

    report             0, 60-69   published results; never used for selection or tuning
    design_qa          1-19       heuristic tuning (--seed 7), QA and ad-hoc tournaments
    ladder             20         checkpoint ladders (catan.zoo.ladder)
    anchors            21         4-way anchor tournament
    analysis_accuracy  30         analysis-engine accuracy studies
    gate               40-42      promotion gate only (catan.zoo.gate): basic, intermediate, champion
    selection          50-59      checkpoint/population selection, one block per research direction
                                  (52 = PBT, catan.pbt.loop)

Unlisted blocks are unassigned: claim one here before using it.
"""
from __future__ import annotations

import re

HELDOUT_BASE = 1_000_000
BLOCK_SIZE = 100_000
NUM_BLOCKS = 90                        # blocks 0..89; block 90 would hit TRAIN_SEED_BASE
TRAIN_SEED_BASE = 10_000_000
TRAIN_SEED_SPAN = 2 ** 30              # catan.rl.rollout draws TRAIN_SEED_BASE + randrange(2**30)
TRAIN_EVAL_BASE = 1_000                # default first seed of the RL in-training eval
BC_SEED_BASE = 20_000_000              # behaviour-cloning data games (inside the train range: training data)

RANGES = {
    "design_qa": (0, TRAIN_EVAL_BASE),
    "train_eval": (TRAIN_EVAL_BASE, HELDOUT_BASE),
    "heldout": (HELDOUT_BASE, HELDOUT_BASE + NUM_BLOCKS * BLOCK_SIZE),
    "train": (TRAIN_SEED_BASE, TRAIN_SEED_BASE + TRAIN_SEED_SPAN),
}

REPORT_BLOCKS = (0,) + tuple(range(60, 70))
DESIGN_QA_BLOCKS = tuple(range(1, 20))
LADDER_BLOCK = 20
ANCHOR_BLOCK = 21
ANALYSIS_ACCURACY_BLOCK = 30
GATE_BLOCKS = {"basic": 40, "intermediate": 41, "champion": 42}
SELECTION_BLOCKS = tuple(range(50, 60))
PBT_SELECTION_BLOCK = 52

BLOCKS = {
    "report": REPORT_BLOCKS,
    "design_qa": DESIGN_QA_BLOCKS,
    "ladder": (LADDER_BLOCK,),
    "anchors": (ANCHOR_BLOCK,),
    "analysis_accuracy": (ANALYSIS_ACCURACY_BLOCK,),
    "gate": tuple(GATE_BLOCKS.values()),
    "selection": SELECTION_BLOCKS,
}


def block_seeds(block: int, count: int) -> list[int]:
    """Seeds ``0..count-1`` of held-out block ``block``."""
    if type(block) is not int or not 0 <= block < NUM_BLOCKS:
        raise ValueError(f"held-out block must be an int in 0..{NUM_BLOCKS - 1} (block {NUM_BLOCKS} and up "
                         f"would overlap RL training seeds), got {block!r}")
    if not 0 <= count <= BLOCK_SIZE:
        raise ValueError(f"a block holds at most {BLOCK_SIZE} seeds, asked for {count}")
    base = HELDOUT_BASE + block * BLOCK_SIZE
    return list(range(base, base + count))


def block_of(seed: int) -> int | None:
    """Held-out block containing game seed ``seed`` (None outside the held-out range)."""
    lo, hi = RANGES["heldout"]
    return (seed - lo) // BLOCK_SIZE if lo <= seed < hi else None


def purpose_of(seed: int) -> str:
    """Name of the range or named block a game seed belongs to."""
    b = block_of(seed)
    if b is not None:
        return next((name for name, bs in BLOCKS.items() if b in bs), f"unassigned block {b}")
    return next((name for name, (lo, hi) in RANGES.items() if lo <= seed < hi), "unregistered")


def is_gate_seed(seed: int) -> bool:
    return block_of(seed) in GATE_BLOCKS.values()


# ----------------------------------------------------------------- promotion guard
def _ints(v):
    if isinstance(v, bool):
        return []
    if isinstance(v, int):
        return [v]
    if isinstance(v, str):  # e.g. "6250000 + gen*1000 + table": the leading base
        m = re.match(r"\s*(\d+)", v)
        return [int(m.group(1))] if m else []
    if isinstance(v, (list, tuple)):
        return [x for item in v for x in _ints(item)]
    return []


def gate_conflicts(meta, path="", _ctx=None) -> list[str]:
    """Places in a candidate's metadata (run meta.json, config, ``selection``
    record) that point at gate seeds. Values under a key containing "block" are
    read as held-out block numbers; under a key containing "seed" (at any depth
    above them) as game seeds. A ``[first, last]`` pair is a seed range and
    conflicts if it overlaps a gate block."""
    if isinstance(meta, dict):
        out = []
        for k, v in meta.items():
            key = str(k).lower()
            ctx = "block" if "block" in key else "seed" if "seed" in key else _ctx
            out += gate_conflicts(v, f"{path}.{k}" if path else str(k), ctx)
        return out
    if isinstance(meta, (list, tuple)) and any(isinstance(x, (dict, list, tuple)) for x in meta):
        return [c for i, x in enumerate(meta) for c in gate_conflicts(x, f"{path}[{i}]", _ctx)]
    if _ctx == "block":
        return [f"{path}: block {b} is a gate block" for b in _ints(meta) if b in GATE_BLOCKS.values()]
    if _ctx == "seed":
        if isinstance(meta, (list, tuple)) and len(meta) == 2 and all(type(x) is int for x in meta) \
                and meta[0] <= meta[1]:
            lo, hi = meta
            hit = [b for b in GATE_BLOCKS.values()
                   if lo < HELDOUT_BASE + (b + 1) * BLOCK_SIZE and HELDOUT_BASE + b * BLOCK_SIZE <= hi]
            return [f"{path}: seeds {lo}..{hi} overlap gate block {b}" for b in hit]
        return [f"{path}: seed {x} is in gate block {block_of(x)}" for x in _ints(meta) if is_gate_seed(x)]
    return []


def eval_range(cfg: dict) -> list[int] | None:
    """[first, last] game seeds of an RL config's in-training eval (selection) games."""
    ev = (cfg or {}).get("eval") or {}
    if "seed" not in ev:
        return None
    return [int(ev["seed"]), int(ev["seed"]) + max(1, int(ev.get("games", 1))) - 1]
