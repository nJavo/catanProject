"""``evox``: intermediate's planner with every hand-set constant exposed as a gene (D29, docs/research/EVOX.md).

A genome is a vector ``z``. Weight k is ``z[k] * SCALE[k]``. SCALE is a fixed per-feature normaliser
(about 1 / typical feature size, rounded to a power of ten or a third of one), so z ~ N(0, 1) is a
random evaluator with random signs. The VP term stays at weight 1: it is the objective.
``INTERMEDIATE_Z`` is the z that reproduces the hand-tuned ``intermediate`` bot exactly.

    evox                                    # models/evox_best.json (the evolved champion)
    evox:checkpoint=runs/evox_r1/best.json  # any genome file with a "z" list
"""
from __future__ import annotations

import json
from pathlib import Path

from catan.engine.actions import A, Action
from catan.engine.board import TOPOLOGY as T, GENERIC_PORT, RESOURCES
from catan.bots import common as C
from catan.bots.intermediate import IntermediateAgent

# (gene, scale, intermediate's value)
GENES = (
    # intermediate.evaluate weights (catan.bots.intermediate.W)
    ("prod", 0.03, 0.045), ("diversity", 0.3, 0.12), ("spot0", 0.1, 0.08), ("spot1", 0.1, 0.04),
    ("spot2", 0.1, 0.015), ("hand", 1.0, 0.3), ("dev_held", 0.3, 0.3), ("knight_played", 0.3, 0.08),
    ("road_len", 0.1, 0.03), ("port2", 0.1, 0.012), ("port3", 0.3, 0.12), ("dev_ev", 1.0, 0.42),
    ("overflow", 0.3, 0.0),
    # setup settlement score
    ("setup_pips", 1.0, 1.0), ("setup_div", 1.0, 1.2), ("setup_comb_div", 1.0, 1.0), ("setup_wb", 1.0, 1.0),
    ("setup_wo", 1.0, 0.8), ("setup_port3", 1.0, 1.0), ("setup_port2", 0.3, 0.25), ("setup_exp", 0.1, 0.15),
    # domestic trade thresholds, knight, robber
    ("accept_margin", 0.1, 0.04), ("margin_per_vp", 0.1, 0.02), ("offer_min", 0.1, 0.05),
    ("knight_bonus", 1.0, 0.4), ("rob_vp", 0.3, 0.3), ("rob_victims", 1.0, 2.0),
)
NAMES = tuple(g[0] for g in GENES)
SCALE = tuple(g[1] for g in GENES)
INTERMEDIATE_Z = tuple(v / s for _, s, v in GENES)
DIM = len(GENES)
DEFAULT_CHECKPOINT = Path(__file__).resolve().parents[3] / "models" / "evox_best.json"
W_KEYS = NAMES[:13] + ("setup_exp", "setup_div")


def weights(z) -> dict:
    genes = GENES
    if len(z) != DIM:
        from .agent_x import DIM_X, GENES_X
        assert len(z) == DIM_X, (len(z), DIM, DIM_X)
        genes = GENES_X
    return {n: float(x) * s for (n, s, _), x in zip(genes, z)}


def _load(checkpoint):
    if str(checkpoint).endswith(".pt"):  # same dict as the .json, for the zoo registry (it torch.loads checkpoints)
        import torch
        return torch.load(checkpoint, map_location="cpu", weights_only=True)["z"]
    with open(checkpoint) as f:
        return json.load(f)["z"]


def make(z=None, checkpoint=None, **kw):
    """Agent for a genome or genome file; the genome length picks the base or extended (agent_x) class."""
    if z is None and checkpoint is None and DEFAULT_CHECKPOINT.exists():
        checkpoint = DEFAULT_CHECKPOINT
    if checkpoint:
        z = _load(checkpoint)
    if z is not None and len(z) != DIM:
        from .agent_x import EvoxXAgent
        return EvoxXAgent(z=z, **kw)
    return EvoxAgent(z=z, **kw)


class EvoxAgent(IntermediateAgent):
    name = "evox"

    def __init__(self, z=None, checkpoint: str | None = None, seed=None, max_offers: int = 2, **_):
        if z is None and checkpoint is None and DEFAULT_CHECKPOINT.exists():
            checkpoint = DEFAULT_CHECKPOINT
        if checkpoint:
            z = _load(checkpoint)
        self.g = weights(INTERMEDIATE_Z if z is None else z)
        super().__init__(seed=seed, max_offers=max_offers, weights={k: self.g[k] for k in W_KEYS})

    # same as IntermediateAgent.setup_settlement with the constants as genes
    def setup_settlement(self, g, me, legal):
        w = self.g
        bi = C.board_info(g.layout)
        have = [0] * 5
        for s in g.players[me].settlements:
            for r in range(5):
                have[r] += bi.vertex_pips[s][r]
        out = []
        for a in legal:
            v = a.value
            pips = bi.vertex_pips[v]
            s = w["setup_pips"] * sum(pips[r] * bi.scarcity[r] for r in range(5))
            s += w["setup_div"] * sum(1 for x in pips if x)
            tot = [have[r] + pips[r] for r in range(5)]
            s += w["setup_comb_div"] * sum(1 for x in tot if x)
            if tot[0] and tot[1]:
                s += w["setup_wb"]
            if tot[3] and tot[4]:
                s += w["setup_wo"]
            port = bi.vertex_port.get(v)
            if port == GENERIC_PORT:
                s += w["setup_port3"]
            elif port:
                s += w["setup_port2"] * tot[RESOURCES.index(port)]
            exp = 0.0
            for u in T.vertex_neighbors[v]:
                for x in T.vertex_neighbors[u]:
                    if x != v and C.spot_free(g, x) and x not in T.vertex_neighbors[v]:
                        exp = max(exp, sum(bi.vertex_pips[x]))
            s += w["setup_exp"] * exp
            out.append((s, a))
        return out

    def hex_block_score(self, g, me, h, bi):
        if any(g.vertex_owner[v] == me for v in T.hex_vertices[h]):
            return -100.0
        r = bi.hex_res[h]
        if r < 0:
            return -50.0
        s, victims = 0.0, False
        for v in T.hex_vertices[h]:
            o = g.vertex_owner[v]
            if o != -1:
                s += bi.hex_pips[h] * g.vertex_level[v] * bi.scarcity[r] * (1 + self.g["rob_vp"] * g.public_victory_points(o))
                victims = victims or C.hand_count(g, o) > 0
        return s + (self.g["rob_victims"] if victims else 0.0)

    def trade_response(self, g, me, legal):
        tr = g.trade
        out = [(0.0, Action(me, A.REJECT_TRADE))]
        acc = Action(me, A.ACCEPT_TRADE)
        has_acc = (len(legal) > 1 and legal[1] == acc) or acc in legal
        if has_acc and self.partner_ok(g, me, tr["proposer"]):
            d = self.trade_delta(g, me, tr["get"], tr["give"])
            lead = max(0, g.public_victory_points(tr["proposer"]) - g.victory_points(me))
            if d > self.g["accept_margin"] + self.g["margin_per_vp"] * lead:
                out.append((d, acc))
        return out

    def offers(self, g, me):
        # shortcut: intermediate already drops offers below 0.05, so offer_min < 0.05 acts as 0.05
        lo = self.g["offer_min"]
        return [c for c in super().offers(g, me) if c[0] > lo]

    def candidate_values(self, g, me, legal, bi=None):
        out = super().candidate_values(g, me, legal, bi)
        kb = self.g["knight_bonus"] - 0.4  # intermediate adds a fixed 0.4 to knight plays
        return [(v + kb, a) if a.type == A.PLAY_KNIGHT else (v, a) for v, a in out]
