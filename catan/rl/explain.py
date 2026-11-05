"""Model-derived diagnostics for RL policy decisions (AI lab inspector).

Everything here is computed from the network itself and labeled as such:

- ``candidates``: the policy's top-k legal actions and probabilities.
- ``value_probe``: the value head evaluated on the position after each
  candidate, averaged over a few determinizations of hidden information
  (opponents' hands, dev cards, deck order, dice), from the actor's point of view.
- ``attribution``: integrated gradients (baseline: same board, empty game state) of the chosen action's
  logit and of the value, summed over named observation blocks from
  ``spaces.OFFSETS``.

None of this is a reason the policy "had". Attributions measure local
sensitivity of the network output to input features relative to a baseline.
The probe asks the value head about positions it may rarely see in training.
It runs after the action is chosen and never touches the agent's RNG, so it
cannot change decisions.
"""
from __future__ import annotations

import random

import numpy as np
import torch

from catan.bots import common as C
from catan.bots.explain import action_label as nice_label
from catan.engine import A, TOPOLOGY as T

from .spaces import (OBS_DIM, OFFSETS, HEX_F, VSTAT_F, VDYN_F, EDGE_F, PUB_F, NV, NE, PHASES, action_label,
                     decode_action, encode_obs)

K = 5
DETERMINIZATIONS = 4
IG_STEPS = 64
LABEL = "model-derived attribution (integrated gradients)"
CAVEATS = [
    "Attributions show how sensitive the network output is to each input block, relative to the same board "
    "with an empty game state. They are not the policy's reasons.",
    "The board layout is held fixed, so layout features get no attribution. The straight path from the "
    "empty state passes through positions that cannot occur in a game.",
    "Blocks are correlated; a large attribution on one block may stand in for another.",
    "Value probes evaluate the critic on post-action positions from this seat's view. Positions where it "
    "is not this seat's turn are rare in training, so those estimates are less reliable.",
]


def _blocks(target_vertices: set[int], target_edges: set[int], target_hexes: set[int],
            obs_dim: int = 0) -> dict[str, np.ndarray]:
    """Named observation blocks -> index arrays. 'near candidate' = the chosen action's location
    plus adjacent vertices and the edges between them."""
    idx: dict[str, list[int]] = {}

    def add(name, rng):
        idx.setdefault(name, []).extend(rng)

    a, _ = OFFSETS["hex"]
    for h in range(T.num_hexes):
        base = a + h * HEX_F
        add("Robber position", [base + HEX_F - 1])
        add("Board layout near candidate" if h in target_hexes else "Board layout", range(base, base + HEX_F - 1))
    a, b = OFFSETS["vertex_static"]
    add("Vertex pips and ports", range(a, b))
    a, _ = OFFSETS["vertex_dyn"]
    for v in range(NV):
        near = v in target_vertices
        add("Buildings near candidate" if near else "Buildings elsewhere", range(a + v * VDYN_F, a + v * VDYN_F + VDYN_F - 1))
        add("Free spots", [a + v * VDYN_F + VDYN_F - 1])
    a, _ = OFFSETS["edge"]
    for e in range(NE):
        add("Roads near candidate" if e in target_edges else "Roads elsewhere", range(a + e * EDGE_F, a + (e + 1) * EDGE_F))
    a, _ = OFFSETS["players"]
    add("My public stats", range(a, a + PUB_F))
    add("Opponents' public counts", range(a + PUB_F, a + 4 * PUB_F))
    a, _ = OFFSETS["own"]
    add("Own hand", range(a, a + 5))
    add("Own dev cards", range(a + 5, a + 15))
    add("Own VP", [a + 15])
    a, _ = OFFSETS["global"]
    np_ = len(PHASES)
    add("Bank", range(a, a + 5))
    add("Dev deck size", [a + 5])
    add("Phase", range(a + 6, a + 6 + np_))
    o = a + 6 + np_
    add("Rolled / dev played this turn", range(o, o + 2))
    add("Last roll", range(o + 2, o + 13))
    add("Turn counters", range(o + 13, o + 16))
    a, b = OFFSETS["trade"]
    add("Trade state", range(a, b))
    if obs_dim > OBS_DIM:  # obs v2 (catan/research/arch/obs.py): derived planning features
        from catan.research.arch import obs as O2
        for name, label in (("vertex2", "Road distances, robber-aware pips (v2)"),
                            ("edge2", "Buildable roads (v2)"), ("players2", "Derived player features (v2)")):
            a, b = O2.OFFSETS[name]
            add(label, range(a, b))
    return {k: np.asarray(v, dtype=np.int64) for k, v in idx.items() if v}


def entity_index(obs_dim: int) -> dict[str, list[np.ndarray]]:
    """Observation indices that describe each board entity: hexes (terrain, number, robber),
    vertices (static pips/ports, buildings, free spot, and v2 distances), edges (roads, v2 buildable)."""
    hexes = [np.arange(OFFSETS["hex"][0] + h * HEX_F, OFFSETS["hex"][0] + (h + 1) * HEX_F) for h in range(T.num_hexes)]
    vs, vd, ed = OFFSETS["vertex_static"][0], OFFSETS["vertex_dyn"][0], OFFSETS["edge"][0]
    verts = [[np.arange(vs + v * VSTAT_F, vs + (v + 1) * VSTAT_F), np.arange(vd + v * VDYN_F, vd + (v + 1) * VDYN_F)]
             for v in range(NV)]
    edges = [[np.arange(ed + e * EDGE_F, ed + (e + 1) * EDGE_F)] for e in range(NE)]
    if obs_dim > OBS_DIM:
        from catan.research.arch import obs as O2
        a2, b2 = O2.OFFSETS["vertex2"][0], O2.OFFSETS["edge2"][0]
        for v in range(NV):
            verts[v].append(np.arange(a2 + v * O2.VX, a2 + (v + 1) * O2.VX))
        for e in range(NE):
            edges[e].append(np.arange(b2 + e * O2.EX, b2 + (e + 1) * O2.EX))
    return {"hexes": hexes, "vertices": [np.concatenate(x) for x in verts], "edges": [np.concatenate(x) for x in edges]}


def attribute(model, obs_row: torch.Tensor, index: int, static_len: int, chosen=None, top: int = 6) -> dict:
    """Integrated-gradient attribution of logit[index] and the value, by observation block and by board
    entity (the top hexes, vertices and edges by |logit attribution|, for board highlights)."""
    base = layout_baseline(obs_row, static_len)
    # Double the steps until both completeness gaps are within 10% (the GNN's max-pooling makes the
    # path integrand kinked, so it converges more slowly than the MLP); the final gap is reported.
    for steps in (IG_STEPS, 2 * IG_STEPS, 4 * IG_STEPS, 8 * IG_STEPS):
        al, av, (l_x, l_0), (v_x, v_0) = integrated_gradients(model, obs_row, index, base, steps)
        if max(abs(l_x - l_0 - al.sum()) / max(1.0, abs(l_x - l_0)),
               abs(v_x - v_0 - av.sum()) / max(1.0, abs(v_x - v_0))) <= 0.1:
            break
    dim = obs_row.shape[-1]
    blocks = _blocks(*_near(chosen), obs_dim=dim) if chosen is not None else _blocks(set(), set(), set(), obs_dim=dim)
    attr = [{"block": k, "logit": _r(al[ix].sum()), "value": _r(av[ix].sum())} for k, ix in blocks.items()]
    attr = [b for b in attr if abs(b["logit"]) >= 5e-5 or abs(b["value"]) >= 5e-5]
    attr.sort(key=lambda b: -abs(b["logit"]))
    ents = {}
    for kind, rows in entity_index(dim).items():
        vals = [(i, float(al[ix].sum()), float(av[ix].sum())) for i, ix in enumerate(rows)]
        vals = [x for x in vals if abs(x[1]) >= 5e-4 or abs(x[2]) >= 5e-4]
        vals.sort(key=lambda x: -abs(x[1]))
        ents[kind] = [{"id": i, "logit": _r(lg), "value": _r(vl)} for i, lg, vl in vals[:top]]
    return {
        "label": LABEL, "method": "integrated gradients", "baseline": "same board layout, empty game state",
        "steps": steps,
        "logit": {"at_obs": _r(l_x), "at_baseline": _r(l_0), "sum": _r(al.sum()), "completeness_gap": _r(l_x - l_0 - al.sum())},
        "value": {"at_obs": _r(v_x), "at_baseline": _r(v_0), "sum": _r(av.sum()), "completeness_gap": _r(v_x - v_0 - av.sum())},
        "blocks": attr, "entities": ents,
        "near_candidate": sorted(_near(chosen)[0]) if chosen is not None else [],
    }


def _near(action) -> tuple[set, set, set]:
    t, v = action.type, action.value
    verts: set[int] = set()
    hexes: set[int] = set()
    if t in (A.BUILD_SETTLEMENT, A.BUILD_CITY):
        verts = {v}
    elif t == A.BUILD_ROAD:
        verts = set(T.edge_vertices[v])
    elif t == A.MOVE_ROBBER:
        verts, hexes = set(T.hex_vertices[v]), {v}
    if not verts:
        return set(), set(), set()
    ring = set(verts)
    for x in verts:
        ring.update(T.vertex_neighbors[x])
        hexes.update(T.vertex_hexes[x])
    edges = {e for e in range(NE) if set(T.edge_vertices[e]) <= ring}
    return ring, edges, hexes


def integrated_gradients(model, obs: torch.Tensor, index: int, baseline: torch.Tensor, steps: int = IG_STEPS):
    """IG of (logit[index], value) w.r.t. obs along the straight path from ``baseline``.
    Returns (attr_logit, attr_value, (logit(x), logit(b)), (value(x), value(b))); midpoint rule."""
    alphas = ((torch.arange(steps, device=obs.device, dtype=obs.dtype) + 0.5) / steps)[:, None]
    diff = obs - baseline
    path = (baseline[None] + alphas * diff[None]).detach().requires_grad_(True)
    with torch.enable_grad():
        logits, v = model(path)
        gl = torch.autograd.grad(logits[:, index].sum(), path, retain_graph=True)[0]
        gv = torch.autograd.grad(v.sum(), path)[0]
    with torch.no_grad():
        ends_l, ends_v = model(torch.stack([obs, baseline]))
    return (diff * gl.mean(0)).cpu().numpy(), (diff * gv.mean(0)).cpu().numpy(), \
        (float(ends_l[0, index]), float(ends_l[1, index])), (float(ends_v[0]), float(ends_v[1]))


def layout_baseline(obs: torch.Tensor, static_len: int) -> torch.Tensor:
    """The same board layout with an empty game state: static features kept, everything else zero.
    (An all-zero observation is degenerate for this network: its first LayerNorm sees almost
    constant activations, and IG along that path does not converge.)"""
    b = torch.zeros_like(obs)
    b[:static_len] = obs[:static_len]
    return b


def _r(x):
    return round(float(x), 4)


def explain_policy(agent, game, legal, obs, probs, chosen_index) -> dict:
    """``obs`` is the (1, OBS_DIM) tensor and ``probs`` the masked softmax used for the decision."""
    me, n = game.actor, game.n
    model = agent.model
    chosen = decode_action(chosen_index, me, n)

    top = torch.topk(probs, min(K, len(legal)))
    idxs = [chosen_index] + [int(j) for j in top.indices if int(j) != chosen_index][:K - 1]  # chosen first
    cands = [decode_action(j, me, n) for j in idxs]

    # value probe: same determinization seeds for every candidate (common random numbers); a fixed,
    # position-derived seed keeps it reproducible without touching the agent's generator
    seeds = [hash((len(game.action_log), game.turn, d)) & 0xFFFFFFFF for d in range(DETERMINIZATIONS)]
    static = agent._static
    encode = getattr(agent, "encode", encode_obs)  # arch_rl agents carry their own (obs v2) encoder
    rows, encs = [], []
    for a in cands:
        for s in seeds:
            c = C.determinize(game, me, random.Random(s))
            c.step(a)
            encs.append(encode(c, me, static))
    with torch.no_grad():
        _, vals = model(torch.from_numpy(np.stack(encs)).to(agent.device))
    vals = vals.float().cpu().numpy().reshape(len(cands), len(seeds))
    attribution = attribute(model, obs[0], chosen_index, len(static), chosen)
    v_x = attribution["value"]["at_obs"]
    v_now = v_x
    for j, a, vs in zip(idxs, cands, vals):
        rows.append({"action": a.to_json(), "label": nice_label(a), "key": action_label(j), "prob": _r(probs[j]),
                     "value_after": _r(vs.mean()), "value_sd": _r(vs.std(ddof=1)) if len(vs) > 1 else 0.0,
                     "n": len(vs), "delta": _r(vs.mean() - v_now)})

    out = {"kind": "policy_model", "phase": game.phase, "value": _r(v_now), "candidates": rows,
           "value_probe": {"label": "value-network estimate", "determinizations": len(seeds),
                           "perspective": "acting seat", "note": CAVEATS[3]},
           "attribution": attribution, "caveats": CAVEATS}
    out["summary"] = _summary(out)
    out["summary_source"] = "generated from model outputs"
    return out


def _fmt(x):
    return f"{x:+.2f}".replace("-", "−")


def _summary(e):
    cs = e["candidates"]
    c = cs[0]
    s = f"Policy chose {c['label']} with probability {c['prob']:.1%}"  # DS30: % like the candidate table
    if len(cs) > 1:
        s += f" (next: {cs[1]['label']} {cs[1]['prob']:.1%})"
    num = lambda x: f"{x:.2f}".replace("-", "\u2212")
    s += f". Value estimate {num(e['value'])}; value-network probe after it {num(c['value_after'])} ({_fmt(c['delta'])})"
    if len(cs) > 1:
        s += f" vs {num(cs[1]['value_after'])} after the next choice"
    top = [b for b in e["attribution"]["blocks"][:2]]
    if top:
        s += ". Largest logit attributions: " + ", ".join(f"{b['block']} {_fmt(b['logit'])}" for b in top)
    return s + ". Attributions show model sensitivity, not reasons."
