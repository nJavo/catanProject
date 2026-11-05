"""Policy architectures for the D5 comparison. All take the flat observation
(v1 = catan.rl.spaces, v2 = catan.research.arch.obs) and return (logits, value)
like catan.rl.model.PolicyNet, so catan.rl.ppo.ppo_update and catan.rl.bc.fit work
unchanged.

- ``mlp``: catan.rl.model.PolicyNet (residual MLP baseline).
- ``gnn``: cross-dimensional graph network over the hex/vertex/edge topology
  (Gendre & Kaneko 2020 "Xdim" idea on a true graph instead of a brick grid):
  node streams for 19 hexes, 54 vertices, 72 edges plus a scalar stream for
  players/hand/phase/trade. Each layer passes mean messages along the board
  incidence (hex-vertex, vertex-edge, vertex-vertex), inflates the scalar stream
  into every node, and deflates nodes back into it with mean+max pooling.
  ``head="node"`` scores roads from edge embeddings, settlements/cities from
  vertex embeddings and robber moves from hex embeddings (one shared linear
  scorer per action type, a pointer-style head as in AlphaStar); the 179
  non-board actions come from the scalar stream. ``head="flat"`` scores all 378
  actions from the pooled state (ablation).
"""
from __future__ import annotations

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from catan.engine import A, TOPOLOGY as T
from catan.rl import spaces as S
from catan.rl.model import PolicyNet, MASK_VALUE
from . import obs as O

NH, NV, NE = S.NH, S.NV, S.NE
N_BOARD = NE + 2 * NV + NH          # road, settlement, city, robber blocks lead ACTIONS
assert [t for t, _ in S.ACTIONS[:N_BOARD]] == [A.BUILD_ROAD] * NE + [A.BUILD_SETTLEMENT] * NV \
    + [A.BUILD_CITY] * NV + [A.MOVE_ROBBER] * NH
assert [v for _, v in S.ACTIONS[:N_BOARD]] == [*range(NE), *range(NV), *range(NV), *range(NH)]


def _mean_matrix(rows: int, cols: int, pairs) -> torch.Tensor:
    m = torch.zeros(rows, cols)
    for i, j in pairs:
        m[i, j] = 1
    return m / m.sum(1, keepdim=True).clamp(min=1)


M_VH = _mean_matrix(NV, NH, [(v, h) for v in range(NV) for h in T.vertex_hexes[v]])
M_HV = _mean_matrix(NH, NV, [(h, v) for h in range(NH) for v in T.hex_vertices[h]])
M_VE = _mean_matrix(NV, NE, [(v, e) for v in range(NV) for e in T.vertex_edges[v]])
M_EV = _mean_matrix(NE, NV, [(e, v) for e in range(NE) for v in T.edge_vertices[e]])
M_VV = _mean_matrix(NV, NV, [(v, u) for v in range(NV) for u in T.vertex_neighbors[v]])


def _sl(name, src=S.OFFSETS):
    a, b = src[name]
    return slice(a, b)


class Split(nn.Module):
    """Flat obs -> (hex [B,NH,*], vertex [B,NV,*], edge [B,NE,*], scalar [B,*])."""

    def __init__(self, v2: bool):
        super().__init__()
        self.v2 = v2
        self.hex_f = S.HEX_F
        self.vert_f = S.VSTAT_F + S.VDYN_F + (O.VX if v2 else 0)
        self.edge_f = S.EDGE_F + (O.EX if v2 else 0)
        self.scal_f = S.MAXP * S.PUB_F + S.OWN_F + S.GLOB_F + S.TRADE_F + (S.MAXP * O.PX if v2 else 0)

    def forward(self, x):
        B = x.shape[0]
        h = x[:, _sl("hex")].reshape(B, NH, -1)
        v = [x[:, _sl("vertex_static")].reshape(B, NV, -1), x[:, _sl("vertex_dyn")].reshape(B, NV, -1)]
        e = [x[:, _sl("edge")].reshape(B, NE, -1)]
        s = [x[:, S.OFFSETS["players"][0]:S.OBS_DIM]]
        if self.v2:
            v.append(x[:, _sl("vertex2", O.OFFSETS)].reshape(B, NV, -1))
            e.append(x[:, _sl("edge2", O.OFFSETS)].reshape(B, NE, -1))
            s.append(x[:, _sl("players2", O.OFFSETS)])
        return h, torch.cat(v, -1), torch.cat(e, -1), torch.cat(s, -1)


class XLayer(nn.Module):
    def __init__(self, d: int, dg: int):
        super().__init__()
        self.nv, self.ne, self.nh, self.ng = (nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d),
                                              nn.LayerNorm(dg))
        self.v = nn.Linear(4 * d, d)          # self, hexes, edges, neighbour vertices
        self.e = nn.Linear(2 * d, d)          # self, endpoints
        self.h = nn.Linear(2 * d, d)          # self, corners
        self.g2n = nn.Linear(dg, 3 * d)       # inflation into the three node types
        self.g = nn.Linear(dg + 6 * d, dg)    # deflation: mean+max of each node type
        # buffers so .to(device) moves the incidence matrices; non-persistent keeps state_dicts unchanged
        for k, m in {"m_vh": M_VH, "m_hv": M_HV, "m_ve": M_VE, "m_ev": M_EV, "m_vv": M_VV}.items():
            self.register_buffer(k, m, persistent=False)

    def forward(self, hv, he, hh, g):
        v, e, h, gg = self.nv(hv), self.ne(he), self.nh(hh), self.ng(g)
        gv, ge, gh = self.g2n(gg).unsqueeze(1).chunk(3, -1)
        hv = hv + torch.relu(self.v(torch.cat([v, self.m_vh @ h, self.m_ve @ e, self.m_vv @ v], -1)) + gv)
        he = he + torch.relu(self.e(torch.cat([e, self.m_ev @ v], -1)) + ge)
        hh = hh + torch.relu(self.h(torch.cat([h, self.m_hv @ v], -1)) + gh)
        pool = [hv.mean(1), hv.amax(1), he.mean(1), he.amax(1), hh.mean(1), hh.amax(1)]
        g = g + torch.relu(self.g(torch.cat([gg] + pool, -1)))
        return hv, he, hh, g


class GraphPolicy(nn.Module):
    def __init__(self, v2: bool = True, d: int = 64, dg: int = 256, layers: int = 3, head: str = "node",
                 grad_checkpoint: int | None = None):
        super().__init__()
        # layers recomputed in backward (activation checkpointing); None = all.
        # Fewer = faster updates, more memory (~90 MB per layer at minibatch 1024).
        # Same math either way: gradients are bit-identical.
        self.grad_checkpoint = layers if grad_checkpoint is None else int(grad_checkpoint)
        self.split = Split(v2)
        sp = self.split
        self.emb_v = nn.Linear(sp.vert_f, d)
        self.emb_e = nn.Linear(sp.edge_f, d)
        self.emb_h = nn.Linear(sp.hex_f, d)
        self.emb_g = nn.Linear(sp.scal_f, dg)
        # learned node identities (absolute board position: coast, centre, port slots)
        self.pos_v = nn.Parameter(torch.zeros(NV, d))
        self.pos_e = nn.Parameter(torch.zeros(NE, d))
        self.pos_h = nn.Parameter(torch.zeros(NH, d))
        self.layers = nn.ModuleList([XLayer(d, dg) for _ in range(layers)])
        self.head = head
        self.norm_v, self.norm_e, self.norm_h, self.norm_g = (nn.LayerNorm(d), nn.LayerNorm(d),
                                                              nn.LayerNorm(d), nn.LayerNorm(dg))
        if head == "node":
            self.pi_e = nn.Linear(d, 1)
            self.pi_v = nn.Linear(d, 2)
            self.pi_h = nn.Linear(d, 1)
            self.pi_g = nn.Linear(dg, S.ACT_DIM - N_BOARD)
            pis = (self.pi_e, self.pi_v, self.pi_h, self.pi_g)
        else:
            self.pi = nn.Linear(dg, S.ACT_DIM)
            pis = (self.pi,)
        for p in pis:
            nn.init.orthogonal_(p.weight, 0.01)
            nn.init.zeros_(p.bias)
        self.v = nn.Sequential(nn.Linear(dg, dg), nn.ReLU(), nn.Linear(dg, 1))

    def forward(self, obs, mask=None):
        h, v, e, s = self.split(obs)
        hv = self.emb_v(v) + self.pos_v
        he = self.emb_e(e) + self.pos_e
        hh = self.emb_h(h) + self.pos_h
        g = self.emb_g(s)
        for i, layer in enumerate(self.layers):
            if torch.is_grad_enabled() and i < self.grad_checkpoint:  # recompute in backward: saves memory
                hv, he, hh, g = checkpoint(layer, hv, he, hh, g, use_reentrant=False)
            else:
                hv, he, hh, g = layer(hv, he, hh, g)
        g = self.norm_g(g)
        if self.head == "node":
            lv = self.pi_v(self.norm_v(hv))                      # [B,NV,2]
            logits = torch.cat([self.pi_e(self.norm_e(he)).squeeze(-1), lv[..., 0], lv[..., 1],
                                self.pi_h(self.norm_h(hh)).squeeze(-1), self.pi_g(g)], -1)
        else:
            logits = self.pi(g)
        if mask is not None:
            logits = logits.masked_fill(~mask, MASK_VALUE)
        return logits, self.v(g).squeeze(-1)


def obs_dim(obs_version: int) -> int:
    return O.OBS_DIM if obs_version == 2 else S.OBS_DIM


def build(spec: dict) -> nn.Module:
    """spec = {"arch": "mlp"|"gnn", "obs": 1|2, ...arch kwargs}."""
    kw = {k: v for k, v in spec.items() if k not in ("arch", "obs")}
    if spec["arch"] == "mlp":
        return PolicyNet(obs_dim(spec["obs"]), S.ACT_DIM, **kw)
    if spec["arch"] == "gnn":
        return GraphPolicy(v2=spec["obs"] == 2, **kw)
    raise ValueError(f"unknown arch {spec['arch']!r}")


def param_count(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())
