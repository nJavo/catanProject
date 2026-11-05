"""Learn a vision profile from labelled screenshots.

    python -m catan.vision.calibrate --out data/screenshots/colonist.npz shot1.png [shot2.png ...]

Each image needs a PositionSpec beside it with the same stem (``shot1.json``): the
correct position, typically exported from the Analysis editor after fixing an
import by hand. The lattice is found from the labels alone (``fisher_scorer``),
so a new client needs no hand-placed points. Seat colours are learned from the
labelled owners: seat i is whatever colour seat i's pieces have in the labels.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from catan.engine import TOPOLOGY as T
from . import detect as D

NB = 4096


def fisher_scorer(terrain: list[str]):
    """Score map that peaks where per-hex mean colours separate the labelled terrains best."""
    cls = np.array([D.TERRAINS.index(t) for t in terrain])

    def make(small):
        def fn(R):
            h = max(1, round(0.15 * R))
            M = np.stack([D._boxsum(small[..., c], h) for c in range(3)], -1) / (2 * h + 1) ** 2
            sq = np.zeros(small.shape[:2], np.float32)
            S = np.zeros((6, *small.shape), np.float32)
            for k, (x, y) in enumerate(D.HEX_XY):
                m = sum(np.stack([D._shift(M[..., c], round(R * (y + oy)), round(R * (x + ox))) for c in range(3)], -1)
                        for ox, oy in D.SAMPLE) / len(D.SAMPLE)
                sq += (m * m).sum(-1)
                S[cls[k]] += m
            n_t = np.bincount(cls, minlength=6).astype(np.float32)
            within = sq - sum((S[t] ** 2).sum(-1) / n_t[t] for t in range(6) if n_t[t])
            total = sq - (S.sum(0) ** 2).sum(-1) / len(cls)
            return (total - within) / (within + 50.0 * len(cls))
        return fn
    return make


def _count(px, out, c):
    out[c] += np.bincount(px.ravel(), minlength=NB)


def calibrate(pairs, robber_offset=(-0.5, 0.01), port_offset=0.62, log=print):
    """``pairs``: [(rgb array, spec)]. Returns the profile dict (save with ``np.savez_compressed``).
    Lattices come from the labels (``fisher_scorer``), then are snapped to the tokens with a first
    profile, and everything is learned again on the snapped lattices."""
    lats = [D.fit_lattice(img, fisher_scorer(spec["layout"]["hex_terrain"])) for img, spec in pairs]
    prof = learn(pairs, lats, robber_offset, port_offset, log=lambda *_: None)
    lats = [D.refine_lattice(img, lat, prof) for (img, _), lat in zip(pairs, lats)]
    for lat in lats:
        log(f"lattice cx={lat['cx']:.1f} cy={lat['cy']:.1f} r={lat['r']:.2f}")
    return learn(pairs, lats, robber_offset, port_offset, log)


def _hexnorm(u, v):
    """Pointy-top hex 'radius' of (u, v) around each hex centre, minimum over the 19 hexes (<= 1 is on a tile)."""
    out = np.full(u.shape, np.inf, np.float32)
    for x, y in D.HEX_XY:
        dx, dy = np.abs(u - x), np.abs(v - y)
        out = np.minimum(out, np.maximum(dx / (np.sqrt(3) / 2), dx / np.sqrt(3) + dy))
    return out


def learn(pairs, lats, robber_offset, port_offset, log, robber_offset_desert=(0.0, -0.04)):
    land = np.zeros((2, NB))            # not land, land
    terr = np.zeros((6, NB))
    piece = np.zeros((5, NB))           # background, seat 0..3
    rob = np.zeros((2, NB))             # other, robber
    for (img, spec), lat in zip(pairs, lats):
        idx = D.bins(img)
        H, W = idx.shape
        yy, xx = np.mgrid[0:H, 0:W]
        hn = _hexnorm((xx - lat["cx"]) / lat["r"], (yy - lat["cy"]) / lat["r"])
        _count(idx[hn < 0.85], land, 1)
        _count(idx[hn > 1.05], land, 0)  # the beach counts as not land: it is what sharpens the coastline
        lay = spec["layout"]
        for h, (x, y) in enumerate(D.HEX_XY):
            _count(D.region(idx, lat, x, y, 0.42, 0.72), terr, D.TERRAINS.index(lay["hex_terrain"][h]))
            if h != spec["robber"]:
                _count(D.region(idx, lat, x, y, 0, 0.85), rob, 0)
        rx, ry = D.HEX_XY[spec["robber"]] + (robber_offset if lay["hex_number"][spec["robber"]] else robber_offset_desert)
        _count(D.region(idx, lat, rx, ry, 0, 0.1), rob, 1)
        vo, eo = spec["vertex_owner"], spec["edge_owner"]
        for vtx, (x, y) in enumerate(D.VERT_XY):
            if vo[vtx] >= 0:
                _count(D.region(idx, lat, x, y - 0.02, 0, 0.09), piece, 1 + vo[vtx])
            elif all(eo[e] < 0 for e in T.vertex_edges[vtx]):
                _count(D.box(idx, lat, x, y, D.VERT_BOX), piece, 0)
        for e, (a, b) in enumerate(T.edge_vertices):
            if eo[e] >= 0:
                _count(D.band(idx, lat, D.VERT_XY[a], D.VERT_XY[b], half=0.03), piece, 1 + eo[e])
            else:
                _count(D.band(idx, lat, D.VERT_XY[a], D.VERT_XY[b], half=0.06), piece, 0)
    prof = {
        "land": (land[1] / (land.sum(0) + 1)).astype(np.float32),
        "terrain": ((terr + 0.5) / (terr + 0.5).sum(1, keepdims=True)).astype(np.float32),
        "piece": np.where(piece.sum(0) > 0, piece.argmax(0), 0).astype(np.uint8),
        "robber": (rob[1] / (rob.sum(0) + 1)).astype(np.float32),
        "port_offset": np.float32(port_offset),
    }
    # templates and piece thresholds need the colour tables, so a second pass
    nums = {n: [] for n in D.NUMBERS}
    ports = {k: [] for k in D.PORT_KINDS}
    none, vX, vy, ee, er = [], [], [], [], []
    for (img, spec), lat in zip(pairs, lats):
        lay = spec["layout"]
        for h in range(19):
            if lay["hex_number"][h]:
                nums[lay["hex_number"][h]].append(D.number_vec(img, lat, h))
        for e in D.COAST:
            vec = D.port_vec(img, lat, e, port_offset)
            if e in lay["port_edges"]:
                ports[lay["port_types"][lay["port_edges"].index(e)]].append(vec)
            else:
                none.append(vec)
        pcls = D.piece_classes(img, prof)
        for vtx in range(len(D.VERT_XY)):
            _, m = D.vertex_shape(pcls, lat, vtx)
            vX.append(m)
            vy.append(spec["vertex_level"][vtx])
        for e in range(T.num_edges):
            f = D.edge_frac(pcls, lat, e)
            o = spec["edge_owner"][e]
            (ee if o < 0 else er).append(f.max() if o < 0 else f[o])
    missing = [n for n, v in nums.items() if not v] + [k for k, v in ports.items() if not v]
    if missing or 1 not in vy or 2 not in vy or not er:
        raise ValueError(f"labels must show every number, port type, a settlement, a city and a road; missing {missing}")

    def unit(vs_):
        m = np.mean(vs_, 0)
        return m / np.linalg.norm(m)

    prof["numbers"] = np.stack([unit(nums[n]) for n in D.NUMBERS]).astype(np.float32)
    prof["ports"] = np.stack([unit(ports[k]) for k in D.PORT_KINDS]).astype(np.float32)
    prof["port_none"] = unit(none).astype(np.float32)
    prof["vertex_X"], prof["vertex_y"] = np.array(vX, np.uint8), np.array(vy, np.int64)
    # a road's colour fill varies a lot by seat (cream fills least), so the cut sits near the empty side
    lo, hi = np.percentile(ee, 99), np.percentile(er, 1)
    prof["edge_road"] = np.float32(lo + 0.3 * (hi - lo))
    log(f"vertices: {vy.count(1)} settlements, {vy.count(2)} cities; edge: empty<= {np.max(ee):.2f} road>= {np.min(er):.2f}")
    return prof


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("images", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--robber-offset", type=float, nargs=2, default=(-0.5, 0.01),
                    help="robber position relative to its hex centre in hex radii (our app draws it left of the token)")
    ap.add_argument("--port-offset", type=float, default=0.62,
                    help="port marker distance outward from its coastal edge, in hex radii")
    a = ap.parse_args()
    pairs = [(D.load_image(Path(p).read_bytes()), json.loads(Path(p).with_suffix(".json").read_text())) for p in a.images]
    prof = calibrate(pairs, tuple(a.robber_offset), port_offset=a.port_offset)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out, **prof)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
