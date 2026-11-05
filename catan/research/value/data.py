"""Value-net data: generate outcome-labelled obs-v2 states, and pack shards into one memmap.

    python -m catan.research.value.data gen --games 2400 --out runs/value_data/gen --workers 3
    python -m catan.research.value.data pack --out runs/value_data/packed \
        --src runs/value_data/gen:1.0 /abs/.../bc_data/int_v2:0.4 /abs/.../bc_data/dagger_g1:1.0

``gen`` plays mixed tables (search, the nego champion, the D5 GNN, heuristics) and keeps a random
``keep`` fraction of every seat's non-forced decisions. Each row: obs (float16), z (that seat
won), gid (game seed), seat, turn, src (table index), margin (intermediate's evaluate margin, used
only as the calibration baseline). Shards of ``chunk`` games keep worker memory small.

``pack`` streams shards into ``obs.f16`` (N x OBS_DIM memmap) + ``meta.npz``; a ``:frac`` suffix
keeps that fraction of each source's games (whole games, chosen by seed hash). D2 shards
(``win``/``seed``) and ours (``z``/``gid``) are both read.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import signal
import sys
import time

import numpy as np

GEN_SEED_BASE = 70_000_000  # >= 1e7; D2 uses 30M-52M, D4 ExIt 63M
NEGO = "nego:checkpoint=models/nego_best.pt"
GNN = "arch_rl:deterministic=true"
TABLES = [
    ["exit_search:sims=16", "intermediate", "basic", NEGO],
    [GNN, "intermediate", "intermediate", "basic"],
    [NEGO, "intermediate", "intermediate", "intermediate"],
    [GNN, NEGO, "basic", "basic"],
    ["intermediate_trader", "basic", "intermediate", GNN],
    ["exit_search:checkpoint=models/arch_best.pt,sims=16", "intermediate", NEGO, GNN],
]
COMBO = "combo:checkpoint=models/combo_best.pt,deterministic=true"  # champion combo_zs-v1
LEAGUE = "arch_rl:checkpoint=models/league_best.pt,deterministic=true"
TABLE_SETS = {
    "base": TABLES,
    "combo": [  # strongest play: the champion against everyone else
        [COMBO, "intermediate", "intermediate", "intermediate"],
        [COMBO, NEGO, "intermediate", "basic"],
        [COMBO, LEAGUE, NEGO, "intermediate_trader"],
        [COMBO, COMBO, NEGO, LEAGUE],
        [LEAGUE, "intermediate", NEGO, "basic"],
    ],
}


def _chunk(job):
    seeds, keep, out, tables = job
    TABLES = TABLE_SETS[tables]  # noqa: N806 (shadows the module default on purpose)
    import torch
    torch.set_num_threads(1)
    from catan.agents.registry import make_agent
    from catan.bots import common as C
    from catan.bots.intermediate import evaluate
    from catan.engine import Game, GameConfig
    from catan.eval.tournament import parse_spec
    from catan.research.arch.obs import encode_obs
    from catan.rl.spaces import static_features

    cache = {}
    rows = {k: [] for k in ("obs", "z", "gid", "seat", "turn", "src", "margin")}
    for seed in seeds:
        rng = random.Random(seed)
        ti = rng.randrange(len(TABLES))
        lineup = TABLES[ti][:]
        rng.shuffle(lineup)
        agents = []
        for s, spec in enumerate(lineup):
            if (spec, s) not in cache:
                name, kw = parse_spec(spec)
                cache[(spec, s)] = make_agent(name, **kw)
            agents.append(cache[(spec, s)])
            agents[-1].reset(s, seed * 10 + s)
        g = Game(config=GameConfig(max_turns=500), seed=seed)
        st, bi = static_features(g.layout), C.board_info(g.layout)
        seats = []
        while not g.is_over:
            legal = g.legal_actions()
            me = g.actor
            if len(legal) > 1 and rng.random() < keep:
                rows["obs"].append(encode_obs(g, me, st).astype(np.float16))
                rows["margin"].append(evaluate(g, me, bi) - 0.3 * max(evaluate(g, q, bi) for q in range(g.n) if q != me))
                rows["turn"].append(g.turn)
                seats.append(me)
            g.step(agents[me].act(g, legal))
        for me in seats:
            rows["z"].append(float(g.winner == me))  # truncated games: everyone lost
            rows["seat"].append(me)
            rows["gid"].append(seed)
            rows["src"].append(ti)
    path = os.path.join(out, f"shard_{seeds[0]}.npz")
    np.savez(path, obs=np.stack(rows["obs"]), z=np.asarray(rows["z"], np.float32),
             gid=np.asarray(rows["gid"], np.int64), seat=np.asarray(rows["seat"], np.int8),
             turn=np.asarray(rows["turn"], np.int16), src=np.asarray(rows["src"], np.int8),
             margin=np.asarray(rows["margin"], np.float32))
    return path, len(rows["z"])


def gen(a):
    os.makedirs(a.out, exist_ok=True)
    done = {int(os.path.basename(p)[6:-4]) for p in glob.glob(os.path.join(a.out, "shard_*.npz"))}
    seeds = [GEN_SEED_BASE + a.offset + i for i in range(a.games)]
    jobs = [(seeds[i:i + a.chunk], a.keep, a.out, a.tables) for i in range(0, len(seeds), a.chunk)
            if seeds[i] not in done]  # resumable
    import multiprocessing as mp
    t0, n = time.time(), 0
    with mp.get_context("spawn").Pool(a.workers) as pool:
        it = pool.imap_unordered(_chunk, jobs)
        for k in range(len(jobs)):
            path, rows = it.next(timeout=10**6)  # timed wait: SIGTERM handler can run
            n += rows
            print(json.dumps({"shard": path, "rows": rows, "done": k + 1, "of": len(jobs),
                              "total_rows": n, "elapsed_s": round(time.time() - t0)}), flush=True)


def _hash_keep(gid, frac):
    """Deterministic per-game subsample (whole games)."""
    return ((gid * 2654435761) % 1000) < frac * 1000


def pack(a):
    from catan.research.arch.obs import OBS_DIM
    os.makedirs(a.out, exist_ok=True)
    srcs = []
    for s in a.src:
        path, _, frac = s.rpartition(":")
        srcs.append((path, float(frac)) if path else (s, 1.0))
    files = [(si, f) for si, (p, _) in enumerate(srcs) for f in sorted(glob.glob(os.path.join(p, "shard_*.npz")))]
    # pass 1: sizes (reads only small arrays)
    sizes = []
    for si, f in files:
        with np.load(f) as d:
            gid = d["gid"] if "gid" in d.files else d["seed"]
            sizes.append(int(_hash_keep(gid, srcs[si][1]).sum()))
    n = sum(sizes)
    # plain sequential writes of a valid .npy: a w+ memmap kept ~6 GB of dirty pages resident
    f_obs = open(os.path.join(a.out, "obs.npy"), "wb")
    np.lib.format.write_array_header_1_0(f_obs, {"descr": "<f2", "fortran_order": False, "shape": (n, OBS_DIM)})
    meta = {k: np.zeros(n, t) for k, t in (("z", np.float32), ("gid", np.int64), ("turn", np.int16),
                                            ("source", np.int8), ("margin", np.float32))}
    meta["margin"][:] = np.nan
    i = 0
    for (si, f), k in zip(files, sizes):
        with np.load(f) as d:
            ours = "gid" in d.files
            gid = d["gid"] if ours else d["seed"]
            m = _hash_keep(gid, srcs[si][1])
            f_obs.write(np.ascontiguousarray(d["obs"][m], "<f2").tobytes())
            meta["z"][i:i + k] = (d["z"] if ours else d["win"])[m]
            meta["gid"][i:i + k] = gid[m]
            meta["turn"][i:i + k] = d["turn"][m]
            meta["source"][i:i + k] = si
            if ours:
                meta["margin"][i:i + k] = d["margin"][m]
        i += k
    f_obs.close()
    np.savez(os.path.join(a.out, "meta.npz"), **meta, sources=np.array([p for p, _ in srcs]))
    print(json.dumps({"rows": n, "games": int(len(np.unique(meta["gid"]))),
                      "per_source": {srcs[s][0]: int((meta["source"] == s).sum()) for s in range(len(srcs))}}))


class ObsFile:
    """Row reader for a packed obs.npy using pread: no mmap, so RSS stays at the batch size."""

    def __init__(self, path):
        with open(path, "rb") as f:
            np.lib.format.read_magic(f)
            self.shape, _, dtype = np.lib.format.read_array_header_1_0(f)
            self.offset = f.tell()
        self.dtype = np.dtype(dtype)
        self.row = self.shape[1] * self.dtype.itemsize
        self.fd = os.open(path, os.O_RDONLY)

    def __len__(self):
        return self.shape[0]

    def rows(self, idx):
        """Rows for sorted indices; contiguous runs are read in one call."""
        idx = np.asarray(idx)
        out = np.empty((len(idx), self.shape[1]), self.dtype)
        brk = np.flatnonzero(np.diff(idx) != 1) + 1
        for s, e in zip(np.r_[0, brk], np.r_[brk, len(idx)]):
            buf = os.pread(self.fd, self.row * (e - s), self.offset + int(idx[s]) * self.row)
            out[s:e] = np.frombuffer(buf, self.dtype).reshape(e - s, self.shape[1])
        return out


def main(argv=None):
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gen")
    g.add_argument("--games", type=int, default=2400)
    g.add_argument("--offset", type=int, default=0)
    g.add_argument("--chunk", type=int, default=40)
    g.add_argument("--keep", type=float, default=0.15)
    g.add_argument("--tables", choices=sorted(TABLE_SETS), default="base")
    g.add_argument("--workers", type=int, default=3)
    g.add_argument("--out", required=True)
    p = sub.add_parser("pack")
    p.add_argument("--src", nargs="+", required=True, help="dir[:fraction of games]")
    p.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    gen(a) if a.cmd == "gen" else pack(a)


if __name__ == "__main__":
    main()
