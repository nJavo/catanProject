"""Batched rollout collection over many in-process games, plus a worker
process wrapper for multi-core collection.

Each tick, every game has exactly one pending external decision. Decisions are
grouped by controller (current policy, or a frozen pool snapshot) and run
through the network in one batch. Only current-policy ("learner") decisions
are stored. Each (game, learner seat) forms its own trajectory; GAE runs over
the linked list of that seat's decisions.
"""
from __future__ import annotations

import random

import numpy as np
import torch

from catan.agents.registry import make_agent
from catan.eval.seeds import TRAIN_SEED_BASE  # 10_000_000; games use TRAIN_SEED_BASE + randrange(2**30)
from .env import CatanEnv, EnvConfig
from .model import PolicyNet, model_kwargs

LEARNER = "L"


class Collector:
    def __init__(self, cfg: dict, n_envs: int, seed: int):
        self.cfg = cfg
        self.env_cfg = EnvConfig.from_dict(cfg.get("env"))
        self.gamma = float(cfg["ppo"]["gamma"])
        self.lam = float(cfg["ppo"]["gae_lambda"])
        self.rng = random.Random(seed)
        self.gen = torch.Generator().manual_seed(seed)
        self.model = PolicyNet(**model_kwargs(cfg))
        self.model.eval()
        self.pool: dict[int, PolicyNet] = {}
        self.active: set[int] = set()   # pool ids new games may sample; retired ones live until unused
        self.opponents = cfg.get("opponents") or [{"type": "agent", "name": "random", "weight": 1.0}]
        self.learner_seats = int(cfg.get("learner_seats", 1))
        self.envs = [CatanEnv(self.env_cfg) for _ in range(n_envs)]
        self.ctrl: list[list] = [None] * n_envs      # per env: per seat LEARNER | ("F", pid) | None (agent)
        self.mode: list[str] = [""] * n_envs
        self.pending: list = [None] * n_envs          # current Step per env
        for i in range(n_envs):
            self._new_game(i)

    # ---------------------------------------------------------- weights
    def set_weights(self, sd):
        self.model.load_state_dict(sd)

    def set_pool(self, pool: dict):
        """pool: pid -> state_dict (None removes)."""
        for pid, sd in pool.items():
            if sd is None:
                self.active.discard(pid)
                continue
            m = self.pool.get(pid) or PolicyNet(**model_kwargs(self.cfg))
            m.load_state_dict(sd)
            m.eval()
            self.pool[pid] = m
            self.active.add(pid)
        self._prune()

    def _prune(self):
        used = {c[1] for ctrl in self.ctrl if ctrl for c in ctrl if isinstance(c, tuple)}
        for pid in list(self.pool):
            if pid not in self.active and pid not in used:
                del self.pool[pid]

    # ---------------------------------------------------------- games
    def _sample_mode(self):
        specs = [o for o in self.opponents if not (o["type"] == "pool" and not self.active)]
        specs = specs or [{"type": "self"}]  # pool-only config before the first snapshot
        w = [float(o.get("weight", 1.0)) for o in specs]
        return self.rng.choices(specs, weights=w)[0]

    def _new_game(self, i):
        n = self.env_cfg.num_players
        spec = self._sample_mode()
        seats = list(range(n))
        self.rng.shuffle(seats)
        learners = set(seats if spec["type"] == "self" else seats[:self.learner_seats])
        ctrl, agents = [], []
        if spec["type"] == "pool":
            pid = self.rng.choice(sorted(self.active))
            name = f"pool:{pid}"
        else:
            name = spec["type"] if spec["type"] == "self" else "+".join(spec["name"]) if isinstance(spec["name"], list) else spec["name"]
        for s in range(n):
            if s in learners:
                ctrl.append(LEARNER)
                agents.append(None)
            elif spec["type"] == "pool":
                ctrl.append(("F", pid))
                agents.append(None)
            else:
                ctrl.append(None)
                # name may be a list: a mixed table, one name per opponent seat in turn order
                names = spec["name"] if isinstance(spec["name"], list) else [spec["name"]]
                k = sum(1 for x in range(s) if x not in learners)
                agents.append(make_agent(names[k % len(names)], **(spec.get("kwargs") or {})))
        self.ctrl[i], self.mode[i] = ctrl, name
        seed = TRAIN_SEED_BASE + self.rng.randrange(2**30)
        self.pending[i] = self.envs[i].reset(seed, agents)

    # ---------------------------------------------------------- collection
    @torch.no_grad()
    def _act(self, model, idx, deterministic=False):
        obs = torch.from_numpy(np.stack([self.pending[i].obs for i in idx]))
        mask = torch.from_numpy(np.stack([self.pending[i].mask for i in idx]))
        logits, v = model(obs, mask)
        if deterministic:
            a = logits.argmax(-1)
        else:
            a = torch.multinomial(torch.softmax(logits, -1), 1, generator=self.gen).squeeze(-1)
        logp = torch.log_softmax(logits, -1).gather(1, a[:, None]).squeeze(1)
        return a.numpy(), logp.numpy(), v.numpy()

    def collect(self, n_steps: int) -> dict:
        masks, acts, logps, vals = [], [], [], []
        # Preallocated (no list + np.stack peak). A tick adds at most one row per env.
        # cfg["obs_dtype"] = "float16" halves rollout memory for very large batches.
        obs = None
        rew, nxt, term, boot = [], [], [], []
        open_: list[dict] = [dict() for _ in self.envs]   # env -> seat -> transition idx
        episodes = []
        while len(acts) < n_steps:
            groups: dict = {}
            for i, st in enumerate(self.pending):
                groups.setdefault(self.ctrl[i][st.seat], []).append(i)
            for c, idx in groups.items():
                model = self.model if c == LEARNER else self.pool[c[1]]
                a, lp, v = self._act(model, idx)
                for k, i in enumerate(idx):
                    st = self.pending[i]
                    if c == LEARNER:
                        t = len(acts)
                        if obs is None:
                            obs = np.empty((n_steps + len(self.envs), len(st.obs)),
                                           dtype=self.cfg.get("obs_dtype", "float32"))
                        obs[t] = st.obs
                        masks.append(st.mask); acts.append(int(a[k]))
                        logps.append(float(lp[k])); vals.append(float(v[k]))
                        rew.append(0.0); nxt.append(-1); term.append(False); boot.append(0.0)
                        prev = open_[i].get(st.seat)
                        if prev is not None:
                            nxt[prev] = t
                        open_[i][st.seat] = t
                    res = self.envs[i].step(int(a[k]))
                    if not res.done:
                        j = open_[i].get(res.seat)
                        if j is not None:
                            rew[j] += res.reward
                        self.pending[i] = res
                        continue
                    for s, r in res.final.items():
                        j = open_[i].pop(s, None)
                        if j is None:
                            continue
                        rew[j] += r
                        if res.bootstrap:
                            boot[j] = self._value(i, s)
                        else:
                            term[j] = True
                    episodes.append(self._episode_stats(i))
                    self._new_game(i)
        # rollout boundary: bootstrap open trajectories from V(current state)
        for i, d in enumerate(open_):
            for s, j in d.items():
                rew[j] += self.envs[i].pending_reward(s)
                boot[j] = self._value(i, s)
        adv = np.zeros(len(acts), dtype=np.float32)
        vals_a = np.asarray(vals, dtype=np.float32)
        g, lam = self.gamma, self.lam
        for t in range(len(acts) - 1, -1, -1):
            n = nxt[t]
            if n >= 0:
                delta = rew[t] + g * vals_a[n] - vals_a[t]
                adv[t] = delta + g * lam * adv[n]
            else:
                nv = 0.0 if term[t] else boot[t]
                adv[t] = rew[t] + g * nv - vals_a[t]
        return {
            "obs": obs[:len(acts)], "mask": np.stack(masks),
            "act": np.asarray(acts, dtype=np.int64), "logp": np.asarray(logps, dtype=np.float32),
            "val": vals_a, "adv": adv, "ret": adv + vals_a,
            "rew": np.asarray(rew, dtype=np.float32),
            "episodes": episodes,
        }

    @torch.no_grad()
    def _value(self, i, seat):
        o = torch.from_numpy(self.envs[i].observe(seat))[None]
        return float(self.model(o)[1][0])

    def _episode_stats(self, i):
        g = self.envs[i].game
        ls = [s for s, c in enumerate(self.ctrl[i]) if c == LEARNER]
        return {"mode": self.mode[i], "learner_seats": ls, "winner": g.winner,
                "learner_won": g.winner in ls, "seat_win_rate": (g.winner in ls) / len(ls), "truncated": g.truncated, "turns": g.turn,
                "learner_vp": float(np.mean([g.victory_points(s) for s in ls]))}


# -------------------------------------------------------------- workers
def _worker(conn, cfg, n_envs, seed):
    torch.set_num_threads(1)
    col = Collector(cfg, n_envs, seed)
    try:
        while True:
            cmd, arg = conn.recv()
            if cmd == "weights":
                col.set_weights(arg)
            elif cmd == "pool":
                col.set_pool(arg)
            elif cmd == "collect":
                conn.send(col.collect(arg))
            elif cmd == "close":
                return
    except (EOFError, BrokenPipeError, ConnectionResetError):
        return  # parent died: exit instead of lingering as an orphan
    finally:
        conn.close()


class WorkerPool:
    """``num_workers`` processes, each with its own Collector. num_workers=0
    collects in-process (deterministic, used by tests)."""

    def __init__(self, cfg: dict, num_workers: int, envs_per_worker: int, seed: int):
        self.n = num_workers
        if num_workers == 0:
            self.local = Collector(cfg, envs_per_worker, seed)
            return
        import multiprocessing as mp
        ctx = mp.get_context("spawn")
        self.conns, self.procs = [], []
        for w in range(num_workers):
            a, b = ctx.Pipe()
            p = ctx.Process(target=_worker, args=(b, cfg, envs_per_worker, seed * 1000 + w), daemon=True)
            p.start()
            self.conns.append(a)
            self.procs.append(p)

    def _bcast(self, cmd, arg):
        for c in self.conns:
            c.send((cmd, arg))

    def set_weights(self, sd):
        sd = {k: v.detach().cpu() for k, v in sd.items()}
        if self.n == 0:
            self.local.set_weights(sd)
        else:
            self._bcast("weights", sd)

    def set_pool(self, pool):
        if self.n == 0:
            self.local.set_pool(pool)
        else:
            self._bcast("pool", pool)

    def collect(self, n_steps: int) -> dict:
        if self.n == 0:
            return self.local.collect(n_steps)
        per = -(-n_steps // self.n)
        self._bcast("collect", per)
        parts = [c.recv() for c in self.conns]
        out = {k: np.concatenate([p[k] for p in parts]) for k in parts[0] if k != "episodes"}
        out["episodes"] = [e for p in parts for e in p["episodes"]]
        return out

    def close(self):
        if not self.n:
            return
        for c in self.conns:
            try:
                c.send(("close", None))
            except (BrokenPipeError, OSError):
                pass
        for p in self.procs:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()
                p.join(timeout=5)
