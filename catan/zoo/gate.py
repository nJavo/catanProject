"""Promotion gate.

Seed blocks (registered in ``catan.eval.seeds``: 1_000_000 + block*100_000 + i)
reserved for the gate and used nowhere else: 40 (vs 3x basic), 41 (vs 3x
intermediate), 42 (champion tables). 1v3 seats are rotated cyclically; champion tables use all
12 seatings.
``selection_guard`` flags candidates whose metadata says they were selected on
these seeds; scripts/promote.py refuses them unless --allow-gate-selected.

Rules (alpha = 0.05):
* champion test, learned champion C (primary): same-seat paired.
  The candidate table [candidate, C, int, int] and a cached copy table [C', C, int, int] (C' = C
  under a distinct label, ``tournament.copy_spec``) are played on the same block-42 seeds in all 12
  seatings. Games are paired on (seed, rotation), so the focal seat sees the same board and seating
  in both. The candidate must win more of the discordant pairs and the exact McNemar test must
  reject 50/50. The copy table is cached per champion id and reused for every candidate.
  Reported but not decisive (``secondary``): the older within-table sign test, candidate seat vs
  champion seat in the candidate table. Within one table the two seats differ in board position
  and dice, which gave an identical copy 231 vs 265 (D24); the same-seat pair has none of that.
* champion test, heuristic champion H (no RL promoted yet): candidate vs 3x H,
  Wilson 95% lower bound of the win rate must exceed 1/N (an equal player's rate).
* heuristics: for basic and intermediate, candidate vs 3x H is paired game by
  game with the champion vs 3x H on the same seeds and seats. Fail if the
  champion wins significantly more (exact McNemar on discordant games). When
  the champion is H itself its rate is 1/N; fail if the Wilson upper bound < 1/N.
Training reward is never an input.
"""
from __future__ import annotations

import json
from pathlib import Path

from catan.eval.stats import binom_two_sided_p, wilson
from catan.eval.seeds import GATE_BLOCKS, eval_range, gate_conflicts
from catan.eval.tournament import copy_spec, run_tournament

BLOCKS = dict(GATE_BLOCKS)  # {"basic": 40, "intermediate": 41, "champion": 42}
HEURISTICS = ("basic", "intermediate")
ALPHA = 0.05


def selection_metadata(entry: dict, root=".", extra: dict | None = None) -> dict:
    """Everything known about how ``entry`` was selected: its ``selection``
    record, its run's meta.json and in-training eval seeds (config.yaml), plus
    ``extra`` (e.g. a JSON file given to scripts/promote.py)."""
    meta = {"selection": entry.get("selection")} if entry.get("selection") else {}
    run = entry.get("source_run")
    if run:
        d = Path(root) / "runs" / run
        if (d / "meta.json").is_file():
            meta["run_meta"] = json.loads((d / "meta.json").read_text())
        if (d / "config.yaml").is_file():
            import yaml
            rng = eval_range(yaml.safe_load((d / "config.yaml").read_text()))
            if rng:
                meta["in_training_eval_seeds"] = rng
    if extra:
        meta["extra"] = extra
    return meta


def selection_guard(entry: dict, root=".", extra: dict | None = None) -> list[str]:
    """Reasons the candidate looks selected on gate seeds (empty = clean).
    A gate result on seeds the candidate was picked on would be biased upward."""
    return gate_conflicts(selection_metadata(entry, root, extra))


def run_provenance(run_dir) -> list[str]:
    """Why results from this run would not be reproducible (empty = clean).
    The training code must be a recorded, clean commit (RL-020)."""
    d = Path(run_dir)
    m = d / "meta.json"
    if not m.is_file():
        return [f"{d.name}: no meta.json, so the training commit is unknown"]
    try:
        meta = json.loads(m.read_text())
    except ValueError:
        return [f"{d.name}: meta.json is not valid JSON"]
    out = []
    commit = meta.get("git_commit")
    if not commit or commit == "unknown":
        out.append(f"{d.name}: meta.json records no git_commit")
    if meta.get("git_dirty") is True:
        out.append(f"{d.name}: trained from a dirty git tree (git_dirty: true at {commit})")
    elif meta.get("git_dirty") is not False:
        out.append(f"{d.name}: meta.json has no git_dirty flag, so the tree state is unknown")
    return out


def tree_provenance(root=".") -> list[str]:
    """Problems with the code that evaluates now: uncommitted changes to tracked files."""
    import subprocess
    try:
        r = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=root,
                           capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError) as e:
        return [f"cannot check the working tree: {e}"]
    if r.returncode != 0:
        return [f"cannot check the working tree: {r.stderr.strip() or 'not a git repository'}"]
    # Evaluation outputs (registry, leaderboard, tournament results) are written by promote
    # itself; they are data, not the code under test.
    data = ("models/registry.json", "results/")
    changed = [ln[3:] for ln in r.stdout.splitlines() if ln.strip() and not ln[3:].startswith(data)]
    return [f"evaluating from uncommitted changes: {', '.join(changed[:5])}{' ...' if len(changed) > 5 else ''}"] \
        if changed else []


def provenance_guard(run_dirs, root=".") -> list[str]:
    """RL-020: reasons not to promote or report (empty = clean)."""
    out = []
    for d in run_dirs:
        out += run_provenance(d)
    return out + tree_provenance(root)


def _rate(k, n):
    return {"wins": k, "games": n, "win_rate": k / n if n else 0.0, "ci95": list(wilson(k, n))}


def heuristic_check(cand: list[int], champ: list[int] | None, null: float, alpha=ALPHA) -> dict:
    """cand/champ: per-game win indicators on the same (seed, seat) games."""
    out = {"candidate": _rate(sum(cand), len(cand))}
    if champ is None:
        out["champion"] = {"win_rate": null, "note": "champion is this heuristic: rate 1/N"}
        out["worse"] = out["candidate"]["ci95"][1] < null
        return out
    assert len(champ) == len(cand), "paired vectors must align"
    b = sum(1 for x, y in zip(cand, champ) if x and not y)
    c = sum(1 for x, y in zip(cand, champ) if y and not x)
    p = binom_two_sided_p(b, b + c) if b + c else 1.0
    out.update(champion=_rate(sum(champ), len(champ)), discordant={"candidate_only": b, "champion_only": c},
               mcnemar_p=p, worse=c > b and p < alpha)
    return out


def same_seat_check(cand: list[int], copy: list[int], alpha=ALPHA) -> dict:
    """Primary champion test: per-pair focal win indicators of the candidate and of the same-seat
    champion copy on identical (seed, rotation) games. Better = more candidate-only wins and an
    exact McNemar p < alpha."""
    assert len(cand) == len(copy), "paired vectors must align"
    n = len(cand)
    b = sum(1 for x, y in zip(cand, copy) if x and not y)
    c = sum(1 for x, y in zip(cand, copy) if y and not x)
    p = binom_two_sided_p(b, b + c) if b + c else 1.0
    d = [x - y for x, y in zip(cand, copy)]
    m = sum(d) / n if n else 0.0
    sd = (sum((x - m) ** 2 for x in d) / (n - 1)) ** 0.5 if n > 1 else 0.0
    h = 1.96 * sd / n ** 0.5 if n > 1 else 0.0
    return {"candidate": _rate(sum(cand), n), "copy": _rate(sum(copy), n), "pairs": n,
            "discordant": {"candidate_only": b, "copy_only": c}, "mcnemar_p": p,
            "diff": m, "diff_ci95": [m - h, m + h], "better": b > c and p < alpha}


def champion_check(t: dict, alpha=ALPHA) -> dict:
    if t["kind"] == "same_seat":
        r = same_seat_check(t["cand"], t["copy"], alpha)
        rest = {k: v for k, v in t.items() if k not in ("cand", "copy", "within")}
        return {**rest, **r, "games": r["pairs"], "secondary": champion_check({**t["within"], "kind": "mixed"}, alpha)}
    if t["kind"] == "mixed":
        k1, k2 = t["cand_wins"], t["champ_wins"]
        p = binom_two_sided_p(k1, k1 + k2) if k1 + k2 else 1.0
        return {**t, "candidate": _rate(k1, t["games"]), "champion": _rate(k2, t["games"]),
                "sign_p": p, "better": k1 > k2 and p < alpha}
    r = _rate(t["wins"], t["games"])
    return {**t, "candidate": r, "better": r["ci95"][0] > t["null"]}


def champion_test_lines(ct: dict) -> list[str]:
    """Human-readable champion test (scripts/promote.py and the other gate CLIs)."""
    pct = lambda r: f"{r['win_rate']:.1%} [{r['ci95'][0]:.1%},{r['ci95'][1]:.1%}]"  # noqa: E731
    if ct["kind"] == "same_seat":
        d, s = ct["discordant"], ct["secondary"]
        return [f"  champion test (primary, same-seat paired vs a {ct['lineup'][1]} copy, n={ct['pairs']} pairs): "
                f"candidate {pct(ct['candidate'])} vs copy {pct(ct['copy'])}; diff {ct['diff']:+.1%} "
                f"[{ct['diff_ci95'][0]:+.1%},{ct['diff_ci95'][1]:+.1%}]; discordant {d['candidate_only']}-{d['copy_only']}, "
                f"McNemar p={ct['mcnemar_p']:.3g}; better={ct['better']}",
                f"  within-table (secondary, not decisive): candidate {s['cand_wins']} vs champion seat "
                f"{s['champ_wins']} of {s['games']}, sign p={s['sign_p']:.3g}"]
    if ct["kind"] == "mixed":
        return [f"  champion test (mixed, within table): candidate {ct['cand_wins']} vs champion {ct['champ_wins']} "
                f"of {ct['games']}, sign p={ct['sign_p']:.3g}; better={ct['better']}"]
    return [f"  champion test ({ct['kind']}): candidate {pct(ct['candidate'])} n={ct['candidate']['games']} "
            f"vs chance {ct['null']:.0%}; better={ct['better']}"]


def decide(heuristics: dict, champ: dict, alpha=ALPHA) -> dict:
    """heuristics: {H: heuristic_check(...)}; champ: champion_check(...)."""
    reasons = []
    if not champ["better"]:
        reasons.append("not significantly better than the champion")
    reasons += [f"significantly worse than the champion vs 3x{h}" for h, r in heuristics.items() if r["worse"]]
    return {"promote": not reasons, "reasons": reasons, "alpha": alpha,
            "champion_test": champ, "heuristic_tests": heuristics}


def _wins(res: dict, spec: str) -> dict:
    return {(g["seed"], g["rotation"]): int(g["winner"] >= 0 and g["lineup"][g["winner"]] == spec)
            for g in res["game_summaries"]}


def run_gate(cand: dict, champ: dict, games: int = 200, workers: int = 3, out_dir="results",
             num_players: int = 4, table_games: int | None = None) -> dict:
    """Play the gate tournaments (saved under results/tournaments) and decide."""
    import torch
    torch.set_num_threads(1)  # forked workers inherit this; torch defaults to one thread per core
    cs, hs = cand["spec"], champ["spec"]
    # Mixed champion table: an identical copy of combo_zs beat the champion seat 82-58 at n=200
    # (D13), so the champion test needs more games than the heuristic tests.
    table_games = table_games or games
    def run(agents, block, tag, tid=None, n=None, rotation="cyclic"):
        n = n or games
        tid = tid or f"gate_{cand['id']}_{tag}"
        cached = Path(out_dir or "") / "tournaments" / f"{tid}.json"
        if tid.startswith("gate_champ_") and cached.is_file():
            # champion baselines are reused across candidates: same seeds, and the
            # leaderboard would otherwise count the identical games twice
            t = json.loads(cached.read_text())
            if t.get("agents") == list(dict.fromkeys(agents)) and t["games"] >= n:
                return t
        return run_tournament(agents, n, seed=BLOCKS[block], workers=workers, num_players=num_players,
                              out_dir=out_dir, tid=tid, progress=True, rotation=rotation)
    # 1 candidate seat vs (n-1) copies: passing [x, h] would fill seats cyclically
    # as x, h, x, h (a 2v2 team game) while the null below is 1/n.
    def one_vs(x, h):
        return [x] + [h] * (num_players - 1)
    heur = {}
    for h in HEURISTICS:
        rc = _wins(run(one_vs(cs, h), h, f"vs3{h}"), cs)
        keys = sorted(rc)
        if hs == h:
            heur[h] = heuristic_check([rc[k] for k in keys], None, 1 / num_players)
        else:
            rh = _wins(run(one_vs(hs, h), h, None, f"gate_champ_{champ['id']}_1v3{h}_n{games}"), hs)
            keys = [k for k in keys if k in rh]  # failed games drop out of both sides
            heur[h] = heuristic_check([rc[k] for k in keys], [rh[k] for k in keys], 1 / num_players)
    if not champ.get("anchor"):  # learned champion (rl, arch_rl, nego, ...)
        # rotation="all" (12 seatings): with cyclic rotation the candidate always sits right before the
        # champion, and an identical copy beat the champion seat 210-167 at n=600 (D15, p=0.03).
        res = run([cs, hs, "intermediate", "intermediate"], "champion", f"table_{champ['id']}", n=table_games,
                  rotation="all")
        # same-seat control: the champion under a distinct label in the candidate's seat, cached per champion
        cp = copy_spec(hs)
        tid = f"gate_champ_{champ['id']}_table_copy_n{table_games}"
        ctl = run([cp, hs, "intermediate", "intermediate"], "champion", None, tid, n=table_games, rotation="all")
        rc, rk = _wins(res, cs), _wins(ctl, cp)
        keys = sorted(k for k in rc if k in rk)  # failed games drop out of both sides
        lineup = [cand["id"], champ["id"], "intermediate", "intermediate"]
        ct = {"kind": "same_seat", "lineup": lineup,
              "control_lineup": [f"{champ['id']} (copy)", champ["id"], "intermediate", "intermediate"],
              "control_table": tid, "rotation": "all",
              "cand": [rc[k] for k in keys], "copy": [rk[k] for k in keys],
              "within": {"lineup": lineup, "cand_wins": res["per_agent"][cs]["wins"],
                         "champ_wins": res["per_agent"][hs]["wins"], "games": res["games"]}}
    elif hs in heur:  # already played candidate vs 3x champion
        c = heur[hs]["candidate"]
        ct = {"kind": "anchor", "lineup": [cand["id"]] + [hs] * (num_players - 1),
              "wins": c["wins"], "games": c["games"], "null": 1 / num_players}
    else:
        res = run(one_vs(cs, hs), "champion", f"vs3{hs}")
        ct = {"kind": "anchor", "lineup": [cand["id"]] + [hs] * (num_players - 1),
              "wins": res["per_agent"][cs]["wins"], "games": res["games"], "null": 1 / num_players}
    out = decide(heur, champion_check(ct))
    out.update(champion=champ["id"], seed_blocks=BLOCKS, games_per_test=games)
    return out
