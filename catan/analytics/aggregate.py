"""Aggregate per-seat behavior rows (from ``analyze_game``) into per-agent summaries."""
from __future__ import annotations

from collections import Counter

from catan.engine.board import RESOURCES
from catan.eval.stats import mean_ci, wilson

# name -> (numerator key, denominator key): pooled over all seat-games, Wilson CI
RATIOS = {
    "robber_hits_leader_rate": ("robber_hits_leader", "robber_moves"),
    "robber_self_block_rate": ("robber_self_block", "robber_moves"),
    "robber_no_opponent_rate": ("robber_no_opponent", "robber_moves"),
    "robber_by_knight_rate": ("robber_by_knight", "robber_moves"),
    "knight_preroll_rate": ("knight_preroll", "knight_plays"),
    "offer_success_rate": ("offers_executed", "offers"),
    "trade_accept_rate_as_responder": ("trade_accepted", "trade_asked"),
    "missed_vp_build_rate": ("missed_vp_build", "end_turns"),
    "missed_city_rate": ("missed_city", "end_turns"),
    "missed_settlement_rate": ("missed_settlement", "end_turns"),
    "missed_dev_rate": ("missed_dev", "end_turns"),
    "missed_any_rate": ("missed_any", "end_turns"),
    "end_turn_has_spot_rate": ("end_turn_has_spot", "end_turns"),
    "end_turn_afford_settlement_rate": ("end_turn_afford_settlement", "end_turns"),
    "end_turn_afford_city_rate": ("end_turn_afford_city", "end_turns"),
    "discard_per_seven_rate": ("sevens_discarding", "sevens_rolled"),
}
SKIP = {"seat", "actions", "dev_played", "open_missing"}


def _num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def aggregate_rows(rows: list[dict]) -> dict:
    """Means with 95% CIs for every numeric metric, pooled ratios, missing-resource
    frequencies, dev plays per game and action mix per phase."""
    if not rows:
        return {"seat_games": 0}
    for r in rows:
        r.setdefault("knight_plays", r.get("knight_preroll", 0) + r.get("knight_postroll", 0))
    out = {"seat_games": len(rows), "metrics": {}, "ratios": {}}
    keys = [k for k in rows[0] if k not in SKIP]
    for k in keys:
        vals = [r[k] for r in rows if _num(r.get(k))]
        if vals:
            m = mean_ci(vals)
            out["metrics"][k] = {"mean": round(m["mean"], 4), "ci95": [round(x, 4) for x in m["ci95"]],
                                 "n": m["n"]}
    for name, (a, b) in RATIOS.items():
        num, den = sum(r.get(a, 0) for r in rows), sum(r.get(b, 0) for r in rows)
        lo, hi = wilson(num, den)
        out["ratios"][name] = {"rate": round(num / den, 4) if den else None, "num": num, "den": den,
                               "ci95": [round(lo, 4), round(hi, 4)]}
    miss = Counter(m for r in rows for m in r.get("open_missing", []))
    out["open_missing_freq"] = {res: round(miss[res] / len(rows), 4) for res in RESOURCES}
    dev = Counter()
    for r in rows:
        dev.update(r.get("dev_played", {}))
    out["dev_played_per_game"] = {c: round(v / len(rows), 4) for c, v in sorted(dev.items())}
    acts = Counter()
    for r in rows:
        acts.update(r.get("actions", {}))
    by_phase: dict = {}
    for key, c in acts.items():
        ph, t = key.split(":", 1)
        by_phase.setdefault(ph, {})[t] = c
    out["actions_per_game"] = {ph: {t: round(c / len(rows), 3) for t, c in sorted(d.items(), key=lambda x: -x[1])}
                               for ph, d in sorted(by_phase.items())}
    out["action_share_by_phase"] = {ph: {t: round(c / sum(d.values()), 4) for t, c in d.items()}
                                    for ph, d in by_phase.items()}
    return out
