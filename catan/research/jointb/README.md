# Research branch: D18 jointb trade head + D21 opponent classifier (the champion)

`jointb_gated` is the shipped champion. It is `value_trade-v1` (D12 value-net search plus the nego trade
model) with two learned parts from this branch:

- **D18, the jointb trade head** (`catan/research/jointb/`). Counterfactual labels: each trade decision's top
  8 options are replayed on clones with common random numbers to the learner's next turn, and the D12 value
  net scores each branch. The head is distilled from those labels. Weights: `models/jointb_best.pt` (run r5).
- **D21, the opponent classifier** (`catan/research/opp/`). A small MLP over public-log features that labels
  each opponent seat. `jointb_gated` uses the jointb head only against seats it does not classify as basic.
  Weights: `models/opp_best.pt`.

Full write-ups: [docs/research/JOINTB.md](../../../docs/research/JOINTB.md), [docs/research/OPP.md](../../../docs/research/OPP.md).

## Result

Promotion gate, 1 candidate seat vs 3 opponents (chance 25%), seed blocks 40-42, deterministic play:

| test | jointb_gated-v1 | 95% CI | n | value_trade-v1 (previous champion) |
|---|---|---|---|---|
| vs 3x basic (block 40) | 70.5% | 63.8-76.4% | 200 | 71.0% (0 vs 1 discordant games) |
| vs 3x intermediate (block 41) | 76.5% | 70.2-81.8% | 200 | 66.5% (46 vs 26 discordant, McNemar p = 0.024) |
| mixed table with value_trade-v1 + 2x intermediate, all 12 seatings (block 42) | 47.7% (286 wins vs 211) | 43.7-51.7% | 600 | sign test p = 0.0009 |

Original gate run: commit `c7e7bb7f2fe6de01941d25f2af4f21256719015e`. Head (r5) code: `cb1a4202`.
Classifier code: `2b2c8a41`.

## Rerun

```bash
uv sync
uv run python -m catan.research.jointb.reproduce                 # the full gate, compared game by game
uv run python -m catan.research.jointb.reproduce --games 8       # first 8 games of each test (minutes)
uv run python -m catan.research.jointb.reproduce --stage train   # regenerate labels, head and classifier
```

`expected.json` holds each gate test's lineup, seating, seed block and the recorded per-game result. The gate
stage exits 1 if a single game differs.

- **Gate: exactly reproducible.** The engine, the bots and the search are seeded, and play is deterministic.
  The first 8 games of the basic and intermediate tests and the first 6 table games were rerun on this
  branch: every game had the same winner, VP and length as the recorded run.
- **Training: same seeds, not bit-exact.** Labels use training seeds TRAIN_SEED_BASE + 18e6 + i, the
  classifier's games TRAIN_SEED_BASE + 21e6 + i (train) and held-out block 55 (test). The recorded label
  job was stopped by hand at 2,040 of 4,500 games, so the rerun takes games 0-2,039, which may not be the
  exact set the stopped job had finished. Training runs on CPU torch with seeded initialisation; thread
  scheduling can still change the last bits. Expect the head's held-out regret near 0.0183 and the
  classifier's turn-16 accuracy near 75.8%, not identical weights.

Labels take about 75 minutes on 3 workers; the full gate takes a few hours on 2 workers.
