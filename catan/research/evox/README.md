# Research branch: D29 evox, island-model evolution of a heuristic evaluator

`evox` is the `intermediate` bot with every weight of its evaluator (27 genes: production, diversity,
opening, robber, dev cards, trades, ...) evolved from random by an island-model genetic algorithm:
120 islands × 4 members, same-opponent tables (2 members + 2 copies of basic, intermediate or a
hall-of-fame genome), common random numbers per table. No search, no network, no trade model.

Code: `catan/research/evox/`. Weights: `models/evox_best.json` (and `.pt`, the same dict for the model
registry): genome `e164972` from generation 1249 of run `evox_r2`. Full write-up:
[docs/research/EVOX.md](../../../docs/research/EVOX.md).

## Result

Paired final test, 1 vs 3 (chance 25%), 600 games per table on report blocks 60 (offsets 1000-1074)
and 61 (0-74), 4 seats:

| genome | vs 3x basic | vs 3x intermediate |
|---|---|---|
| **evolved (e164972)** | **43.2% [39.3, 47.2]** | **33.3% [29.7, 37.2]** |
| generation-0 ancestor (e000196) | 1.3% [0.7, 2.6] | 2.2% [1.3, 3.7] |
| hand-tuned `intermediate` weights | 37.0% [33.2, 40.9] | 25.0% [21.7, 28.6] |

Evolved minus hand-tuned: +6.2 pp vs 3x basic (134 vs 97 discordant games, p = 0.018) and +8.3 pp vs
3x intermediate (129 vs 79, p = 0.0006). It is the first evolution result in this project that beats the
hand-tuned baseline on held-out tables.

Gate (information only, not promoted; blocks 40-42, n = 200 / 200 / 600): 42.5% [35.9, 49.4] vs 3x basic,
36.5% [30.1, 43.4] vs 3x intermediate, 11.2% [8.9, 13.9] in the mixed table against the champion
`jointb_gated`.

Original run: `evox_r2` at commit `287eea6`; gate at `4713349c46c7ff8f30979434494c95e4a8737e69`.

## Rerun

```bash
uv sync
uv run python -m catan.research.evox.reproduce                       # the gate, compared game by game
uv run python -m catan.research.evox.reproduce --games 8             # first 8 games of each gate test
uv run python -m catan.research.evox.reproduce --stage evolve        # r2 again (seed 2), then pick and test
uv run python -m catan.research.evox.reproduce --stage evolve --generations 3   # a quick check
```

`expected.json` holds each gate test's lineup, seating, seed block and the recorded per-game result.

- **Gate: exactly reproducible.** The first 8 games vs basic, 8 vs intermediate and 4 table games were
  rerun on this branch: same winner, VP and length as the recorded run.
- **Evolution: exactly reproducible, as far as checked.** Every random draw comes from the run seed (2) and
  the per-table seeds of selection block 56, and the worker pool only changes where games run. Generations
  0-2 rerun with 1 worker instead of the recorded 3 gave the recorded fitness, prior, sigma and champion
  (e000095, Elo 1601.6, 15.5% vs 3x basic) to the last digit. The recorded run was stopped by hand after
  generation 1339; the final pick only looks at the last 10 checkpoints up to generation 1299, so
  `--generations 1300` covers it. The full rerun takes about 2.5 hours on 3 workers and was not repeated here.
