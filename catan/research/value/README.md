# Research branch: D12 value net + value_trade

`value_trade` is `exit_search` whose leaf is a learned value net (D12), with the nego trade model for
trades. It was the champion before `jointb_gated`, and it is still the core of the shipped champion.

- **Value net** (`catan/research/value/`). The D5 graph network (`models/arch_best.pt`) with only its value
  head trained, by binary cross-entropy on the final result, over positions from mixed-table bot games.
  Weights: `models/value_best.pt`.
- **Agent.** `value_trade:checkpoint=models/value_best.pt` runs 32 search simulations per move with the
  league GNN (`models/league_best.pt`) as the prior.

Full write-up: [docs/research/VALUE.md](../../../docs/research/VALUE.md).

## Result

Promotion gate against the previous champion combo_zs-v1, 1 candidate seat vs 3 opponents (chance 25%),
seed blocks 40-42, deterministic play:

| test | value_trade-v1 | 95% CI | n | combo_zs-v1 |
|---|---|---|---|---|
| vs 3x basic (block 40) | 71.0% | 64.4-76.8% | 200 | 46.5% (66 vs 17 discordant games) |
| vs 3x intermediate (block 41) | 66.5% | 59.7-72.7% | 200 | 48.0% (56 vs 19 discordant games) |
| mixed table with combo_zs-v1 + 2x intermediate (block 42, cyclic seating) | 102 wins vs 40 | | 200 | sign test p = 2e-7 |

Value net calibration on 60k held-out positions: Brier 0.158, log-loss 0.479 (a constant predictor: 0.199,
0.588). Original gate run: commit `32ed65e622ea069c6a78aad9de9a875cd85240ed`. Data and training code:
`08596f0c`.

## Rerun

```bash
uv sync
uv run python -m catan.research.value.reproduce                 # the full gate, compared game by game
uv run python -m catan.research.value.reproduce --games 8       # first 8 games of each test
uv run python -m catan.research.value.reproduce --stage train   # data generation and training
```

`expected.json` holds each gate test's lineup, seating, seed block and the recorded per-game result. The gate
stage exits 1 if a single game differs.

- **Gate: exactly reproducible.** Seeded engine, deterministic agents. The first 8 games vs basic, 4 vs
  intermediate and 4 table games were rerun on this branch: same winner, VP and length as the recorded run.
- **Data generation: reproducible.** 6,000 games on seeds 70,000,000 + i, 610k positions, written in shards
  of 40 seeds. The first shard was regenerated on this branch: 4,156 rows, the recorded count.
- **Training: not reproducible from this branch alone.** The recorded net was also trained on 2.0M
  positions from an older experiment (D2: behaviour-cloning and DAgger shards), whose code and
  data are not part of this branch. The recorded run was also stopped by hand at 0.5 epochs. The train stage
  therefore trains on this branch's data only, for one epoch, and keeps the best checkpoint by validation
  log-loss. Expect a weaker net, not the shipped one.

The data stage takes about 35 minutes on 3 workers; the full gate takes a few hours on 2 workers.
