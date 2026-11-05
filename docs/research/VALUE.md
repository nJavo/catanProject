# D12: Learned evaluator (value network)

Prefix `value`, selection block 53, package `catan/research/value/`. This follows from D4
(docs/research/EXIT.md): search strength is capped by the leaf evaluator, which is
`intermediate.evaluate`, and the policy networks' value heads were useless as leaves.

## Background

- **AlphaGo** (Silver et al. 2016, [Nature](https://www.nature.com/articles/nature16961)). Its
  value net was trained on one position per self-play game, because consecutive positions of a
  game are strongly correlated and sharing a label across them overfits. That is the same leak
  D4 hit with a per-decision split. Here, validation holds out whole games, and the 10–15%
  decision subsampling of the data sources limits within-game correlation.
- **AlphaZero / MuZero.** The value target is the outcome z, regressed with MSE. For a 0/1
  winner in a 4-player game, BCE on a logit gives a calibrated win probability directly.
- **Calibration.** Guo et al. 2017, "On Calibration of Modern Neural Networks"
  ([arXiv 1706.04599](https://arxiv.org/abs/1706.04599)), for reliability diagrams and ECE.
  Brier score and log-loss are proper scoring rules.
- **Search leaf use.** A leaf evaluator must *rank sibling positions*: states reached after
  different actions in the same turn. Calibration on whole-game positions is necessary but not
  sufficient. The decisive test is therefore playing strength as the `exit_search` leaf, on
  block 53 and then the gate.

## Method

- **Data** (`python -m catan.research.value.data gen|pack`). Our generator plays mixed tables:
  `exit_search` with the MLP or GNN prior, the nego champion, the D5 GNN, `intermediate_trader`,
  intermediate and basic. It keeps 15% of every seat's non-forced decisions and records obs v2,
  z, the game id and intermediate's evaluate margin. The margin is used only for the calibration
  baseline. It is combined with D2's int_v2 shards (intermediate seats, 10% of decisions) and the
  dagger_g1/g1b shards (GNN-learner states). The packer streams the shards into a float16 memmap
  and subsamples whole games.
- **Model** (`python -m catan.research.value.train`). The D5 GraphPolicy, initialised from
  `models/arch_best.pt`. Only the value-head logit is trained, with BCE on z. Validation is the
  games with gid % 10 == 0. The best checkpoint by validation log-loss is kept.
- **Search use.** `value_search:value=<ckpt>` is `exit_search` with leaf sigmoid(V). It works
  with raw q (`qnorm=false`), min-max q, or `leaf=mix` (V + margin / mix_scale).

## Data (v1)

| source | rows | notes |
|---|---|---|
| `runs/value_data/gen` (ours, 6000 games, seeds 70,000,000+) | 610k | mixed tables, all seats, keep 15% |
| D2 `bc_data/int_v2` (40% of games) | 1.38M | intermediate seats in intermediate/basic tables |
| D2 `bc_data/dagger_g1` and `dagger_g1b` | 426k + 191k | GNN-learner states |
| **total** | **2.61M rows / 51.5k games** | 12.6 GB float16 on disk, read by `pread` |

Validation is 5,133 whole games (gid % 10 == 0), subsampled to 60k rows. Training stopped at 0.5
epochs (step 2226, about 25 min on 3 threads), because validation log-loss had flattened
(0.482 after 0.25 epochs, 0.479 after 0.5).

## Calibration (validation, whole held-out games)

| predictor | rows | Brier | log-loss | ECE |
|---|---|---|---|---|
| constant (train base rate 0.291) | 60k | 0.199 | 0.588 | 0.017 |
| logistic(intermediate `evaluate` margin), fit on train | 11.3k (our rows) | 0.179 | 0.541 | 0.031 |
| **value v1**, same 11.3k rows | 11.3k | **0.157** | **0.477** | **0.016** |
| value v1, all validation rows | 60k | 0.158 | 0.479 | 0.005 |

Per source (value v1): ours 0.157 / 0.477, int_v2 0.160 / 0.485, dagger_g1 0.147 / 0.450,
dagger_g1b 0.160 / 0.485 (Brier / log-loss).

Reliability (value v1, 10 bins, mean predicted vs observed):
0.05/0.05, 0.15/0.15, 0.25/0.24, 0.35/0.34, 0.45/0.45, 0.55/0.54, 0.65/0.65, 0.75/0.75, 0.85/0.83, 0.93/0.93.
Every bin is within 0.012 of the diagonal.

## Decisive test: search leaf (block 53, n=100, 1 vs 3)

| prior | leaf | sims | vs basic | vs intermediate | ms/move |
|---|---|---|---|---|---|
| ppo_baseline | `intermediate.evaluate` | 16 | 39% [30, 49] | 20% [13, 29] | 8 |
| ppo_baseline | value v1, raw q | 16 | 42% [33, 52] | 29% [21, 39] | 16 |
| ppo_baseline | value v1, min-max q | 16 | 41% [32, 51] | 26% [18, 35] | 16 |
| ppo_baseline | value v1 + margin (mix) | 16 | 49% [39, 59] | 28% [20, 38] | 16 |
| league_best (GNN) | `intermediate.evaluate` | 16 | 51% [41, 61] | 32% [24, 42] | 9 |
| league_best | value v1, raw q | 16 | 57% [47, 66] | 45% [36, 55] | 14 |
| league_best | value v1 + margin (mix) | 16 | 61% [51, 70] | 46% [37, 56] | 15 |
| league_best | value v1, raw q | 64 | 56% [46, 65] | 52% [42, 62] | about 60 |

On the same seeds and prior, the learned leaf beats `intermediate.evaluate` in every
comparison. The gap is largest vs 3×intermediate: +6 to +14 points.

## Gate (blocks 40–42, n=200, champion combo_zs-v1)

`value_search:checkpoint=models/league_best.pt,value=models/value_best.pt,sims=64,qnorm=false`

| test | candidate | champion | paired |
|---|---|---|---|
| vs 3×basic | **58.0% [51.1, 64.6]** | 46.5% | candidate-only wins 50, champion-only 27, McNemar p = 0.012: **significantly better** |
| vs 3×intermediate | **54.5% [47.6, 61.3]** | 48.0% | 49 vs 36, p = 0.19 |
| mixed table (cand, champ, 2×intermediate) | 76 wins | 79 wins | sign p = 0.87: **parity** |

It is not promoted, because promotion needs a significant head-to-head win. It has the highest
1v3 rates of any agent in the program. Note that it has no trade layer, while the champion's
strength comes partly from trading.

## value_trade: search core + combo/nego trade layer

`catan/research/value/trade.py` subclasses `catan.research.combo.agent.ComboAgent` and replaces
its GNN core with `value_search` (league prior, `models/value_best.pt` leaf, raw q):
- TRADE_RESPONSE and TRADE_CONFIRM decisions go to the nego layer.
- Search never sees OFFER_TRADE.
- When search picks END_TURN or a bank trade, the nego offer head may replace it with a
  domestic offer.
- The nego belief tracker is fed through `observe`.

Tests: `tests/research/value/test_value.py::test_value_trade_routes_trades_to_nego`.

Block 53, n=100:

| agent | vs 3×basic | vs 3×intermediate | mean ms/move (vs basic / vs intermediate) |
|---|---|---|---|
| value_search, 16 sims | 57% [47, 66] | 45% [36, 55] | 14 / 16 |
| value_search, 64 sims | 56% [46, 65] | 52% [42, 62] | 66 / 54 |
| **value_trade, 32 sims** | **69% [59, 77]** | **53% [43, 62]** | **20 / 12** |
| value_trade, 64 sims | 67% [57, 75] | 55% [45, 64] | 40 / 28 |

**Latency.** 32 sims is as strong as 64 here and half the cost. The mean is about 12–20 ms per
decision on one core, but contested MAIN decisions cost several times the mean. The gate
tournament files record `max_sec_per_move`. For human play, 32 sims is the recommended default.

## Analysis rollouts (proposal, not applied: catan/analysis is not mine)

`catan/analysis/engine.py::_rollout_batch` plays every rollout to the end of the game, up to 200
turns: several hundred engine steps of the rollout policy. One value-net call costs about
1–2 ms. A calibrated V at the end of a short horizon could replace the 0/1 outcome:

```python
# in _rollout_batch, behind an opt-in evaluator=ValueEvaluator(models/value_best.pt):
c.step(action); play until me's next turn (as catan.research.exit.search._rollout does)
out.append((evaluator.p_win(c, me), 0, c.victory_points(me)))   # probability, not 0/1
```

Each sample then has far lower variance. Across held-out games, V's Brier is 0.157, while a
single binary rollout outcome is irreducibly noisy (about 2·p(1−p), roughly 0.4 at p = 0.3).
That should mean fewer rollouts for the same verdict confidence. It needs measuring in the
Analysis benchmark (catan/analysis/bench.py) before it's switched on.

## Gate: value_trade (promoted as value_trade-v1)

`python -m scripts.promote --checkpoint models/value_best.pt --agent value_trade --name value_trade --deterministic --selection-meta runs/value_v1/selection.json`
Spec: `value_trade:checkpoint=models/value_best.pt,deterministic=true` (32 sims, league prior).
Selected on block 53. The gate uses blocks 40–42, n=200, against champion combo_zs-v1.

| test | value_trade | champion | paired |
|---|---|---|---|
| vs 3×basic | **71.0% [64.4, 76.8]** | 46.5% | 66 vs 17 discordant, significantly better |
| vs 3×intermediate | **66.5% [59.7, 72.7]** | 48.0% | 56 vs 19 discordant, significantly better |
| mixed table (cand, champ, 2×intermediate) | **102 wins** | 40 wins | sign p = 2e-7 |

Behavior vs 3×intermediate (block 53, n=100, `results/behavior/value_trade_vs_intermediate.json`):
53% win, 8.89 VP, 2.08 cities and 2.32 settlements per game. combo_zs on the same seeds: 49%,
8.58, 1.77 and 2.33. No robber self-blocks.

## Next

- **value v2** on the 204k combo-champion rows (`runs/value_data/gen_combo`). Repack with
  `pack --src runs/value_data/gen:1.0 runs/value_data/gen_combo:1.0 <D2 int_v2>:0.4 <dagger_g1>
  <dagger_g1b>`, then fine-tune from `models/value_best.pt`. The trainer's step-0 row reports v1
  on the combo games first.
- **Self-play data from value_trade itself.** The evaluator should see the new champion's
  positions, then iterate: the ExIt loop with a learned leaf.
- **D14:** distil value_trade's moves into a fast policy for the UI.
- **Analysis rollouts:** add a value-net horizon evaluator (proposal above), measured with
  catan/analysis/bench.py.
