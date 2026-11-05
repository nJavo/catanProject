# D18: Fix joint_vt's loss to rule-based opponents (`jointb`)

Goal: keep D15's mixed-table edge for the core-aware trade head without losing ground against 3×basic.
At the D15 gate, `joint_vt-v1` reached 59.5% vs 3×basic, against value_trade-v1's 71.0% (p=0.008).

## Literature

- **Opponent modelling.** He et al., [Opponent Modeling in Deep RL (DRON), ICML 2016](https://arxiv.org/abs/1609.05559).
  The policy is conditioned on features of each opponent's observed behaviour, either concatenated or
  through a mixture of experts. We use the concatenation form, with hand-built public-history features.
- **Catan negotiation.** Keizer et al.,
  [Evaluating Persuasion Strategies and Deep RL methods for Negotiation Dialogue agents, EACL 2017](https://aclanthology.org/E17-2077/).
  In Settlers of Catan, deep-RL trade policies beat rule-based and supervised negotiators. How much they
  gain depends on the opponent's acceptance behaviour.
- **Expert iteration.** Anthony et al., [Thinking Fast and Slow with Deep Learning and Tree Search, 2017](https://arxiv.org/abs/1705.08439).
  The search-labelled targets are distilled into a fast policy. D15's counterfactual option labels are an
  ExIt-style target, and D18 keeps that scheme.
- **Opponent-type gating.** D21 (`catan/research/opp`) built an online opponent-type classifier from the
  public log. The final variant uses it to choose which trade layer to play. This is a hard version of
  DRON's mixture of experts, with D21's classifier as the gate.

## Method

Code is in `catan/research/jointb/`.

1. **Rebalanced labels** (`labels.py`). Same scheme as D15: the learner plays combo_zs, the top 8 options
   are replayed on clones with common random numbers, and the D12 value net scores each branch end. The
   table mix is now 30% 3×basic, 25% 3×intermediate, 20% 3×intermediate_trader, 15% the gate mixed table
   and 10% 3×combo_zs. D15 had 10% basic and 40% mixed.
2. **Longer horizon.** A branch runs until the learner's next turn starts, so a full round for offers.
   Every possible partner's next turn, including any build it completes with the traded cards, is inside
   the branch. D15 stopped at the end of the current turn.
3. **Opponent-history features** (`features.py`, `opp_history`). For each opponent in relative seat order,
   from the public log: offers per own turn, log1p(responses), smoothed accept rate and smoothed counter
   rate. That is 12 more inputs, so the state is 61 nego + 258 core + 12 = 331.
4. **Training** (`train.py`). Same loss as D15: soft targets softmax(q/τ), weighted by the option spread,
   plus a BC anchor toward the nego choice. Initialisation is either D15's head with zeroed new columns
   (`--init joint`) or the nego BC head (`--init nego`). Held-out regret is reported per table.
5. **Agents.**
   - `jointb_vt` (`agent.py`) is value_trade's core with the jointb head.
   - `jointb_gated` (`gated.py`) is the lead's follow-up. It uses v1's nego layer when the trade
     partner, or every possible partner of an offer, is classified basic by D21's classifier
     (P(basic) >= 0.1). Otherwise it uses the jointb head.

Labels: 2,040 games and 89k labelled decisions, about 73 minutes on 2–3 workers. Seeds start at
TRAIN_SEED_BASE + 18e6. The job was stopped at 2,040 of a planned 4,500 games to stay within the CPU cap.

| table | games | labelled decisions | mean option spread |
|---|---|---|---|
| 3×basic | 616 | 12,483 (14%) | 0.0297 |
| 3×intermediate | 521 | 32,236 | 0.0179 |
| 3×intermediate_trader | 398 | 18,456 | 0.0184 |
| mixed (combo + 2×int) | 307 | 17,170 | 0.0166 |
| 3×combo_zs | 198 | 9,036 | 0.0197 |

Basic tables give about 40 trade decisions per game, against about 120 for intermediate. Basic opponents
never propose, so the learner only gets offer and confirm decisions there.

## Results

### Offline: held-out regret (204 held-out games, 6,430 decisions)

Regret is the best evaluated option's q minus the chosen option's q, in win probability. The figures for
nego BC and D15 come from these new labels.

| head | all | 3×basic | 3×int | 3×trader | mixed | 3×combo |
|---|---|---|---|---|---|---|
| nego BC (v1's layer) | 0.0213 | 0.0278 | 0.0206 | 0.0174 | 0.0181 | 0.0254 |
| D15 head (joint_vt), init of r1/r4 | 0.0188 | **0.0292** | — | — | — | — |
| r1 joint init, τ .002, bc .2, lr 5e-4 | 0.0187 | 0.0286 | 0.0173 | 0.0151 | 0.0158 | 0.0180 |
| r2 nego init, same | 0.0194 | 0.0279 | 0.0188 | 0.0162 | 0.0163 | 0.0185 |
| r3 nego init, τ .005, bc .5 | 0.0196 | 0.0278 | 0.0182 | 0.0167 | 0.0164 | 0.0209 |
| r4 joint init, lr 2e-3, 16 ep (ep6) | 0.0184 | **0.0267** | 0.0173 | 0.0159 | 0.0143 | 0.0189 |
| **r5 nego init, lr 2e-3, bc .1, 16 ep (ep14)** | **0.0183** | 0.0285 | **0.0166** | **0.0149** | 0.0147 | 0.0180 |

On the new labels, the D15 head is worse than nego BC on basic tables (0.0292 vs 0.0278). This matches its
gate loss. No head improves much over nego on basic. Offline basic regret did not predict basic win rate
in games: r4 has the best basic regret but r5 played better vs basic (see below).

### Screen: block 58, paired against a same-seat value_trade-v1 copy, n=600 per test, cyclic seats

The copy spec is `value_trade:checkpoint=models/value_best.pt,deterministic=true,offers=true`. The mixed
table is focal + value_trade-v1 + 2×intermediate.

| focal | vs 3×basic | vs 3×intermediate | mixed (focal wins / champion-seat wins) |
|---|---|---|---|
| value_trade-v1 copy | 61.7% [57.7, 65.5] | 59.8% [55.9, 63.7] | 236 / 213 |
| joint_vt (D15) | 62.2% (+0.5 [−4.3, +5.3]) | 72.8% (+13.0, p=1e-7) | 293 / 180 (+9.5, p=2e-4) |
| jointb_vt r4 | 61.7% (+0.0 [−4.6, +4.6]) | 72.5% (+12.7, p=4e-7) | 306 / 182 (+11.7, p=7e-6) |
| **jointb_vt r5** | **66.0% (+4.3 [−0.4, +9.1], p=0.09)** | **75.3% (+15.5, p=3e-9)** | **313 / 177 (+12.8, p=1e-6)** |

Brackets give the paired difference against the copy with its 95% CI, followed by the McNemar p-value.
Block 58 does not reproduce D15's basic gap: there, joint_vt is level with the copy at n=600.

Trades vs 3×basic, per game:

| agent | offers | executed | maritime |
|---|---|---|---|
| copy | 34.4 | 3.98 | 4.16 |
| joint_vt | 33.6 | 3.26 | 4.40 |
| r5 | 24.4 | 4.03 | 4.33 |

The D15 symptom of fewer executed offers is gone. r5 makes fewer offers, but more of them are accepted
(17% vs 12% success).

r5 was selected on block 58 and bundled as `models/jointb_best.pt`. It is the league GNN weights plus the
`joint_head` with extra=270.

### Gate: `jointb_vt-v1` vs value_trade-v1 (blocks 40–42, `--deterministic`): NOT PROMOTED

| test | jointb_vt-v1 | value_trade-v1 | paired |
|---|---|---|---|
| mixed table, all 12 seatings, n=600 | **286 wins vs 211** (47.7% [43.7, 51.7]), sign p=0.0009: better | — | — |
| vs 3×intermediate, n=200 | **76.5%** [70.2, 81.8] | 66.5% | 46 vs 26 discordant, p=0.024: better |
| vs 3×basic, n=200 | 62.5% [55.6, 68.9] | 71.0% | 23 vs 40 discordant, **p=0.043: worse** |

Against D15's gate, the result moves on every test: basic 59.5% → 62.5% (p 0.008 → 0.043), intermediate
70.5% → 76.5% (level → significantly better), and the mixed table stays clearly ahead. I re-ran the
champion's cached block-40 baseline under the current code and it reproduced 142/200 exactly, so the gap
is not stale data.

On block 40, jointb executes more offers than v1 (4.17 vs 3.98) and makes more bank trades (4.79 vs 4.38).
The D15 explanation, too few accepted offers, does not account for the remaining gap. Pooling the paired
basic evidence gives about +1 pp (block 58, n=600: +4.3; block 40, n=200: −8.5). The basic difference
is small and block-dependent, but at the gate it is significant.

### Follow-up: `jointb_gated` (D21 classifier gates the trade layer)

The lead proposed this follow-up. `jointb_gated:checkpoint=models/jointb_best.pt` uses the same head as
jointb_vt-v1. D21's public-log classifier (`models/opp_best.pt`) decides, per trade decision, whether to
use that head or v1's nego layer. A seat counts as basic when P(basic) >= 0.1. In the first turns, basic
seats dip to about 0.2 and negotiators sit near 0.0, so at a 0.5 threshold 2 of 4 test games vs 3×basic
diverged from v1. At 0.1, 12 of 12 were identical.

**Screen: block 60, paired against a same-seat v1 copy, n=300 per test, cyclic seats.** Block 61 was
stopped early at the lead's request, so the gate could run.

| test | jointb_gated | v1 copy | paired |
|---|---|---|---|
| vs 3×basic | 68.3% [62.9, 73.3] | 68.3% | +0.0, identical games (205/205) |
| vs 3×intermediate | **78.0%** [73.0, 82.3] | 61.3% | **+16.7 [+10.1, +23.3], p=2e-6** |
| mixed (focal / champion seat) | **158 / 78** | 123 / 99 | **+11.7 [+4.6, +18.7], p=0.002** |

**Gate: `jointb_gated-v1` vs value_trade-v1 (blocks 40–42, `--deterministic`): PROMOTED, new champion.**

| test | jointb_gated-v1 | value_trade-v1 | paired |
|---|---|---|---|
| mixed table, all 12 seatings, n=600 | **286 wins vs 211** (47.7% [43.7, 51.7]), sign p=0.0009 | — | — |
| vs 3×intermediate, n=200 | **76.5%** [70.2, 81.8] | 66.5% | 46 vs 26 discordant, p=0.024 |
| vs 3×basic, n=200 | 70.5% [63.8, 76.4] | 71.0% | 0 vs 1 discordant: v1's play |

The mixed and intermediate numbers match jointb_vt-v1 game for game. Those tables have no basic seats, so
the gate always selects the jointb head. Against 3×basic, 199 of 200 games are v1's. The one game that
differs comes from an early, uncertain classification.

Behavior report: `results/behavior/20261104-151852-vs-intermediate.json` (68% vs 3×intermediate, n=100) and
`results/behavior/20261104-141143-vs-basic.json` (jointb_vt).

## Interpretation

- **The rebalanced, longer-horizon labels with opponent features improved the head against negotiators.**
  Paired against a v1 copy on block 58, the gain was +15.5 vs 3×intermediate and +12.8 in the mixed table.
  D15's head gained +13.0 and +9.5. The head no longer gets fewer offers accepted against basic.
- **The head did not fix the basic gap.** At the gate, vs 3×basic went from 59.5% to 62.5%, still
  significantly below 71.0%. On block 58 the head was +4.3 [−0.4, +9.1]. The basic difference is small and
  depends on the seed block. Offline basic regret barely moved: no head beat nego BC on basic tables by
  more than 0.001 in win probability.
- **The basic gap is fixed by gating, not by learning.** Basic opponents give a cheap, known best response:
  v1's layer. D21's classifier identifies them from the public log. Using the learned head only where it
  helps keeps v1's 71% vs basic and all of jointb's gains elsewhere. This is DRON's mixture of experts with
  a hard gate taken from a separate classifier.

## Next steps

1. **Block 61 replication** of the gated screen. It was stopped early, and the gate result already stands.
2. **More labels.** The run stopped at 2,040 of 4,500 games. Basic tables are now covered by v1's layer,
   so future label budgets can go to negotiator tables only (3×intermediate, trader, value_trade, mixed),
   where the head gains most.
3. **Learned-opponent tables.** No label table has value_trade or jointb opponents, and the mixed table
   is where the next champion has to win. Add value_trade seats to the label tables (slower, about 32 sims
   per move).
4. **Stacking.** D24 is combining the gated trade layer with a robber layer.
