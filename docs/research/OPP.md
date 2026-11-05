# D21: Opponent modelling inside the champion's search (`opp`)

Question: D15 (joint_vt), D19 (setupv) and D17 (evo) all gained against intermediate and lost
against basic. value_trade's search (catan/research/exit/search.py, used by
catan/research/value/search.py and trade.py) plays **every** opponent with the basic bot inside
its simulations, including the opponents' setup replies. Does simulating each opponent with
its own (classified) policy help?

## Literature

- Type-based reasoning (Albrecht & Ramamoorthy, HBA; Albrecht & Stone, "types with
  parameters"): keep a posterior over a fixed set of hypothesised opponent types from their
  observed actions, and plan against the posterior. Weakness: no type may fit. Survey and
  convergence results: [arXiv 1907.06995](https://arxiv.org/pdf/1907.06995v1.pdf),
  [arXiv 1907.09810](https://arxiv.org/pdf/1907.09810.pdf).
- Bayesian Policy Reuse / BPR+ (opponent-type detection and response), and its
  partially observable extension with a VAE (Bayes-Lab):
  [arXiv 1809.04240](https://ar5iv.arxiv.org/html/1809.04240),
  [arXiv 2206.00113](https://arxiv.org/pdf/2206.00113.pdf).
- Type-based MCTS under partial observability (POTMMCP), which plans with a meta-policy over
  opponent types inside MCTS: [arXiv 2306.06067](https://arxiv.org/pdf/2306.06067v1.pdf).
- Opponent models in MCTS improve search when they are (partly) correct, but in multiplayer
  games the agent's own actions often matter more than opponent prediction (Baier &
  Kaisers 2020, cited in the MCTS review [arXiv 2103.04931](https://arxiv.org/pdf/2103.04931)).
- Catan MCTS (Szita, Chaslot & Spronck 2010,
  [Springer](https://link.springer.com/doi/10.1007/978-3-642-12993-3_3)) used a uniform
  opponent model; we found no Catan work with explicit opponent typing.

Adaptation here: a discriminative type classifier (cheaper and easier to train than a policy
likelihood over hidden hands), plus an exact likelihood test where the state is fully
public (setup), with a confidence threshold that falls back to the old model.

## Method

**Features** (`catan/research/opp/features.py`), a pure function of the board layout, the game
config and the public action log, so they are the same from `game.observe_for(seat)` (tested):
- *Setup policy match.* Setup has no hidden state, so the public setup prefix is replayed on a
  fresh `Game` with the same layout. At each of a seat's 4 setup decisions we record whether
  basic's and intermediate's own setup rules would play the action it played (+1 / -1 / 0 not
  yet). This is an exact likelihood test (trader and search share intermediate's setup).
- *Behaviour rates* per own turn: domestic offers (and bundle / overpay shares), accept /
  reject / counter shares of responses, confirms, cancels, bank trades, builds, dev buys and
  plays, knights played before rolling or while robbed, robber placement (own hex, leader on
  hex, pips, desert), steals from the leader, plus log turn counts. 38 features.

**Classifier** (`classifier.py`, `scripts/opp_train.py`): MLP 38-64-64-5, class-weighted
cross-entropy, AdamW, 60 epochs, epoch picked on 10% of training games. Classes: basic,
intermediate, intermediate_trader, search, learned (value_trade-v1, combo_zs, nego, joint_vt,
league arch_rl).

**Data** (`scripts/opp_data.py`): every seat draws a class uniformly, then a spec of that class.
1,500 training games (seeds TRAIN_SEED_BASE + 21e6 + i), 400 test games (held-out block 55),
features snapshotted at global turns -1 (after the first setup round), 0 (end of setup), 4, 8,
12, 16, 24, 32, 48. About 13 minutes on 3 workers. Saved in `runs/opp_data/`, model
`models/opp_best.pt` (32 KB).

**Agent** `opp_value_trade` (`agent.py`): value_trade-v1 with one change. Before every decision
the classifier labels each opponent seat; the class probabilities are summed per rollout policy
(basic -> basic; intermediate and search -> intermediate; trader -> trader without counter
offers, for speed; learned -> `learned`, default basic). If the top policy has probability >=
`threshold` (0.6), that seat's moves inside the search's simulations (and so its setup replies)
are played by that policy; otherwise the seat keeps the basic bot, i.e. value_trade-v1.
`pool=true` additionally gives unsure seats the policy all confident opponents share.
Infoset: `tests/bots/test_infoset_agents.py::test_agent_does_not_peek[opp_value_trade]` passes.

## Results

### Classifier accuracy (400 held-out block-55 games, 1,600 seats per row)

`acc` = exact class (5-way, chance 20%). `policy` = the rollout policy the class maps to is
right (basic / intermediate+search / trader / learned), which is what the search uses.
`conf>=0.6` = share of seats over the agent's threshold and their exact-class accuracy.
Per-seat columns give exact-class accuracy by seat position 0-3.

| global turn | acc | policy | seat 0 | seat 1 | seat 2 | seat 3 | conf>=0.6 (share, acc) |
|---|---|---|---|---|---|---|---|
| -1 (1st setup round) | 30.5% | 48.9% | 40.8 | 44.5 | 20.5 | 16.3 | 11.5%, 95.1% |
| 0 (setup done) | 51.5% | 70.3% | 47.3 | 52.5 | 50.0 | 56.3 | 33.9%, 96.3% |
| 4 | 70.3% | 86.4% | 71.8 | 66.3 | 68.8 | 74.3 | 54.8%, 91.2% |
| 8 | 72.3% | 89.1% | 72.3 | 71.3 | 71.0 | 74.5 | 63.6%, 87.8% |
| 16 | 75.8% | 92.5% | 76.0 | 73.8 | 75.0 | 78.3 | 72.8%, 86.8% |
| 24 | 77.9% | 93.3% | 76.5 | 76.0 | 77.5 | 81.5 | 77.1%, 85.9% |
| 48 | 82.2% | 93.8% | 82.1 | 80.3 | 81.6 | 84.9 | 83.4%, 86.6% |

Per class at turn 4 / 16 / 48: basic 100 / 100 / 100%; intermediate 42 / 68 / 84%;
intermediate_trader 78 / 94 / 97%; search 55 / 39 / 53%; learned 72 / 74 / 74%.
Nearly all errors are intermediate vs search. Those two share every rule except search's
rollout-based main-phase choice, and they map to the same rollout policy. Learned agents are
mostly confused with the trader: nego is intermediate plus a trader-like head. Basic is found
from its setup alone (89% at turn 0) and is never wrong after turn 4.

Grouping matters. With the threshold on the top *class*, an intermediate seat's probability is
split about 0.5/0.5 with search, and 81% of intermediate seats at turn 8 were "unsure". That
left them on the basic rollout, so the first screen was nearly value_trade-v1. Thresholding the
summed policy probability fixed it: 93% of intermediate seats at turn 8 are modelled as
intermediate, and value_trade seats stay on basic (92%).

### In games: `opp_value_trade` vs a same-seat value_trade-v1 copy (deterministic)

Focal `opp_value_trade:checkpoint=models/value_best.pt,deterministic=true` against the copy
`value_trade:checkpoint=models/value_best.pt,deterministic=true,sims=32` (the same agent under
a distinct spec string), on the same seeds and seatings. Mixed table = focal + value_trade-v1
champion seat + 2x intermediate, all 12 seatings (`combo_eval --rotation all`). 1v3 tables use
cyclic seats. Per-game records are in `runs/opp_eval/tournaments/`; pooled with
`scripts/opp_pool.py` (`runs/opp_eval/pooled_60_61.txt`).

**Report blocks 60 + 61 (300 games per block per test, n=600 per test, paired):**

| test | opp_value_trade | value_trade-v1 copy | paired diff [95% CI] | McNemar p (only opp / only copy won) |
|---|---|---|---|---|
| mixed table | 250/600 = 41.7% [37.8, 45.7] | 225/600 = 37.5% [33.7, 41.4] | **+4.2 [+0.7, +7.6]** | **0.024** (69 / 44) |
| 3x intermediate | 369/600 = 61.5% [57.5, 65.3] | 367/600 = 61.2% [57.2, 65.0] | +0.3 [-3.5, +4.2] | 0.93 (70 / 68) |
| 3x basic | 393/600 = 65.5% [61.6, 69.2] | 393/600 = 65.5% [61.6, 69.2] | 0.0 | 1 (0 / 0: identical games) |

Per block, mixed: block 60 +6.3 [+1.7, +11.0] (p=0.011); block 61 +2.0 [-3.1, +7.1] (p=0.53).
Champion-seat wins: 245 in the opp tables vs 251 in the copy tables.

Against 3x basic the two agents play **identical games** (0 discordant in 600, and again in
200 on block 55). Basic is identified with certainty once its setup is visible, and a confident
basic seat gets the basic rollout, which is v1's model. Unsure seats fall back to v1 as well.
That is the "fall back when unsure" property: the add-on cannot lose to basic tables.

**Selection block 55 screens (n=200 per test).** vs 3x basic: identical (132 vs 132). vs 3x
intermediate: threshold-on-class +2.0 (p=0.64), grouped (final) +1.0 (p=0.88), pool=true +2.0,
learned=intermediate +3.5 (p=0.43). Mixed: 62-67 vs 83 wins. **These mixed screens are not
paired.** Before platform-rebuild ae818e5, `lineups(rotation="all")` sorted seatings by spec
string. `opp_value_trade:...` and `value_trade:...,sims=32` sort on opposite sides of the
champion's spec, so the same (seed, rotation) seated focal and champion swapped. I found this
from a seat-assertion failure; the lead fixed it at the source. The report runs above use the
fixed ordering. Read unpaired, the block-55 screen pointed the other way (31-33% vs 41.5%,
about -2 SE). The paired n=600 report supersedes it, but it is a reason to treat the mixed gain
as modest.

### Gate: `opp_vt-v1` vs champion `value_trade-v1` (blocks 40-42, `--deterministic`): NOT PROMOTED

`python -m scripts.promote --checkpoint models/value_best.pt --agent opp_value_trade --name opp_vt
--deterministic --workers 2 --selection-meta runs/opp_eval/selection_meta.json` (exit 2).

| test | opp_vt-v1 | value_trade-v1 | paired vs v1's gate games |
|---|---|---|---|
| vs 3x basic, n=200 | 71.0% [64.4, 76.8] | 71.0% | identical games (0 / 0 discordant) |
| vs 3x intermediate, n=200 | 66.5% [59.7, 72.7] | 66.5% | 20 / 20 discordant |
| mixed table, all 12 seatings, n=600 | 247 wins (41.2% [37.3, 45.1]) | 237 wins | sign p=0.68: not better |

Not worse anywhere, but the gate's mixed table doesn't show the report blocks' +4.2. The fair
reading: the opponent model is at best a small mixed-table gain (somewhere in 0 to +5 points)
and is neutral against both rule-bot tables.

## Interpretation

- **Opponent typing is easy; using it in this search buys little.** Basic is recognised from
  its setup alone, and the rollout policy is right for 86% of seats by turn 4 and 93% by turn 16.
  But value_trade's simulations stop at the agent's next decision, so the opponent model only
  shapes about one round of opponent moves before the value net scores the leaf. In 3x
  intermediate, simulating intermediates as intermediates instead of basic changes ~23% of
  games (138 discordant of 600) with no net effect. This matches the MCTS literature: in
  multiplayer games the agent's own actions matter more than opponent prediction (Baier &
  Kaisers 2020).
- **The mixed-table gain (+4.2, p=0.024) is the only significant effect.** It shows up where the
  table mixes a learned agent (kept on the basic model) with intermediates (now modelled as
  intermediate). Block 60 carries most of it.
- **The D15/D19/D17 pattern (gain vs intermediate, loss vs basic) is not explained by a uniform
  opponent model inside the search.** v1 already simulates everyone as basic, so basic tables
  are modelled exactly, and switching intermediates to their true policy does not move the
  3x intermediate result. Those losses are more likely in the add-ons themselves (e.g. joint_vt's
  trade head was trained mostly on negotiating tables). D18's per-opponent trade features are
  the better place for the type signal.

## Next steps

1. **Put the type signal where opponents act on it: trades.** Feed the classifier's per-seat
   probabilities to the nego/joint trade head (D18 item 3), and condition the offer layer on
   "this seat is basic: accepts only build-completing trades".
2. **Longer opponent horizon.** The opponent model only acts for about one round per simulation.
   A search that continues to my decision after next, or a value net conditioned on opponent type,
   would let a correct model matter more.
3. **Opponent offers in rollouts.** Rollouts still replace opponents' domestic offers with
   END_TURN (`exit/search.py`). Intermediate and trader opponents propose trades often; allowing
   them for confidently typed seats is a small additive change to `_rollout`.
4. **Setup replies.** At the first setup round only 11.5% of seats clear the threshold (seats
   after me have no evidence yet), so setup replies still mostly use basic. A table prior
   (`pool=true`) changed nothing measurable on block 55.

## Reproduce

```
python -m scripts.opp_data --games 1500 --split train --out runs/opp_data/train.npz --workers 2
python -m scripts.opp_data --games 400 --split test --out runs/opp_data/test.npz --workers 2
python -m scripts.opp_train --train runs/opp_data/train.npz --test runs/opp_data/test.npz \
    --run runs/opp_clf --save models/opp_best.pt
python -m scripts.combo_eval --specs "opp_value_trade:checkpoint=models/value_best.pt,deterministic=true" \
    "value_trade:checkpoint=models/value_best.pt,deterministic=true,sims=32" \
    --champion "value_trade:checkpoint=models/value_best.pt,deterministic=true" \
    --tests mixed intermediate basic --games 300 --block 60 --rotation all --workers 2 \
    --out runs/opp_eval/report60.json --save-dir runs/opp_eval      # and --block 61
python -m scripts.opp_pool --dir runs/opp_eval/tournaments --files report60 report61 \
    --a "opp_value_trade:checkpoint=models/value_best.pt,deterministic=true" \
    --b "value_trade:checkpoint=models/value_best.pt,deterministic=true,sims=32" \
    --champion "value_trade:checkpoint=models/value_best.pt,deterministic=true"
```
