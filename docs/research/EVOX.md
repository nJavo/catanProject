# D29: island-model evolution of a heuristic evaluator (prefix `evox`)

Package `catan/research/evox/`, agent `evox`, selection block 56, report blocks 60/61.
Earlier population attempts were D3 (PBT on the MLP), D16 (noise-robust PBT on the GNN) and
D17 (Elo evolution of value_trade knobs). None of them showed selection doing the work. D3's
selection was noise. D16's gain came from training. D17's knob landscape was flat, and its
8-member in-population fitness did not carry over to the heuristic tables. D29 changes three
things:

1. **The genome.** It is cheap enough that one generation is seconds of games.
2. **The scale.** The population is 120 islands × 6 members, run for thousands of generations.
3. **The starting point.** Evolution starts from random weights, so there is room to improve and
   no ceiling set by a hand-tuned start.

```bash
PY="uv run python"
$PY -m catan.research.evox.loop --run-id evox_r1 --islands 120 --members 6 --generations 2000 --seed 1
$PY -m catan.research.evox.loop --run-id evox_r1 --resume --generations 3000   # continue
$PY -m scripts.evox_eval --run runs/evox_r1            # paired final test, blocks 60/61
$PY -m pytest tests/research/evox -q
```

## Literature

I read the papers below as PDFs or abstracts, as marked. The video sources come from caption
transcripts fetched with the watch tool. Where a video had no captions, the entry says so.

**Evolving game evaluators**

- **Chellapilla & Fogel 2001**, "Evolving an expert checkers playing program without using
  human expertise", IEEE TEC 5(4) ([PDF](http://www.sci.brooklyn.cuny.edu/~sklar/teaching/f05/alife/papers/ali-Evolving-an-expert-checkers-playing-program-without-using-human-expertise.pdf);
  book: Fogel, *Blondie24*, 2001).
  - Setup: 15 parents and 15 offspring. Each network played 5 games per generation against
    random population members (+1 win, 0 draw, −2 loss). The best 15 survived.
  - Mutation was self-adaptive per weight, with σ' = σ·exp(τN) and τ = 1/√(2√n).
  - The run took 840 generations over 6 months and reached expert level with no human
    knowledge in the weights.
  - **Taken:** self-adaptive σ, and very few noisy games per individual per generation.
- **Pollack & Blair 1998**, "Co-evolution in the successful learning of backgammon strategy",
  Machine Learning 32 ([PDF](https://cgi.cse.unsw.edu.au/~blair/pubs/1998PollackBlairML.pdf)).
  - A (1+1) hill-climber: the champion plays a mutant on paired games (same dice, colours
    swapped).
  - When the challenger wins, the champion moves only partway: champion := 0.95·champion +
    0.05·challenger. The win bar rose over time, from 3 of 4 to 5 of 6 to 7 of 8.
  - **Taken:** paired evaluation (common random numbers), and caution against "lucky novice
    challengers", handled here by shrinkage.
- **Lucas & Runarsson 2006**, "Temporal difference learning versus co-evolution for acquiring
  Othello position evaluation", CIG ([abstract](https://www.sciweavers.org/node/100628)).
  - TD learns faster. Well-tuned coevolution reaches better weighted piece counters.
  - Parent-child averaging was essential.
- **Szubert, Jaśkowski & Krawiec 2009**, "Coevolutionary temporal difference learning for
  Othello", CIG ([PDF](https://mszubert.github.io/papers/Szubert_2009_CIG.pdf)).
  - Setup: population 50 in round robin. A hall-of-fame variant adds 50 HoF games per
    individual and archives the best of each generation.
  - Results: the HoF hybrids did best, and zero-initialised weights beat random init.
- **Jaśkowski et al. 2013**, "Improving coevolution by random sampling", GECCO
  ([DOI](https://doi.org/10.1145/2463372.2463512)). Evolution against fixed or random opponents
  generalised better than pure coevolution, and adding them to coevolution recovered the gap.
- **Szubert et al. 2013**, IEEE TCIAIG 5(3)
  ([PDF](https://www.cs.put.poznan.pl/kkrawiec/wiki/uploads/Research/2012TCIAIG.pdf)).
  - Evolution won against the fixed heuristic it trained on. Coevolution plus TD generalised
    best to unseen opponents.
  - **Taken:** fitness here mixes peers, a hall of fame and fixed anchors.

**Competitive coevolution and measuring progress**

- **Rosin & Belew 1997**, "New methods for competitive coevolution", Evolutionary Computation
  5(1) ([DOI](https://doi.org/10.1162/evco.1997.5.1.1)).
  - Introduces the hall of fame (old champions kept as opponents), competitive fitness sharing
    and shared sampling.
  - **Taken:** every checkpoint champion joins a HoF, and a third of the opponent seats are
    drawn from it.
- **Stanley & Miikkulainen 2002**, NEAT, Evolutionary Computation 10(2)
  ([PDF](http://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf)).
  **Stanley & Miikkulainen 2004**, "Competitive coevolution through evolutionary
  complexification", JAIR 21 ([arXiv](https://arxiv.org/abs/1107.0037)): the dominance
  tournament.
  **Stanley et al. 2009**, HyperNEAT, Artificial Life 15(2).
  **Gauci & Stanley 2010**, HyperNEAT checkers, Neural Computation 22(7)
  ([DOI](https://doi.org/10.1162/neco.2010.06-09-1042)).
  - These methods search network topology. With 27 fixed features there is no topology to
    search, so they are not used here.
- **Miconi 2009**, "Why coevolution doesn't 'work': superiority and progress in coevolution",
  EuroGP ([entry](https://gpbib.cs.ucl.ac.uk/gp-html/Miconi_2009_eurogp.html)).
  - Distinguishes local progress (against current opponents), historical progress (against past
    opponents) and global progress (against everything).
  - CIAO plots and master tournaments only measure historical progress, because the opponents
    come from the run itself.
  - **Taken:** the success curve here is global. The champion plays a fixed panel that never
    enters fitness (random, basic, intermediate), on report-block seeds that selection never
    reads.
- **Cliff & Miller 1995**, "Tracking the Red Queen", ECAL
  ([DOI](https://doi.org/10.1007/3-540-59496-5_300)): CIAO plots.
- **Floreano & Nolfi 1997**, "God save the Red Queen!", GP-97: the master tournament, where each
  generation's best plays all past bests.
- **Ficici & Pollack 2003**, "A game-theoretic memory mechanism for coevolution", GECCO: the
  Nash memory.
- **de Jong 2004**, "The Incremental Pareto-Coevolution Archive", GECCO
  ([PDF](https://cs.york.ac.uk/rts/docs/GECCO_2004/Conference%20proceedings/papers/3102/31020525.pdf)).
- **Klijn & Eiben** (speaker from VU Amsterdam, names from the caption transcript), "A coevolutionary approach to deep multi-agent RL", GECCO 2021 poster talk
  ([video](https://www.youtube.com/watch?v=GOpzjJzpWrw)).
  - Coevolutionary ES and GA with a hall of fame, where each individual is evaluated against the
    newest HoF members [03:01–03:28]. They evaluated against a random agent [03:57].
  - Progress did not climb monotonically. They report agents stuck in local optima and agent
    cycling [04:25–04:49].
  - **Taken:** the reason a fixed panel is needed.

**Evolution strategies and GAs at scale**

- **Salimans et al. 2017**, "Evolution Strategies as a Scalable Alternative to RL"
  ([arXiv 1703.03864](https://arxiv.org/abs/1703.03864)).
  - Uses antithetic perturbations and rank fitness shaping. Workers share seeds, so they only
    exchange scalar returns. σ is fixed.
  - Paper-presentation video ([Schörkmeier](https://www.youtube.com/watch?v=s8O-Tkhsjk0)):
    - Workers are initialised with common random seeds and exchange only accumulated returns
      [03:30–04:31].
    - ES is robust to sparse versus dense rewards and to long horizons [05:06–05:41].
    - ES needs about 3–10× more data than policy gradients [06:12].
- **Such et al. 2017**, "Deep Neuroevolution"
  ([arXiv 1712.06567](https://arxiv.org/abs/1712.06567)).
  - Truncation selection with mutation only.
  - Noise handling: the top 10 candidates get 30 extra episodes each before the elite is chosen.
  - **Taken:** at each checkpoint the top 6 by training fitness are re-evaluated on fresh
    selection-block seeds before a champion is crowned.
- **Hansen 2016**, "The CMA Evolution Strategy: A Tutorial"
  ([arXiv 1604.00772](https://arxiv.org/abs/1604.00772)).
  **Hansen et al. 2009**, "A method for handling uncertainty in evolutionary optimization",
  IEEE TEC 13(1) ([PDF](https://www.lri.fr/~hansen/TEC2009.pdf); abstract only). UH-CMA-ES
  re-evaluates a few solutions per generation and raises σ or the evaluation count when noise
  scrambles the ranking.
- **Jaderberg et al. 2017**, PBT ([arXiv 1711.09846](https://arxiv.org/abs/1711.09846)).
  Jaderberg's FTW talk ([video](https://www.youtube.com/watch?v=dltN4MxV1RI)) has no captions;
  that is D30's territory.

**Population structure and diversity**

- **Whitley, Rana & Heckendorn 1998**, "The island model genetic algorithm"
  ([PDF](https://neo.lcc.uma.es/Articles/WRH98.pdf)). Islands help on separable problems.
  Migration frequency changes which optimum the islands converge to.
- **Cantú-Paz 2001**, "Migration policies, selection pressure, and parallel EAs", J.
  Heuristics 7 ([DOI](https://doi.org/10.1023/A:1011375326814)). Best-replaces-worst migration
  raises selection pressure the most.
  - **Taken:** a ring topology with migration every 10 generations.
- **Mouret & Clune 2015**, MAP-Elites ([arXiv 1504.04909](https://arxiv.org/abs/1504.04909)).
  **Pugh, Soros & Stanley 2016**, quality diversity, Frontiers in Robotics and AI 3:40
  ([DOI](https://doi.org/10.3389/frobt.2016.00040)).
  - Summary video ([QD explainer](https://www.youtube.com/watch?v=DlVinRLhWvU)): keep an archive
    of many good and different solutions instead of one optimum [02:02–02:27].
  - Not used in r1. A MAP-Elites archive over play style (city-, road- or dev-heavy) is a
    candidate for a more diverse hall of fame.
- **Risto Miikkulainen on the Lex Fridman Podcast #177**
  ([video](https://www.youtube.com/watch?v=CY_LEa9xQtg)).
  - Population methods suit domains where trials are cheap (simulators) and only the final
    best candidate matters [49:45–52:03].
  - Biology works with "deep time, weak pressure and large populations" [104:52].
  - **Taken:** the argument for many islands at low per-member evaluation cost.

**Noisy fitness**

- **Beyer 2000**, "Evolutionary algorithms in noisy environments", CMAME 186
  ([DOI](https://doi.org/10.1016/S0045-7825(99)00386-2)).
- **Jin & Branke 2005**, "Evolutionary optimization in uncertain environments: a survey", IEEE
  TEC 9(3) ([PDF](https://www.cs.le.ac.uk/people/sy11/ECiDUE/JinBranke-TEVC05.pdf)).
- **Fitzpatrick & Grefenstette 1988**, Machine Learning 3
  ([DOI](https://doi.org/10.1007/BF00113893)): "in some cases, more efficient search results
  from less accurate evaluations", meaning a big population beats resampling.
  - **Taken:** each member plays 1 game per generation. Its fitness is the lifetime mean, so
    survivors sharpen with age, and it is shrunk toward the population mean so a lucky one-game
    child does not take over.

**Catan**

- **Szita, Chaslot & Spronck 2010**, "Monte-Carlo Tree Search in Settlers of Catan", ACG
  ([DOI](https://doi.org/10.1007/978-3-642-12993-3_3)).
- **"Evolvers of Catan"** (Carleton comps 2021,
  [PDF](https://cs.carleton.edu/cs_comps/2021/game-ai/Final-Results/Evolvers-of-Catan-Website/files/Evolvers-of-Catan-Technical-Documentation.pdf)).
  Genetic programming in JSettlers, with fitness = mean score + win bonus against the JSettlers
  bots. The result was "competitive with or somewhat better than" those bots.
- **catanatron** (`players/value.py`) uses hand-set weights. I found no published GA-tuned Catan
  weight vector.

**Videos tried without usable content**

- Jaderberg FTW: no captions.
- Miikkulainen clip `R6zStewfuZs`: no captions.
- Jeff Clune HORSE2016 (`_QmSlzvVDbs`): about image-classifier understanding, off topic.
- AlphaZero Connect-4 explainer (`yzwjnZiwHUg`): only German auto-captions, and about
  AlphaZero, not evolution.
- I found no Fogel/Blondie24 talk on YouTube, so that source is the paper and the book only.

## Method

**Genome** (`catan/research/evox/agent.py`):
- 27 genes covering every hand-set constant of `intermediate`:
  - the 13 `evaluate` weights (production, diversity, spot quality at 0/1/2 roads, hand,
    dev cards, knights, road length, ports, dev-card EV, overflow);
  - 8 opening-settlement constants;
  - the domestic-trade thresholds (accept margin, margin per VP the proposer leads by, offer
    threshold);
  - the knight bonus;
  - 2 robber constants.
- Weight = z · SCALE. SCALE is a per-feature normaliser of about 1 / typical feature size, so
  z ~ N(0, 1) is a random evaluator with random signs. The VP term is fixed at 1.
- `INTERMEDIATE_Z` reproduces `intermediate` move for move (tested on full game logs).
- `agent_x.py` adds 11 more genes (goal weights, hand-score and resource-value constants from
  `catan.bots.common`), 38 in total. That reproduces intermediate too. It was written for an r3
  that has not run.
- One game costs about 0.02–0.05 s on one core.

**Loop** (`catan/research/evox/loop.py`). Islands of M members. Each generation:
- **Tables.** Every member plays 1 game. Tables seat `per_table` members, and the other seats
  are opponents: basic, intermediate, or a hall-of-fame genome.
- **Common random numbers.** Table t of every island uses the same seed (selection block 56,
  offsets 0..49,999), the same seating and the same opponents.
- **Fitness.** The lifetime mean of win + VP/20, shrunk toward the population mean with 4
  virtual games. Survivors get sharper with age, and a lucky one-game child cannot take over.
- **Reproduction.** The bottom `replace` members of each island are replaced by children of the
  top half:
  - binary tournament;
  - uniform crossover with p 0.3;
  - self-adaptive Gaussian mutation, σ' = σ·exp(0.3·N), σ ∈ [σ_min, 1].
- **Migration.** Every 10 generations, along a ring, each island's best replaces its neighbour's
  worst.

**Checkpoints** (every 40–50 generations):
1. The top candidates by training fitness are re-evaluated 1v3 vs 3×basic and 3×intermediate
   on block 56 offsets 50,000+.
2. The best becomes the checkpoint champion and joins the hall of fame.
3. The champion is measured 1v3 vs 3×random, 3×basic and 3×intermediate on report block 60
   (seeds 0..49 × 4 seats = 200 games per opponent). The same seeds are used at every
   checkpoint, and selection never reads them.
4. Elo is a 1-D Bradley-Terry fit of those games against fixed anchor ratings (random = 1000;
   basic and intermediate rated from a mixed anchor tournament on block 60 offsets 20,000+).
   The SE treats the 3 pairs of a game as independent, so it is optimistic.

**Final pick** (`scripts/evox_pick.py`). The last 10 checkpoint champions are re-evaluated on
fresh block-56 seeds (offsets 60,000+, 200 games per opponent). This is Such et al.'s elite
re-evaluation, done because the per-checkpoint pick is a max of noisy estimates.

**Final test** (`scripts/evox_eval.py`).
- The genomes compared: the picked genome, its generation-0 ancestor (followed down the parent
  chain), the all-zero genome, and the intermediate genome.
- Each plays the same 600 games per table: 1v3, 75 seeds on block 60 (offsets 1,000+, after the
  curve's seeds) plus 75 on block 61, × 4 seats.
- The test is an exact McNemar test on the discordant games.

**Logging.** `runs/evox_r*/` holds:
- `population.jsonl`: per checkpoint, the candidates with Elo, parent, birth generation, σ, and
  6 charted gene weights;
- `champions.jsonl`: the fixed-panel `vs` results the Population tab charts, plus each
  champion's genome and ancestor;
- `generations.jsonl`: fitness, σ and wall time for every generation;
- `meta.json`: the anchors.

## Runs

Hardware: the shared 16-core Mac at load 25–100, 3 processes (driver plus 2 pool workers),
`OMP_NUM_THREADS=1`.

| run | islands × members | table | opponents | gens | games | wall |
|---|---|---|---|---|---|---|
| `evox_r1` | 120 × 6 (720) | 3 members + 1 opponent | basic / int / HoF, per seat | 205 (stopped) | 49k | 0.56 h |
| `evox_r2` | 120 × 4 (480) | 2 members + 2 copies of one opponent | basic 40% / int 40% / HoF 20% | 1339 (last checkpoint 1299) | 321k | 2.3 h |

Both runs start from random z ~ N(0, 1), σ0 = 0.3. r2 also sets σ_min = 0.05 and checkpoints
every 50 generations with 4 candidates × 100 selection games each.

### evox_r1: peer-heavy tables


- **Rise.** From gen 0 to gen 79 the champion went 13.5% → 45.0% [38.3, 51.9] vs 3×basic and
  12.0% → 24.0% vs 3×intermediate (n = 200 each). Elo went 1651 → 1766, against anchors
  basic 1620 and intermediate 1799.
- **Drift.** vs-basic then fell to 30–36% by gen 199, while fitness inside the population kept
  creeping up. That is D17's transfer problem again: with 3 of 4 seats held by population
  members, fitness mostly measures play against near-copies.
- **σ collapse.** σ fell from 0.30 to 0.13 by gen 200.
- r1 was stopped and r2 changed the tables.

### evox_r2: same-opponent tables (the model)


Champion of each checkpoint, report block 60, n = 200 per opponent (95% Wilson CI). vs 3×random
was 100% from gen 0.

| gen | Elo | vs 3×basic | vs 3×intermediate |
|---|---|---|---|
| 0 | 1602 | 15.5% [11.1, 21.2] | 7.5% [4.6, 12.0] |
| 49 | 1708 | 21.0% [15.9, 27.2] | 20.0% [15.0, 26.1] |
| 99 | 1747 | 36.0% [29.7, 42.9] | 25.0% [19.5, 31.4] |
| 249 | 1799 | 35.0% [28.7, 41.8] | 33.0% [26.9, 39.8] |
| 499 | 1798 | 41.5% [34.9, 48.4] | 31.0% [25.0, 37.7] |
| 699 | 1791 | 40.5% [33.9, 47.4] | 31.5% [25.5, 38.2] |
| 999 | 1781 | 41.0% [34.4, 47.9] | 27.0% [21.3, 33.5] |
| 1099 | 1819 | 48.5% [41.7, 55.4] | 33.0% [26.9, 39.8] |
| 1249 | 1820 | 50.0% | 32.0% |
| 1299 | 1746 | 36.0% [29.7, 42.9] | 27.0% [21.3, 33.5] |

Anchors: basic 1620, intermediate 1799 (random = 1000).

- **Shape of the curve.** Fixed-panel strength rises steeply up to gen 250, then holds at about
  intermediate level (Elo 1780–1820) through gen 1300.
- **Training fitness.** It keeps creeping up, with population mean 0.85 → 0.95 between
  gens 250 and 1300. Its opponents include the strengthening hall of fame.
- **Stopping.** The lead stopped the run at gen 1339 because the machine was oversubscribed
  (load 68–100) and the plateau was clear.

### Final model and the success test

`models/evox_best.json` (and `.pt`, the same dict for the zoo registry) is r2's gen-1249
champion `e164972`, picked from the last 10 checkpoint champions on fresh block-56 seeds. It
scored 44.5% vs 3×basic and 35.5% vs 3×intermediate there. Its generation-0 ancestor is
`e000196`, a random genome.

Paired test, 600 games per table on blocks 60 (offsets 1000–1074) and 61 (0–74), 1v3, 4 seats:

| genome | vs 3×random | vs 3×basic | vs 3×intermediate |
|---|---|---|---|
| **evolved (e164972)** | 100.0% [99.4, 100] | **43.2% [39.3, 47.2]** | **33.3% [29.7, 37.2]** |
| gen-0 ancestor (e000196) | 99.2% [98.1, 99.6] | 1.3% [0.7, 2.6] | 2.2% [1.3, 3.7] |
| zero genome | 0.0% | 0.0% | 0.0% |
| intermediate genome (= `intermediate`) | 100.0% | 37.0% [33.2, 40.9] | 25.0% [21.7, 28.6] |

| paired comparison | vs 3×random | vs 3×basic | vs 3×intermediate |
|---|---|---|---|
| evolved − ancestor | +0.8 pp, 5/0, p = 0.062 | **+41.8 pp, 253/2, p = 1e-72** | **+31.2 pp, 196/9, p = 6e-47** |
| evolved − intermediate | 0, 0/0 | +6.2 pp, 134/97, p = 0.018 | **+8.3 pp, 129/79, p = 0.00064** |

(x/y = games only the first genome won / only the second won.)

- **Success criteria.** All three are met:
  - the fixed-panel curve rises clearly;
  - the evolved model beats its generation-0 ancestor at p < 0.01 in two 1v3 tables with
    n = 600 and is not worse in the third;
  - population Elo is measured against fixed anchors.
- **Against the hand-tuned bot.** The evolved weights also beat the hand-tuned `intermediate`
  weights. The gain is +8.3 pp vs 3×intermediate (p = 0.0006) and +6.2 pp vs 3×basic
  (p = 0.018). It is the first evolution result in this project that beats a hand-tuned
  baseline on held-out tables.
- **Zero genome.** It wins nothing even against random. I have not looked into why. The likely
  cause is that every score ties, so every choice falls back to the first listed action.

**What evolution found** (weight = z·scale; intermediate's value in brackets):
- **Dev cards.** dev_ev +4.13 (0.42), dev_held −0.55 (+0.30). It buys a dev card whenever it
  can afford one and plays them out.
- **Knights.** knight_bonus −2.25 (+0.4). It does not play knights in MAIN unless needed; the
  pre-roll knight rule is unchanged.
- **Opening.** setup_pips +2.29 (1.0), setup_comb_div +2.68 (1.0), setup_wb −1.78 (+1.0),
  setup_port3 −2.34 (+1.0). The opening is driven by production and diversity, and avoids
  3:1 ports.
- **Robber.** rob_victims +3.60 (2.0): it prefers hexes where it can steal.
- **Trades.** accept_margin +0.20 (0.04). It accepts far fewer domestic trades.
- **Production.** prod is +0.011 (0.045) and diversity +0.38 (0.12): it values producing many
  resource types more than the raw pip count.

## Gate (information only)

Command: `python -m scripts.promote --agent evox --checkpoint models/evox_best.pt --name evox_r2
--deterministic --workers 2`. The entry is `evox_r2-v1`, run on gate blocks 40–42. None of
those blocks was used for selection.

| test | evox_r2-v1 | champion jointb_gated-v1 |
|---|---|---|
| vs 3×basic, n = 200 | 42.5% [35.9, 49.4] | 70.5% |
| vs 3×intermediate, n = 200 | 36.5% [30.1, 43.4] | 76.5% |
| mixed table vs champion, n = 600 | 11.2% [8.9, 13.9] | |

Not promoted, as expected. The evolved bot uses no search, no value net and no trade model, and
the champion is value-net search with a learned trade head. On the gate seeds its 1v3 numbers
match the blocks 60/61 test: above intermediate, below every search agent.

## Interpretation and next steps

**Why this worked when D3, D16 and D17 did not.**
- **Room to improve.** Starting from random weights leaves a lot to gain. D17 started at the
  champion, on a flat plateau.
- **Cheap evaluation.** Evaluation is cheap enough for 320k games and 1339 generations in 2.3 h.
- **Selection.** Lifetime-averaged, shrunk fitness under common random numbers lets selection
  work with 1 game per member per generation.
- **The fix that mattered most (r1 vs r2).** Fitness tables must look like the target tables:
  1–2 members against copies of one fixed or HoF opponent. Tables of mostly peers drifted
  (r1).

**Plateau.** Fixed-panel strength stopped rising around gen 250, at slightly above
intermediate. Three explanations, not separated yet:
1. The genome can't express much more than this.
2. Selection noise: 1 game per member per generation and 4 members per island.
3. The fitness target: wins + VP/20 against a mix that is 40% basic.

**Next.**
- r3 with the 38-gene genome, warm-started from `e164972` (`--genome ext --init intermediate
  --init-from models/evox_best.json`), with 2 games per member per generation. This tests (1)
  and (2) together.
- Per-island fitness variants (vs-basic-only and vs-intermediate-only islands, with
  migration), a step toward a MAP-Elites style archive.
- The evolved weights could also seed value_trade's heuristic leaf (`leaf_mix`) or the trader
  layer.
