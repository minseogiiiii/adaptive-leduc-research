# Adaptive decision-making in Leduc poker

Week 1 game engine, Week 2 CFR baseline, and Weeks 3-4 Bayesian opponent
model for a research project on learning from legally observable actions. The
repository has a complete two-player fixed-limit Leduc hand engine, full-tree
CFR, exact strategy evaluation, and an exact information-set response to a
mixture of opponent types.

## Try one hand

From this directory, with Python 3.10 or newer:

```bash
python3 -m pokerlab --seed 7
python3 -m unittest discover -s tests -v
python3 -m pokerlab.cfr --iterations 100 --output strategy.json
python3 -m pokerlab.bayes --hands 12 --opponent calling --seed 7
python3 -m pokerlab.experiment --opponent calling --hands 12 --replicates 20 --seed 7 --output results/calling.json
python3 -m pokerlab.experiment --opponent calling --switch-to aggressive --switch-after 6 --hands 12 --replicates 20 --seed 7 --output results/switch.json
python3 -m pokerlab.profile --opponent calling --switch-to aggressive --switch-after 2 --hands 4 --replicates 2 --seed 13 --output results/profile-switch-example.json
python3 -m pokerlab.report --output REPORT.md
python3 -m pokerlab.followup --output FOLLOWUP.md
python3 -m pokerlab.validate --player0 baseline --player1 calling --hands 20000 --seed 7
```

The included `strategy.json` was trained for 300 iterations with the same
deterministic full-tree algorithm. To regenerate it exactly, use
`python3 -m pokerlab.cfr --iterations 300 --output strategy.json`.

The trace is a *post-hand evaluator view*: it prints both private cards so that
you can inspect the outcome. Agents only receive `PlayerObservation` objects,
which exclude the opponent's card until showdown.

## What to read first

1. `rules.md` gives the exact variant, two complete sample hands, and the
   observation contract.
2. `pokerlab/game.py` implements the state transitions. `GameState` is an
   immutable full simulator state; `observe(player)` is the agent interface.
3. `pokerlab/simulate.py` runs random agents with independent action and chance
   random-number streams.
4. `tests/test_game.py` checks terminal payoffs, legal actions, chance cards,
   information leakage, and complete reachable game paths.
5. `pokerlab/cfr.py` learns an average CFR strategy and evaluates exact game
   value, best responses, and exploitability. `tests/test_cfr.py` checks the
   information partition, payoff evaluator, convergence, and saved policy.
6. `pokerlab/bayes.py` specifies four example opponent types, updates a
   posterior from terminal observations, and computes an adaptive response.
   `tests/test_bayes.py` checks hidden-card marginalization, showdown reveal,
   repeated updates, and mixture decision value against independent profile
   evaluation.
7. `pokerlab/experiment.py` compares the updating agent with a frozen-prior
   mixture response and CFR, using identical hand seeds for the three arms.
   It supports one prespecified opponent-policy switch within a match.
8. `pokerlab/validate.py` compares the exact profile evaluator with independent
   seeded hand rollouts. Its tests include a known one-chip fold payoff and an
   asymmetric stochastic profile.
9. `pokerlab/profile.py` measures where the paired experiment spends CPU time
   and hashes its deterministic output. `tests/test_profile.py` compares that
   hash with a separately run, unprofiled switch experiment.
10. `REPORT.md` is a descriptive pilot with three prespecified opponent
    conditions. `pokerlab/report.py` reconstructs its means, paired contrasts
    and bootstrap intervals from saved hand rows and checks input hashes.
11. `FOLLOWUP_PROTOCOL.md` fixes the held-out family analysis before its
    outcomes. `FOLLOWUP.md` records the larger run; `pokerlab/followup.py`
    validates and reproduces its joint-replicate bootstrap summary.
12. `pokerlab/open_spiel_check.py` exhaustively compares the hand engine with
    OpenSpiel 2.0.2. `results/open-spiel-crosscheck.json` records the audit.

## Evaluator cross-check

`python3 -m pokerlab.validate --player0 baseline --player1 calling --hands
20000 --seed 7` calculates the exact expected net chips for player 0 and
estimates the same quantity by running 20,000 complete hands through the hand
simulator. It reports the Monte Carlo standard error and exits with an error if
the means differ by more than four estimated standard errors. A larger sample
can resolve an occasional noisy failure; a passing check does not establish
mathematical correctness. The two paths use different chance/action evaluation
methods but share `pokerlab/game.py`.

## External Leduc rules audit

With OpenSpiel 2.0.2 installed in a compatible Python environment (verified
here using Python 3.12), run:

```bash
python3 -m pip install open-spiel==2.0.2
python3 -m pokerlab.open_spiel_check --output results/open-spiel-crosscheck.json
python3 -m unittest discover -s tests -v
```

This optional audit loads
`leduc_poker(players=2,action_mapping=false,suit_isomorphism=false,starting_player=0)`.
OpenSpiel's ante is 1, the round increments are 2 and 4, and its two raises
per round count the opening bet. Its `call` action means check when no bet is
outstanding, and `raise` means the opening bet in that case. The audit checks
every ordered pair of physical private cards and every legal continuation,
including public chance probabilities, legal actions, actor, betting history,
pot, contributions and net terminal returns. It traversed 30 private deals,
3,780 decision nodes, 150 public chance nodes and 5,520 terminal nodes with
no mismatch. The inspected upstream source is
[`leduc_poker.cc`](https://github.com/google-deepmind/open_spiel/blob/48401890ee9857e611678302371378175a8e4c6b/open_spiel/games/leduc_poker/leduc_poker.cc)
and its [header](https://github.com/google-deepmind/open_spiel/blob/48401890ee9857e611678302371378175a8e4c6b/open_spiel/games/leduc_poker/leduc_poker.h);
the executable comparison uses the pinned 2.0.2 distribution.

This establishes agreement of the **hand transitions and payoffs** under
those parameters. OpenSpiel with `suit_isomorphism=false` exposes the physical
copy in its information-state string, whereas our CFR `info_key` merges the
two copies of a rank. Thus OpenSpiel's unmodified CFR policy and exploitability
numbers cannot be compared directly to ours. The project also still needs an
independent check of its Bayesian likelihood and information-set response.
The earlier `REPORT.md` and `FOLLOWUP.md` are historical snapshots written
before this external audit and retain their original provenance text.

## Stationary-opponent experiment

`python3 -m pokerlab.experiment --opponent calling --hands 12 --replicates 20
--seed 7 --output results/calling.json` writes a deterministic JSON record.
For a prespecified opponent outside the candidate set, replace `calling` with
`rank_selective`: its betting, calling and folding biases depend on its private
rank and whether it has paired the public card. Its factors are illustrative
assumptions; no hand used for evaluation is used to tune them.
Each independent replicate contains one match in each seat. Within each match,
the opponent policy is fixed and the learning agent starts from a fresh uniform
prior. All three arms share the same dealt cards and per-seat action random
streams for a given hand; their public histories may diverge as they choose
different actions. The frozen-prior arm computes the same initial exact
mixture response as the learner, but never changes it across hands. The CFR
arm plays the loaded average policy.

The output records each hand's seed, seat, reward, and learner posterior,
seat-averaged reward per replicate, and paired learning-minus-control
differences with 95% percentile bootstrap intervals over entire two-seat
replicates. It also records hashes of `strategy.json` and `rules.md` to tie
the run to its inputs. Bootstrap intervals with few replicates are unstable;
20 replicates and 12 hands above illustrate the workflow, not a definitive
power calculation or a claim that learning helps. The first four opponent
choices are also the model's own candidates; `rank_selective` is a single
held-out behavioral rule. A broader set of held-out types and larger runs
remain to be evaluated before drawing general conclusions.

## Opponent-policy switch experiment

`python3 -m pokerlab.experiment --opponent calling --switch-to aggressive
--switch-after 6 --hands 12 --replicates 20 --seed 7 --output
results/switch.json` uses `calling` for hands 0–5 and `aggressive` for hands
6–11, for both seats and every replicate. The policy changes **before** the
hand at zero-based index 6. Both phases must contain at least one hand.
`rank_selective` is also available on either side of the switch. The same
predeclared schedule applies to all three arms; each arm shares a hand seed.
The learner is not told the schedule and does not reset its posterior at the
switch. Thus this tests the existing stationary-type Bayesian learner under
a violated assumption; it does not implement a change-point detector.

The JSON includes the opponent names and switch point, per-hand phase and
posteriors before and after each hand, and separate phase reward summaries.
`change_in_paired_contrasts` subtracts the pre-switch learning-minus-control
contrast from the post-switch contrast **within each two-seat replicate**,
then reports its mean and a 95% percentile bootstrap interval over replicates.
Phase means and intervals use the same replicate unit. This change measures
a shift in relative payoff under a changed opponent, not a causal estimate of
learning speed. The sample command is illustrative; assess uncertainty with
many independent replicates and prespecified phase lengths and opponents.

## CPU profiling

`python3 -m pokerlab.profile --opponent calling --switch-to aggressive
--switch-after 2 --hands 4 --replicates 2 --seed 13 --top 15 --output
results/profile-switch-example.json` runs the full paired experiment under Python's
`cProfile`. It records the workload and input SHA-256 hashes, Python/platform
details, total calls, elapsed seconds, and the functions with the largest
**cumulative** time. Nested cumulative times overlap and must not be added.
Strategy loading and JSON serialization happen outside the timed region. The
`result_sha256` hashes the entire deterministic `run_experiment` output (all
hands and posteriors), before profile metadata is added. It can be compared
with an unprofiled run using the same inputs; the test suite checks this for
a switched opponent. Timing depends on hardware, Python and profiler overhead.

The small example diagnoses implementation cost, not the effectiveness of
learning or a statistically stable speedup. For comparable measurements,
hold the input hashes, opponent schedule, Python version, machine and system
load fixed, run multiple independent timings, and inspect the same call sites.
On the example workload the nested exact mixture best response dominates
cumulative time. This identifies a candidate for future optimization, while
the present profiler does not alter its information-set semantics. The
committed `results/profile-switch-example.json` records one such diagnostic
run, including its exact input hashes and the full experiment-result digest.

## Research pilot report

`REPORT.md` compares a candidate calling type, one held-out rank-selective
policy, and a calling-to-aggressive switch. Each condition has ten independent
two-seat replicates with twelve hands per seat, using seed 7. Its tables are
generated from the committed full records in `results/` using
`python3 -m pokerlab.report --output REPORT.md`. The builder checks coverage of
every replicate/seat/hand, input hashes, phase assignment, posterior validity,
and point estimates and bootstrap intervals against raw hand rewards. It
rejects a different workload instead of silently mixing protocols.

This is a **pilot**, not evidence of robust improvement: intervals from ten
replicates are imprecise, the held-out behavior is only one constructed rule,
and the switch violates the learner's stationary-type assumption. The
report includes reproduction commands, JSON hashes and remaining external
validation and power-analysis work.

The pilot report preserves hashes of the implementation used to generate its
original outcomes. Later additions to `pokerlab/experiment.py` do not change
its saved hand rows or statistics, but regenerating the pilot Markdown under
the expanded code changes the provenance hash printed for that file. Use the
original Git revision when reproducing the historical report byte for byte.

## Held-out opponent follow-up

`FOLLOWUP_PROTOCOL.md` fixes the three constructed held-out policies, the
calling-to-aggressive switch stress condition, seed, sample size, primary
family contrast, bootstrap unit and stopping rule. The first policy depends
on private rank and public pair, `round_polarized` changes bias across betting
rounds, and `pressure_reactive` backs off after public opposing pressure.
They all receive only legal `PlayerObservation` information and assign
positive probability to every legal action. The learner's four candidate
types remain the same.

`FOLLOWUP.md` is generated from four full raw JSON records with
`python3 -m pokerlab.followup --output FOLLOWUP.md`. The generator checks the
complete hand grid, provenance, phase labels, posteriors, reward summaries,
intervals and shared seed schedule. The family interval jointly resamples
two-seat replicate indices across the three held-out conditions. The separate
switch condition probes the stationary-type assumption. These constructed
opponents and the finite number of replicates limit generalization.

## Bayesian opponent learning and adaptive decisions

The *example* candidate types are a smoothed CFR baseline and three fixed
action-bias variants: folding, calling, and aggressive. Their multipliers and
5% action smoothing are **modeling assumptions**, not fitted poker statistics.
Every type assigns positive probability to every legal action. The learner
assumes a fixed type across observed hands, even in the policy-switch test.

For a completed hand, `BayesianOpponentModel` receives only
`terminal_state.observe(learner_seat)`. For each candidate type it enumerates
opponent private cards compatible with our card and any public card, replays
the observed public actions, and multiplies the type's probabilities of the
**opponent's** actions. Our own action probabilities cancel between types.
At showdown the revealed opponent card restricts the sum to that card;
after a fold all compatible opponent cards remain in the sum. Chance weights
are `1/5` for the opponent card and, if revealed, `1/4` for the public card
conditional on both private cards. The posterior is updated in log space
to avoid underflow across many hands. An impossible observation raises an
error instead of silently inventing a posterior.

Before each hand, `AdaptiveAgent` computes an exact information-set best
response to a **whole-hand mixture**: an opponent type is sampled at the start
of a hand and remains fixed during that hand. This automatically accounts
for information from opponent actions within the hand. At each decision,
one action must be chosen for all hidden cards and types consistent with the
learner's observation. `choose()` receives only a `PlayerObservation`;
`finish_hand()` receives the learner's terminal observation and updates the
posterior before the next hand. Exact tree search is practical in this small
game, but should not be treated as a scalable Hold'em implementation.

The demo prints each hand's net payoff and type posterior. Its sample mean
over 12 hands is **not** a controlled comparison or evidence that learning
beats CFR. In particular, optimizing against the **initial** mixture can
change actions even before any learning occurs. To isolate the effect of
learning, the experiment compares the updating agent with the same mixture
response whose prior remains frozen, as well as with fixed CFR. It includes
a prespecified held-out opponent, seat swaps, repeated independent matches,
and uncertainty estimates.

## What the CFR numbers mean

The command prints the expected **net chips per hand for player 0** when both
players use the learned average strategy. Player 0 always acts first in this
variant, so even a symmetric strategy need not have zero value. For each player,
the best response is a strategy that conditions on **that player's card and
public history**, never the other player's hidden card. `NashConv` is the sum
of each player's improvement by unilaterally switching to a best response;
the printed `exploitability` is `NashConv / 2` in chips per hand. It approaches
zero for a Nash equilibrium. It is **not** the strategy's payoff against an
arbitrary biased opponent.

The evaluator enumerates all `6 × 5 × 4 = 120` ordered draws of two private
and one eventual public physical card, each with probability `1/120`. A future
public card is predetermined by the evaluator, but hidden from CFR and the
policy until the round ends. If the hand ends by fold, all four possible future
public draws are still averaged; this reproduces the correct marginal chance
of the private cards. CFR uses chance-weighted counterfactual opponent reach
for regrets and chance-weighted own reach for the average policy. It freezes
the current strategy throughout each iteration before applying regret updates.

Use `TabularPolicy.load("strategy.json")` to load the baseline. To play seeded
hands, pass `PolicyAgent(loaded)` to `run_hand`; the agent interface receives
only `PlayerObservation`. Keep the saved policy and this exact `rules.md`
version together: another Leduc betting convention changes the game.

For the included 300-iteration policy, exact evaluation prints 288 information
sets, value for player 0 `-0.088324`, and exploitability `0.081756` chips per
hand (six decimal places). These are baseline diagnostics, **not** evidence
that opponent adaptation improves payoff. CFR convergence is asymptotic; 300
iterations do not make the policy an exact equilibrium.

## Roadmap and current status

| Stage | Intended output | Status |
| --- | --- | --- |
| Week 1 | Rules, game engine, observations, seeded traces, correctness checks | Implemented; exhaustive OpenSpiel 2.0.2 hand-rule audit passed |
| Week 2 | CFR, exact values and legal best response | Implemented; rollout check and external hand-rule audit passed; information-set response remains independently unverified |
| Weeks 3-4 | Hidden-card likelihood and Bayesian decision agent | Implemented for a finite stationary model set |
| Weeks 5-6 | Stationary opponent experiments | Paired, seat-balanced runner and a prespecified three-type held-out follow-up implemented; broader generalization study pending |
| Weeks 7-8 | Policy switches, profiling and research report | Switch experiment, reproducible CPU profiler, pilot and follow-up reports implemented; external hand-rule audit passed |

The reference convention in the accompanying PDF must be matched explicitly
before any numeric comparison with that PDF; the OpenSpiel audit above checks
the specified OpenSpiel variant. The small controlled pilot does not establish
a general adaptation benefit or strong poker play.
