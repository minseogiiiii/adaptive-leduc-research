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

## Evaluator cross-check

`python3 -m pokerlab.validate --player0 baseline --player1 calling --hands
20000 --seed 7` calculates the exact expected net chips for player 0 and
estimates the same quantity by running 20,000 complete hands through the hand
simulator. It reports the Monte Carlo standard error and exits with an error if
the means differ by more than four estimated standard errors. A larger sample
can resolve an occasional noisy failure; a passing check does not establish
mathematical correctness. The two paths use different chance/action evaluation
methods but share `pokerlab/game.py`. Comparison with a separately implemented
game remains outstanding, and must first reconcile its betting convention
with `rules.md`.

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
| Week 1 | Rules, game engine, observations, seeded traces, correctness checks | Implemented here; OpenSpiel cross-check remains |
| Week 2 | CFR, exact values and legal best response | Implemented; simulation cross-check added, external rules cross-check pending |
| Weeks 3-4 | Hidden-card likelihood and Bayesian decision agent | Implemented for a finite stationary model set |
| Weeks 5-6 | Stationary opponent experiments | Paired, seat-balanced runner and one held-out behavioral opponent implemented; large runs pending |
| Weeks 7-8 | Policy switches, profiling and research report | Switch experiment implemented; profiling and report pending |

The reference convention in the accompanying PDF must be matched explicitly
before any numeric comparison with OpenSpiel. This package has no controlled
adaptation results or claim of strong poker play yet.
