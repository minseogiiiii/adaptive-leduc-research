# QT V5: Precommitted Evaluation Protocol

## Motivation

V4 fixed hazard 0.1 improved exact conditional EV after a calling->aggressive
switch yet worsened it against a stationary calling opponent. Avoid choosing
hazards after looking at a held-out evaluation set.

## Unchanged rules and baselines

Frozen Leduc v0.1, strategy.json, historical V4 experiments unchanged.

Reference beliefs:
- Original Bayes, stationary type assumption (hazard 0).
- V4 switching type filter with **fixed hazard 0.1**.
- V5 Bayesian model average: H0 hazard 0, H1 hazard 0.2, prior P(H1)=0.25.
- V5 gated single-hand worst-case payoff budget:
  epsilon = 0.02 + 0.08 x type_concentration x (1-P(H1)),
  where type_concentration = 1 - posterior_entropy / log(4).
- Existing V4 exact-safe uses fixed epsilon=0.1.
None of these constants may be selected on confirmation results.

## Stage A (inference diagnostic, common evidence)

All models observe **identical historical hands**, from a frozen CFR probing
player facing the test opponent. Every hand is generated independently,
but Bayesian state persists within each match. The hidden opponent card is
not fed to the inference engine except at legally exposed showdown.

Predeclared six scenarios:
1. stationary_calling
2. stationary_aggressive
3. switch_calling_aggressive, at 50% of match
4. switch_aggressive_calling, at 50%
5. heldout_rank_selective
6. heldout_pressure_reactive

Pilot (development-only): 12 replicates x 24 hands x 2 seats, seed 20261014.
Confirmation: 36 independent replicates x 24 hands x 2 seats, seed 20261017.

Primary contrast per independent replicate (two-seat match mean):
V5 evidence model minus V4 fixed-hazard model in **predictive log-likelihood
per observed hand**, in nats/hand. Larger is better. Paired cluster bootstrap
(2000 draws) over replicate units, not individual hands.

Co-primary cases: stationary_calling and switch_calling_aggressive.
Secondary: other four cases and post-switch-half contrasts, clearly labeled
exploratory. No hypothesis selection from the results.

**Adoption gate (inference)**:
- For BOTH co-primary scenarios, improvement >=0.005 nats/hand AND
  two-sided cluster-bootstrap 97.5% intervals exclude zero in positive
  direction (Bonferroni for two co-primary claims).
- Held-out inference must not suffer a degradation over 0.01 nats/hand
  on either scenario (effect-size guardrail, descriptive only).
- Positive detection scores alone do NOT prove decision improvement.

## Stage B (actual game decision and risk utility)

On-policy benchmark against V4 with paired card seeds and seat balancing:
- Original Bayesian, V4 switching hazard0.1, V5 evidence, V4 exact-safe
  epsilon0.1, V5 gated exact-safe epsilon0.02..0.1.
- The safety bound is EXACT *single-hand worst-case expected payoff*;
  it is not a drawdown or CVaR guarantee.
- Evaluate realized chips and oracle **conditional expected chips per hand**
  using the simulator's true opponent only in a separate offline evaluator.
- Primary decision comparison: V5 evidence minus V4 switching. Secondary
  comparisons: V5 gated-safe vs fixed-safe, and versus original Bayesian.
- No same-seed selection of risk parameters. Bootstrap replicate units, and
  report runtime. A positive mean of <0.02 chips/hand is not considered
  practically meaningful by itself.

## Limitations and stopping rule

- Artificial opponent policies, frozen small Leduc game; no demonstration
  of market alpha, human opponent robustness, or live QT performance.
- A flat or negative forecast-stage result fails the V5 evidence gate,
  regardless of positive policy profit measured on a few sample hands.
- Standard bootstrap intervals from ~12 pilot replicates are NOT sufficient
  evidence of statistical significance. Confirmation uses 36 replicates but
  is still conditional on these scenarios.
- If Stage A fails, report the failure honestly and recommend reducing
  complexity or using V4 fixed models. Do not continue searching parameters
  until a nominally significant result is found.
