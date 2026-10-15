# QT V7 — Precommitted long-horizon value-of-information experiment

**Study question:** Does sacrificing immediate expected chips to play *more
informative* hands improve the return from opponent adaptation over the next
24–72 hands? In V6 A3 the aggressive probe increased opponent-type
classification accuracy but its risk-constrained prehand frequency was only
~6.61%. This study asks whether later adaptive profit repays probe costs.

## Immutable starting point and controls

- Unchanged original six-card Leduc rules, strategy.json, four candidate
  opponent models, exact information-set Bayesian best-response solver,
  and V4 switching beliefs with hazard=.1.
- Probe is V6 aggressive-biased ActionBiasPolicy(CFR), and the safe probe
  selection probability is computed once from the **exact one-hand**
  CFR-vs-probe adversarial floor epsilon=.05, restricted to the smaller
  of the seat-specific safe weights.
- Probe event is shared by the arms per replicate, seat, hand, and is
  independent of the chance card seed.
- Policy arms:
  1. No Probe: exact BR to learned switching posterior, on every hand.
  2. Matched CFR Control: at the shared random probe event choose CFR,
     otherwise choose exact BR to the same learner's own switching posterior.
  3. Active Probe: at the shared event choose aggressive information-probe
     policy, otherwise choose exact BR to its own switching posterior.
- All agents update Bayesian beliefs after each observed TERMINAL hand,
  strictly from their own player-visible observation. Each arm plays its own
  full sequence, not an oracle replay of another arm's observations.
- Data generator opponent policies: stationary Calling,
  Calling->Aggressive switch at 50%, and Rank-Selective held-out.
- The oracle exact conditional payoff evaluator receives the true synthetic
  opponent policy ONLY after the agent precommits to its policy, and this
  information is never fed into the policy or Bayesian learner.
- The A3 *standalone CFR vs probe lottery* worst-case expected single-hand
  guarantee does **NOT** guarantee the larger BR/active agent, multi-hand
  bankroll return, drawdown or ruin. Report this prominently.

## Hypotheses and goals

H0: At low 6.61% probe rate, Active minus Matched CFR Control does not yield
more than +0.02 chips/hand net of probe costs over the whole horizon.
H1: Long horizon Active minus Matched CFR Control >= +0.02 chips/hand and
the paired 95% bootstrap CI lower endpoint >0. This is the
**precommitted practical adoption threshold**.

We separately compare Active minus No Probe (actual deployment hurdle)
and the Matched CFR Control minus No Probe (cost of exploration *timing*).
Isolate immediate probe-hand cost vs non-probe future gains and
split pre/post regime switch.

Important statistical rule: report paired bootstrap over INDEPENDENT
replicate-block averages, with both seats combined; do NOT treat each
serially dependent hand as an independent replicate. No p-hacking; no
parameter tuning after seeing held-out runs.

Pilot (development/feasibility): seed 20261026, 4 independent blocks,
24 hands per seat, two scenarios stationary/switch. Pilot does not provide
definitive statistical evidence.

**New-seed confirmation** (only if pilot code works): seed 20261028,
12 independent blocks, 48 hands per seat, same predeclared three scenarios.
Also examine horizon sensitivity (short 24 vs long 48), but report
the long 48-hand comparison as primary. If lack of power makes CIs
inconclusive, report "unresolved," not success.

Note a large one-hand exact BR computation is expensive and performance
is an algorithmic cost, not an assumed realistic latency. Failed job /
timeout is not evidence of strategy performance.

## Scientific limits

- All opponent policies in these tests are scripted and stationary or
  exogenously switching; genuinely adaptive strategic adversaries are not
  established.
- Even a positive expected result under this synthetic game is not
  evidence of actual market alpha.
- A probe can change opponents' *observable actions* and thus learner
  posterior; otherwise identical deck seeds only reduce sampling noise,
  they do not enforce identical observations.
- These tests estimate matched average policy value, not exact EVPI, and
  not a fully optimized belief-state POMDP solution.
