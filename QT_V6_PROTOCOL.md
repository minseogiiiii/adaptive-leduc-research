# QT V6 — Locked two-track research protocol

**Scope:** Both studies use the ORIGINAL frozen Leduc game, CFR strategy, and
type likelihoods. Previous V1-V5 research artifacts are not modified.

## Research A — Changepoint posterior

A0 (mathematical negative control): BOCPD with a **constant hazard of 0.1**
must give EXACTLY the same marginal type posterior and predictive likelihood
as V4's SwitchingBeliefs(0.1). No alpha gain is possible merely by retaining
run-length states under identical assumptions.

A1 (testable hypothesis): If opponent strategy changes have a
**duration-dependent** rate, modeling run age can outperform constant hazard
in predictive log score AND the posterior probability of a recent change
can reveal the switch without large false-alarm probabilities.

Predeclared duration hazard: P(switch before next hand | age<6)=0.03;
P(switch|age>=6)=0.20. This is an assumption, NOT learned from test data.

Scenarios: stationary Calling, stationary Aggressive, Calling->Aggressive
at hand halfway, Aggressive->Calling at halfway, earlier Calling->Aggressive
after first quarter, heldout Rank Selective, heldout Pressure Reactive.

Pilot: seed 20261018, 12 replicate blocks x 2 seats x 24 hands.
Confirmation: independent seed 20261020, 48 replicate blocks x 2 seats x 24
hands. Each replicate's pooled two-seat mean is the bootstrap unit.
Prequential log score measured before updating with the observed hand.
Offline truth labels only score change point detection AFTER each hand
and are never given to an agent.

Co-primary comparisons: stationary Calling and Calling->Aggressive.
Bonferroni compatible 97.5% two-sided paired bootstrap CIs on two
co-primary claims; require both lower bounds >0 AND each mean
>+0.005 nats/hand. Heldout loss not worse than -0.010 nats/hand
as a descriptive guardrail.
If fails, do not reinterpret a favorable scenario as universal success.
The posterior change-point probability may still constitute useful
diagnostic information even if predictive superiority fails.

## Research B — Exact risk/reward frontier

B0: Confirm CFR's EXACT worst-case one-hand EXPECTED payoff floor and a
policy mixture drawn secretly ONCE at each new hand.

B1: Optimize a **one-dimensional family only**:
mix(w) = (1-w)*frozen-CFR + w*BR-to-declared-posterior, w in [0,1].
For each epsilon in [0, .02, .05, .1, .2, .4, .8, 2.0], find the
largest feasible w s.t. min_opp E[reward(mix(w),opp)] >=
Vmin(CFR)-epsilon. Exact opponent best response evaluated on hidden
whole-hand mixture, NOT independent action probabilities.
Control: compare with earlier conservative worst-value inequality bound.
Because posterior EV is affine and BR weakly improves posterior EV, the
largest feasible w is optimal *within this limited CFR/BR mixture family*,
not globally across all poker strategies.

Primary descriptive outputs: exact worst-case single-hand payoff,
posterior expected payoff, opportunity cost versus full BR,
exact-minus-conservative improvement, held-out Rank Selective value,
risk budget actually spent. Check monotone Pareto frontier and constraint
at every point; no p-value is meaningful for deterministic exact values.
Analyze risk shadow prices (slope) but do not claim Kelly/CVaR guarantees.

Predeclared contexts: two seats x two opponent-type priors:
uniform (4 types, 0.25 each), and Calling-concentrated (Calling .85,
other 3 types .05 each). One common strategy.json reference.
**No regime fitted on validation output.**

Hypotheses:
B1: exact worst-risk constrained mixture gains EV vs conservative bound
at the same epsilon without ever violating floor.
B2: expected reward gained per added worst-case chip of permitted risk
is nonconstant; held-out payoff may weaken with larger exploitation.
Failure to show gains on heldout is an honest risk price, not a
technical test failure.

Economic meaning: metric is chips/hand in synthetic Leduc, not a
financial return, realized drawdown protection, bankroll VaR or
significance about real-world opponents.

## Iteration policy

1. Run pilot. Distinguish mathematical correctness from empirical success.
2. Run independent confirmation only once with locked parameters.
3. On A failure, perform *diagnostic decomposition* (stationarity, first/
   second half, change posterior true-vs-false) without retuning A1 on
   confirmation data.
4. A2 may be designed based on failure mechanism, but it requires new
   unexposed seeds and independent scenarios. Keep A1 failure public.
5. B exact frontier needs no bootstrap CIs because its values come from
   deterministic enumeration; any sampled terminal rewards must be evaluated
   and reported separately.
