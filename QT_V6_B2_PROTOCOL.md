# B2 — Locked richer-strategy safety frontier hypothesis

B1 used a ONE-dimensional hidden hand-level lottery: CFR vs target-type BR.
This is NOT globally optimal across all policies.

B2 adds another pure precomputed policy:
- pi_0: fixed baseline CFR
- pi_1: exact BR to calling-concentrated belief (.85 Calling, other types .05)
- pi_2: exact BR to Aggressive opponent type (full-support synthetic).

A policy draw is made *once per whole hand* and not disclosed. For every
declared phi in [0,.2,.4,.6,.8,1], consider a lottery:
P(pi_0)=1-w, P(pi_1)=w*phi, P(pi_2)=w*(1-phi).

For each phi, use exact adversarial best response over ALL legal opponent
strategies to compute maximum feasible w such that worst EV >=
worstEV(CFR) - epsilon. Bisection with 10 steps.
Only allow positive posterior expected reward gain over CFR; otherwise w=0.

Prespecified risk budgets eps=[.1,.4] and seats=[0,1].
Single target posterior is calling-concentrated, as in B1. No new data
or stochastic uncertainty in exact Leduc expectations.
Compare maximum posterior EV over phi to 1D (phi=1) frontier at the same
epsilon. The richer class includes the 1D class, so must weakly improve
EV by construction. A practically meaningful gain is >.005 chips/hand,
with *exactly verified* worst-case floor. Do NOT label as globally optimal
for all Leduc policies or as real-market evidence.
Also report held-out Rank-Selective payoff vs 1D baseline.
If no material gain, retire the extra BR and prefer simpler B1.
