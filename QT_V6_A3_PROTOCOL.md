# A3 locked follow-up — exact safety constrained information acquisition

A2 found aggressive probing provides extra single-hand opponent-type
information, but at a large expected-chip cost. A3, designed AFTER
observing A2, uses **fresh seed 20261024** for statistical validation.

Hypothesis: a hidden whole-hand lottery over CFR and aggressive probing,
whose mix weight satisfies an exact worst-case opponent-payoff floor of
Vmin(CFR) - 0.05 chips/hand ON EACH SEAT, retains enough information
gain to justify further research.

For each seat, compute the largest feasible aggressive probe probability
with exact zero-sum worst-case best response. Use the SMALLER of the two
seat-specific safe weights as one common choice. The opponent cannot
observe this policy lottery draw, except through our actions.
No opponent hidden information is used.

Information metrics for a mixture are **linear** in the probability of the
pre-hand policy draw WHEN THE ANALYST KNOWS WHICH POLICY WAS DRAWN** for
the particular per-hand classifier accuracy and fixed log-likelihood-ratio
threshold sensitivity. We do NOT claim AUC is linear under mixtures.
The actual probe data are separately evaluated at seed 20261024:
24 independent replicate blocks, both seats, 48 observations per true
opponent type and probe, as per A2.

Compare base CFR vs constrained aggressive probe on:
- 50/50 Calling/Aggressive one-hand classification accuracy;
- posterior>0.5 detection sensitivity with 10% known-type switch prior;
- exact model-specific expected one-hand chips;
- exact worst-case opponent one-hand expected payoff floor.
- cluster paired bootstrap CI for the accuracy gain on 24 replicate blocks.

Practical pass criterion: classification accuracy gain >= 0.01
(one percentage point) AND CI lower bound positive, while satisfying the
exact worst-case 0.05 risk budget on BOTH seats.

No optimization of AUC or likelihood thresholds on seed 20261024.
This is not a long-horizon EVSI or bankroll risk guarantee.
