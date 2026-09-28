# Fixed follow-up experiment protocol

This protocol was written before the follow-up outcomes were generated. The
primary question is whether between-hand Bayesian updates improve reward over
the same initially optimized response with its prior frozen across a family of
three opponent policies absent from the learner's four candidate types.

## Conditions and stopping rule

- Three stationary held-out opponents: `rank_selective`, `round_polarized`,
  `pressure_reactive`. Their action multipliers are fixed in
  `pokerlab/experiment.py`. The first uses the private rank and board pair;
  the second changes bias between betting rounds; the third backs off after
  an opponent bet or raise in the current round. All use the same 5% action
  smoothing as the candidate policies.
- A separate stress condition changes `calling` to `aggressive` immediately
  before zero-based hand 6. The learner retains its posterior and is not
  informed of the change. This condition is secondary.
- Exactly 24 independent replicates per condition, 12 hands in each seat per
  replicate, seed `20261001`, the committed 300-iteration `strategy.json` and
  frozen `rules.md`. A replicate comprises both seats. Each condition starts
  a new learner for each seat. All arms and conditions share the same
  replicate/seat/hand seed schedule. Stop when all four conditions complete;
  no outcome-dependent extension or selection.

## Analysis fixed in advance

The **primary contrast** is the equally weighted mean across the three held-out
conditions of learning minus frozen-prior net chips per hand. For each
replicate, first average over its two seats and 12 hands within each condition,
then average its three paired differences. Use the mean of those 24 replicate
values, and a 95% percentile bootstrap interval from 2,000 resamples of the
24 **joint** replicate indices with replacement, using random seed
`20261002`. This preserves the shared random seed pairing across conditions.
No hypothesis test or multiplicity-adjusted inference is claimed.

Show condition-specific learning, frozen-prior and CFR means and the two
paired contrasts with their existing replicate bootstrap intervals. Show the
switch condition separately, including its pre/post phase and change in
paired contrasts. All intervals are descriptive. Do not choose a subset of
opponents after observing payoffs. Report the raw input hashes, code hashes,
and exact reproduction commands.

For planning, the pilot held-out paired-difference standard deviation was
about 0.48 chips/hand across ten replicates. A normal approximation with that
standard deviation gives roughly 20 replicates to resolve a 0.30-chip effect
at 80% power with a two-sided 5% test; 24 allows some margin. This is a
planning approximation, not a guarantee: the family-level variance, model
misspecification and non-normality may differ. The new seed avoids reusing
the pilot draws, while the experiment design still builds on its findings.

The companion `pokerlab.followup` validates the saved raw rows and regenerates
`FOLLOWUP.md`. The pilot `REPORT.md` remains a separate historical analysis.
