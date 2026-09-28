# Leduc opponent-learning pilot report

## Research question and fixed design

Does updating a posterior over opponent types improve net chips per hand relative to a policy optimized for the same initial mixture but never updated? This pilot compares that learning arm with a frozen-prior response and the 300-iteration CFR policy. The three conditions were selected before these runs: an in-model calling opponent, a held-out rank-selective opponent, and a calling-to-aggressive switch halfway through a match.

Each condition has 10 independent replicates. A replicate contains 12 hands in each seat, with a fresh uniform prior for each match. Arms share deals and action random streams within each hand. The opponent schedule is shared across arms, while their public histories may diverge. The learning policy uses its own legal observations; it is not told the switch point. All conditions use seed 7, and intervals use 2000 percentile bootstrap resamples of whole two-seat replicates.

## Observed rewards

All values are net chips per hand, averaged over seats. A contrast is learning minus its control; brackets are descriptive 95% bootstrap intervals over replicates.

| Condition | Learning | Frozen prior | CFR | Learning − frozen [interval] | Learning − CFR [interval] |
| --- | ---: | ---: | ---: | ---: | ---: |
| Calling (candidate type) | +0.4667 | +0.1583 | +0.2417 | +0.3083 [+0.0458, +0.5417] | +0.2250 [-0.1083, +0.6042] |
| Rank selective (held out) | +0.2500 | +0.0333 | +0.0250 | +0.2167 [-0.0250, +0.5167] | +0.2250 [-0.3792, +0.9125] |
| Calling to aggressive | +0.5958 | +0.8833 | +0.4625 | -0.2875 [-0.5042, -0.0667] | +0.1333 [-0.4125, +0.6833] |

## Policy-switch phases

The switch occurs before hand 6 (zero-based). The change compares post-switch minus pre-switch paired contrasts within each replicate. It is descriptive, not an estimate of a change-point detector's performance.

| Phase | Learning | Frozen prior | CFR | Learning − frozen [interval] | Learning − CFR [interval] |
| --- | ---: | ---: | ---: | ---: | ---: |
| pre_switch | +0.4000 | +0.6333 | +0.1917 | -0.2333 [-0.5583, +0.1167] | +0.2083 [-0.2500, +0.6167] |
| post_switch | +0.7917 | +1.1333 | +0.7333 | -0.3417 [-0.8417, +0.1333] | +0.0583 [-0.6583, +0.8583] |
- Change in learning − frozen_prior: -0.1083 [-0.8750, +0.5833].
- Change in learning − cfr: -0.1500 [-0.7667, +0.5083].

## Interpretation and limits

The first contrast isolates the value of between-hand posterior updates within this finite model family, since the frozen control makes the same initial mixture response. The CFR contrast also changes the initial policy, so it does not isolate learning. The rank-selective opponent is one constructed held-out rule, not a sample of real players. The switch violates the learner's fixed-type assumption; its posterior is not reset. These seed-dependent outcomes are a small pilot with only ten independent replicate units per condition. Bootstrap intervals are imprecise at that size. Neither phase contrasts nor between-condition gaps identify a causal effect of learning speed or establish generalization to other opponents. No claim of superior play or statistical significance follows from this pilot.

The simulator and exact evaluator share game rules. An independent implementation with reconciled betting rules remains an external cross-check. Before a substantive claim, increase independent replicates, define a power target and broader held-out opponent family, and prespecify the comparison and stopping rule.

## Reproduction and provenance

- `strategy.json` SHA-256: `fefb21881d3a4abe5ae13c27ceb420d896e7e74797cb1b87763c90628a25219a`.
- `rules.md` SHA-256: `d16565525a7fb70955804aa41fc6face224eb619266d4b395dcc35a062c0c644`.
- `pokerlab/game.py` SHA-256: `f8e68df9fbc1ab39e5d3fc04bb1c05c7801d33f75ffa97bd6787afae72efde6c`.
- `pokerlab/cfr.py` SHA-256: `be77d00d5fd21fbd2471d38c00745a785968fe6540db6e308dc4af60313cd870`.
- `pokerlab/bayes.py` SHA-256: `dca961a52da993252b6bb105fae7d8c4b91fa5336339a68cb8627eb7e7d6e98a`.
- `pokerlab/experiment.py` SHA-256: `5d1f70a5a6d4c26d04fea68133d4599b400e106ce79f7c07e0f9008f84cd8786`.
- `pokerlab/report.py` SHA-256: `5f6dd40e543afb123231117f17a7e9674dc4dd0a7ce432f34f7bb090bc9d07a7`.

- From the repository root, run:

```bash
python3 -m pokerlab.experiment --opponent calling --hands 12 --replicates 10 --seed 7 --output results/calling-pilot.json
python3 -m pokerlab.experiment --opponent rank_selective --hands 12 --replicates 10 --seed 7 --output results/rank-selective-pilot.json
python3 -m pokerlab.experiment --opponent calling --switch-to aggressive --switch-after 6 --hands 12 --replicates 10 --seed 7 --output results/switch-pilot.json
python3 -m pokerlab.report --output REPORT.md
```

The pre-switch hand records exactly match the calling-only run for the same seed, seat and hand index. The report builder rejects incomplete seat/hand grids, repeated seeds, wrong phase labels, mismatched input hashes and means or contrasts inconsistent with raw hand rewards. Timing-dependent profiler metadata is not used here.

| Condition | Input JSON SHA-256 |
| --- | --- |
| Calling (candidate type) (`results/calling-pilot.json`) | `75b0d828bc4008aa705bd91bc4e07933b70e1af08206b74b51cf3d2cdc62818d` |
| Rank selective (held out) (`results/rank-selective-pilot.json`) | `4073b71a36f9846cb3460fd29323011bf1977fa5885bf203a4c658fdbc6d3e8a` |
| Calling to aggressive (`results/switch-pilot.json`) | `29d9702869548e643177176315761d2f1e3a213082138b4992f6bf135f749245` |
