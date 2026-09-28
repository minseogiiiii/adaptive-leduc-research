# Leduc held-out opponent follow-up

## Fixed design and primary result

The conditions, primary contrast, replicate unit, seed and stopping rule were fixed in `FOLLOWUP_PROTOCOL.md` before these outcomes. Each condition uses 24 independent two-seat replicates with 12 hands per seat. All share the same hand seed schedule; each match starts with a fresh prior. The three held-out policies are absent from the four candidate types.

Primary equally weighted learning minus frozen-prior contrast over the three held-out opponents: **+0.0012 [-0.1314, +0.1244] net chips/hand** (95% descriptive joint-replicate bootstrap, 2,000 resamples; seed 20261002).

## Condition results

All rewards are chips per hand averaged over seats. Brackets are descriptive 95% replicate bootstrap intervals.

| Condition | Learning | Frozen prior | CFR | Learning − frozen [interval] | Learning − CFR [interval] |
| --- | ---: | ---: | ---: | ---: | ---: |
| rank_selective | -0.6181 | -0.5608 | -0.3472 | -0.0573 [-0.2517, +0.1285] | -0.2708 [-0.6545, +0.1424] |
| round_polarized | -0.2396 | -0.1858 | -0.2795 | -0.0538 [-0.3438, +0.2274] | +0.0399 [-0.2500, +0.3333] |
| pressure_reactive | +0.9288 | +0.8142 | -0.2656 | +0.1146 [-0.1372, +0.3403] | +1.1944 [+0.6302, +1.7535] |
| calling to aggressive | -0.4583 | -0.3594 | -0.4323 | -0.0990 [-0.3333, +0.1250] | -0.0260 [-0.3698, +0.3177] |

## Switch stress condition

The opponent changes before zero-based hand 6; the learner's beliefs persist across the switch. This condition is secondary and excluded from the family contrast.

| Phase | Learning | Frozen prior | CFR | Learning − frozen [interval] | Learning − CFR [interval] |
| --- | ---: | ---: | ---: | ---: | ---: |
| pre_switch | -0.7292 | -0.9583 | -0.4931 | +0.2292 [-0.0521, +0.5312] | -0.2361 [-0.6424, +0.1840] |
| post_switch | -0.1875 | +0.2396 | -0.3715 | -0.4271 [-0.7639, -0.0938] | +0.1840 [-0.2951, +0.6806] |
- Post minus pre learning − frozen_prior: -0.6562 [-1.0729, -0.2083].
- Post minus pre learning − cfr: +0.4201 [-0.1597, +1.0451].

## Interpretation and limits

The frozen-prior response starts from the same policy as the learner; their between-hand posterior updates are the intended difference. The three held-out policies are constructed rules rather than a representative sample of people. Common random numbers reduce some comparison noise, but the trajectories can diverge. The experiment builds on a small earlier pilot, and 24 replicate units leave substantial uncertainty. The bootstrap interval is descriptive; it is not a claim of general performance or a multiplicity-adjusted significance test. The switch violates the learner's stationary-type assumption. The simulator and evaluator still share implementation; external rule-matched validation remains outstanding.

## Reproduction and provenance

Run from the repository root, then rerun `python3 -m unittest discover -s tests -v`.

```bash
python3 -m pokerlab.experiment --opponent rank_selective --hands 12 --replicates 24 --seed 20261001 --output results/followup-rank_selective.json
python3 -m pokerlab.experiment --opponent round_polarized --hands 12 --replicates 24 --seed 20261001 --output results/followup-round_polarized.json
python3 -m pokerlab.experiment --opponent pressure_reactive --hands 12 --replicates 24 --seed 20261001 --output results/followup-pressure_reactive.json
python3 -m pokerlab.experiment --opponent calling --switch-to aggressive --switch-after 6 --hands 12 --replicates 24 --seed 20261001 --output results/followup-switch.json
python3 -m pokerlab.followup --output FOLLOWUP.md
```

The generator checks design hashes, complete hand grids, posterior continuity, rewards, bootstrap intervals, and the identical seed schedule. It recomputes the primary contrast from replicate means.

| Input | SHA-256 |
| --- | --- |
| `strategy.json` | `fefb21881d3a4abe5ae13c27ceb420d896e7e74797cb1b87763c90628a25219a` |
| `rules.md` | `d16565525a7fb70955804aa41fc6face224eb619266d4b395dcc35a062c0c644` |
| `FOLLOWUP_PROTOCOL.md` | `8e3efb9d2b9e64eb5f1f3a2ccad012dbe527e5ee74ab803bfc73ec634caef1ae` |
| `pokerlab/game.py` | `f8e68df9fbc1ab39e5d3fc04bb1c05c7801d33f75ffa97bd6787afae72efde6c` |
| `pokerlab/cfr.py` | `be77d00d5fd21fbd2471d38c00745a785968fe6540db6e308dc4af60313cd870` |
| `pokerlab/bayes.py` | `dca961a52da993252b6bb105fae7d8c4b91fa5336339a68cb8627eb7e7d6e98a` |
| `pokerlab/experiment.py` | `d71a32b4d876d7d724f945fc501e0d86ea9012e6b1b8baa09d1a20da6d8cb46c` |
| `pokerlab/report.py` | `5f6dd40e543afb123231117f17a7e9674dc4dd0a7ce432f34f7bb090bc9d07a7` |
| `pokerlab/followup.py` | `728ea7ad01d8534b2999c1d9d4c3faed5b0d22cde416687d475882772dfb5228` |
| `results/followup-rank_selective.json` | `c506e90c88007028661336becb2bc549f4e092d1b1ce0d090467da13c19a4f99` |
| `results/followup-round_polarized.json` | `0882105b0cf343cd546ca0945a8e9f1990f005c2c705f8b5b02c34bcef15babf` |
| `results/followup-pressure_reactive.json` | `46c3844919ae9438785a0c673310839146024ca263f4b55891408eef41fc5cba` |
| `results/followup-switch.json` | `4a486a8801b2eb513a00b5b5dbdd757aa4f86bef5e02ec4f65b6be0606768f4b` |
