# QT V5 — Precommitted Research and Evidence Audit

## Question

Can sequential evidence about opponent policy stability jointly determine
(1) whether to forget old type evidence and (2) how much worst-case risk
to spend on exploitation, in a way that improves actual Leduc decisions?

The repo is a **small synthetic poker game**, not proof of real trading alpha.

## Methods (locked before validation)

- Opponent models and game rules identical to V4.
- `H0`: static hidden opponent type; `H1`: type refresh probability 0.2
  between hands. Start P(H1)=0.25.
- Update P(H1|history) from predictive hand likelihood under each H.
- Mixture of the two next-hand type distributions yields the policy model.
- Risk budget `0.02 + 0.08 * type_concentration * (1-P(H1))`.
  The exact-safe solver still enforces a per-hand expected worst-case payoff
  floor. Type concentration **does not** establish model calibration.
- Primary inference measure: prequential log-likelihood of the opponent's
  *legally observed* terminal hand, evaluated under **identical** CFR
  probing histories for all candidate learners.
- Primary contrasts: V5 vs V4 fixed hazard0.1 on
  `stationary_calling` and `switch_calling_aggressive`.
- Adoption requires both contrasts >= +0.005 nats/hand with multiplicity
  adjusted positive bootstrap intervals; also no large heldout regression.
- Protocol: `QT_V5_PROTOCOL.md`, written before outcome examination.

## Stage A pilot and independent confirmation

Pilot: 12 independent replicates, 24 hands per seat, seed 20261014.
https://github.com/minseogiiiii/adaptive-leduc-research/actions/runs/38665949945

Independent confirmation: **36 replicate blocks**, two seats per block,
24 hands per seat, **1728 hands per scenario**, seed 20261017.
Six scenarios = **10,368 total observed hands** across the test suite.
https://github.com/minseogiiiii/adaptive-leduc-research/actions/runs/38666056544

A positive contrast means V5 predicts the next observed hand better than
V4. Values are mean paired log-score differences (nats/hand). Confidence
intervals are *cluster bootstrap over 36 replicate means*, NOT resampling
the 1,728 correlated hands independently.

| Scenario | V5 minus V4 | Bonferroni-compatible 97.5% bootstrap CI |
|---|---:|---:|
| Stationary calling | +0.01315 | [+0.00873, +0.01759] |
| Stationary aggressive | +0.00868 | [+0.00422, +0.01300] |
| Switch calling→aggressive | **-0.01903** | **[-0.02329, -0.01456]** |
| Switch aggressive→calling | **-0.02283** | **[-0.02805, -0.01729]** |
| Held-out rank-selective | +0.00056 | [-0.00114, +0.00231] |
| Held-out pressure-reactive | +0.01546 | [+0.01241, +0.01873] |

**Primary inference adoption gate FAILED.** Although stationary calling
improves, the switch scenario significantly *deteriorates* and misses the
+0.005 minimum practical effect by a wide margin. No overall universal
benefit should be claimed. Negative results are retained, not tuned away.

Mechanism: The model posterior over a stationary process can be sticky.
Average P(H1), across all observed decisions:
- stationary calling: 0.2091,
- switch calling→aggressive: 0.2882.
The models distinguish the environments somewhat but not fast enough to
match V4's fixed 0.1 hazard after switching. The second-half
switch calling→aggressive log-score contrast was even worse:
**-0.04338 nats/hand** with 97.5% CI [-0.05377, -0.03377].

These are statistically defensible statements only for the exact synthetic
protocol. Bootstrap interval exclusion of zero is not a guarantee of
distributional generalization.

## Stage B: On-policy economic value

See the `qt_v5_decision.py` workflow. This tests actual game play and
exact conditional EV as a separate metric. It compares V5 against original
Bayesian, fixed-switching, and fixed-exact-safe policies, with paired seeds
and two-seat match replicate bootstrap. **Do not confuse an improvement in
prediction log score with an improvement in expected trading utility.**

## Current research decision

The V5 *inference* mechanism should NOT replace V4 based on the
precommitted evidence gate. Do not cherry-pick its positive stationary
or heldout result to claim success; those come with negative switch results.

Avoid post-hoc tuning the switch hazard or model prior on these
confirmation scenarios and reporting the new value on the same seeds.
A radically revised design would need a new locked specification, fresh
evaluation scenarios, and a new confirmation dataset.

## Reproduce

```bash
python3 -m pip install -e .
python3 -m unittest discover -s tests -v
python3 -m pokerlab.qt_v5_forecast --scenario stationary_calling \
  --hands 24 --replicates 36 --seed 20261017
python3 -m pokerlab.qt_v5_forecast --scenario switch_calling_aggressive \
  --hands 24 --replicates 36 --seed 20261017
python3 -m pokerlab.qt_v5_decision --scenario switch_calling_aggressive \
  --hands 12 --replicates 8 --seed 20261018
```
