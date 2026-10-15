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

## Stage B: On-policy economic value — COMPLETED

[GitHub Actions raw outputs and logs](https://github.com/minseogiiiii/adaptive-leduc-research/actions/runs/38666177869).
All three scenario jobs passed. Each scenario used **8 independent paired
match replicates x both seats x 12 hands = 192 sampled hands per policy**
(5 arms, seed 20261018). Paired bootstrap uses the 8 replicate means, not
individual hands. Exact conditional EV is an offline oracle metric that knows
the true synthetic opponent policy, **not given to the agents**.

### Exact conditional EV (chips/hand)

| Policy | Stationary calling | Calling→aggressive | Held-out rank selective |
|---|---:|---:|---:|
| Original Bayesian | +0.15973 | +0.30940 | -0.01828 |
| V4 fixed switching | +0.14327 | **+0.32779** | -0.01925 |
| V5 evidence | +0.15753 | +0.31844 | -0.01835 |
| V4 exact-safe budget 0.1 | +0.07809 | +0.10186 | +0.00389 |
| V5 gated exact-safe | +0.07389 | +0.09030 | +0.00405 |

**V5 evidence minus V4 fixed switching**:
- Stationary **+0.01426** (95% paired CI **[-0.00256, +0.03280]**)
- Switch **-0.00935** (95% paired CI **[-0.01752, -0.00014]**;
  97.5% interval contains zero)
- Held-out **+0.00090** (95% paired CI **[-0.00193, +0.00373]**)
- Post-switch half only **-0.02857** (95% CI **[-0.04616, -0.01242]**).
- Practical threshold for policy improvement was +0.02 chips/hand:
  NONE of the V5 positive overall contrasts clear this threshold, and the
  most critical switching case is negative.

**V5 gated-safe minus V4 fixed exact-safe**:
- Stationary **-0.00420** (95% CI [-0.00595, -0.00254])
- Switch **-0.01156** (95% CI [-0.01404, -0.00921])
- Held-out **+0.00016** (95% CI [-0.00027, +0.00056]).
- Gated budget averaged **0.03117** stationary, **0.02822** switch,
  **0.02717** held-out, compared with V4 constant epsilon **0.1**.
  This is a STRICTER worst-case one-hand expected-payoff floor; it does not
  prove superior risk-adjusted utility because no agreed common risk
  preference or wealth-utility objective is optimized here.
- The safety certificate was not violated in the sampled prehand decisions.

**Realized sampled chips/hand are much noisier**:
- V5 evidence: -0.1250 stationary, +0.2031 switch, +0.0677 held-out.
- V5 gated-safe: -0.3177 stationary, -0.0885 switch, -0.2188 held-out.
- These empirical rewards can diverge widely from conditional EV in just 8
  replicate blocks; DO NOT advertise them as stable performance or alpha.

**Compute burden**:
- Each scenario's complete 5-arm offline evaluation took ~1161-1218 sec
  on a GitHub-hosted runner.
- Reported per-hand `prepare_and_exact_eval` times averaged roughly
  **0.74-0.75 sec** for evidence, and **1.90-2.03 sec** for safe agents.
  These timings INCLUDE the *offline exact opponent oracle*, so they are
  **not pure live decision latencies**, but demonstrate high research
  evaluation cost. The exact-safe optimization also invokes multiple
  full game-tree best responses. No production latency claim is justified.

**Decision-stage conclusion**: no robust or practically material expected
reward improvement was demonstrated by V5 over V4. In the main changing
regime V5 did worse. The risk gate achieved a mathematically stricter
one-hand worst-case expected-payoff floor but paid a substantial foregone
expected-reward cost compared with unrestricted exploitation. 8 replicate
blocks are inadequate for broad external inference; the stronger 36-replicate
prequential prediction experiment already failed the preregistered gate.

**Do not confuse predictive log score, realized chips and oracle conditional
expected payoff.** These estimate different quantities.

## Current research decision

**NO GO: retain V4 as the candidate and V5 as documented negative research.**
Both the precommitted inference gate and the practical on-policy decision
gate failed. Do not cherry-pick positive stationary or held-out cases.
V5 was preserved in a draft PR rather than merged into main.

The Bayesian probability of the H1 *changing-process hypothesis* is
NOT the posterior probability of an actual change point on this hand.
Future research should model run length/change points explicitly
and evaluate with new locked opponent families and distinct test seeds.

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
