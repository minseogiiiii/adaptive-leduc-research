# QT V7 — The Long-Horizon Value of Information

**Question:** Does accepting the immediate cost of an information-seeking
probe increase later trading/game decision value enough to compensate over
a multi-hand Leduc match?

Status: **pilot completed; independent 48-hand confirmation is running**.
The locked experiment protocol is [QT_V7_PROTOCOL.md](QT_V7_PROTOCOL.md).

## Design

This is a synthetic imperfect-information poker/game experiment, NOT actual
financial trading alpha, nor proof that any human opponent can be exploited.

Three independently learning **on-policy** arms:
1. **No Probe:** exact BR to V4 switching-beliefs type posterior every hand.
2. **Matched CFR:** identical hand schedule but at a 6.6063% predeclared
   shared exploration event, play frozen CFR instead of exact posterior BR.
3. **Active Probe:** on the exact same scheduled events, play a more
   aggressive, information-seeking action-bias policy instead of CFR;
   on other hands compute exact BR from its OWN learned posterior.

Each arm experiences its own legal hand observations; card seeds and
exploration event draws are paired to lower extraneous variance.
At every hand we compute an offline exact conditional expected payoff
vs the known **simulator** opponent policy after the action policy has
already been selected. The oracle is never fed back to the agent.
We also retain realized sample chips (higher variance).

The event rate is computed from an *independent* exact CFR/probe one-hand
risk budget of epsilon=.05 chips; crucially, this **does not certify the
full adaptive hybrid** and makes no multi-hand ruin or drawdown guarantee.

## Stage 1: 24-hand pilot (four independent replicate blocks)

GitHub artifacts and logs:
https://github.com/minseogiiiii/adaptive-leduc-research/actions/runs/38672176754

Values are exact conditional expected chips/hand, not random realized
sample returns.

| Scenario | No Probe | Matched CFR | Active Probe |
|---|---:|---:|---:|
| Stationary Calling | +0.157396 | +0.138884 | +0.125299 |
| Calling→Aggressive halfway | +0.289361 | +0.261315 | +0.254442 |

Paired Active minus Matched-CFR:
- Stationary **-0.013585** chips/hand, nominal 95% bootstrap CI
  [-0.02123,-0.00716].
- Switching **-0.006874** chips/hand, nominal 95% bootstrap CI
  [-0.00992,-0.00384].

Paired Active minus No-Probe:
- Stationary **-0.032097** chips/hand;
- Switching **-0.034919** chips/hand.

Disaggregate expected value differences between Active and Matched:
- Stationary on exploration-event hands: **-0.23604** chips/hand;
  non-exploration hands: **-0.00005** chips/hand.
- Switch on exploration-event hands: **-0.15340** chips/hand;
  non-exploration hands: **+0.00070** chips/hand.

Both pilot cases had mean **2.75 shared probe opportunities per 2-seat
match**, since the opportunity RNG does not depend on opponent scenario.
In the switch case, tiny downstream benefits did not repay the direct
probing loss. Four replicate blocks are NOT confirmatory; small intervals
here do not justify real-world generalization.

Pilot compute cost: ~220 sec stationary and ~413 sec switching, which
shows why naive exact BR+oracle per-hand evaluation is impractical
for latency-sensitive applications. Timing includes offline oracle work.

## Stage 2: independent 48-hand confirmation

New seed **20261028**, **12 independent replicate blocks** x two seats
x 48 hands, three opponent scenarios including held-out Rank-Selective.
The acceptance criterion was locked before inspection:
Active minus Matched >=+0.02 expected chips/hand and paired lower 95% CI >0.
The harder deployment contrast is Active minus No-Probe.
Full data: https://github.com/minseogiiiii/adaptive-leduc-research/actions/runs/38672345692

Results will be inserted after job completion. Do not presume V7 passes.

## Interpretation and limitations

- The aggressive information-probe policy was chosen before V7 from V6's
  observation-identifiability work. No calibration or hyperparameter
  optimization was performed on the pilot/confirmation test data.
- The probe may indirectly change the information received in future hands;
  this is why independently generated on-policy action/observation
  trajectories matter.
- However exact observed average net gains are small and could depend
  on the simple synthetic opponent type model.
- Confidence intervals bootstrap *whole paired-match replicate blocks*,
  not temporally dependent hand observations.
- One-hand safety certificates for the CFR/probe lottery are **not**
  bankroll risk guarantees for the always-exploiting V7 agent.
- Long-horizon population utility across strategic, opponent-adapting
  human players remains untested.

## Research decision gate

The pilot supports a negative preliminary assessment: exploration
significantly decreases average expected chips against the tested scripted
opponents, and its small subsequent benefits do not recover the cost.
Do not advertise profitable exploration. Await independent confirmation
before a final adoption decision.

A future distinct hypothesis might study **belief-conditioned selective
observation** or multi-step Bayesian value of information, but it must
require new locked validation seeds and out-of-model opponents. Do not
tune the current exploration rate after seeing this pilot.
