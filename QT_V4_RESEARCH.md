# Adaptive Leduc QT V4 — Research Audit

**Research question:** Can Bayesian opponent-type switching and an exact worst-case
risk constraint improve decision quality in a small imperfect-information game,
without leaking private cards or silently changing the historical experiment?

**Status:** The code is research software for a *synthetic frozen Leduc variant*.
It is NOT evidence of live-market trading alpha, profitable poker, or a robust
edge against arbitrary real human opponents. The existing `main` experiments
and reference strategy are unchanged; this report belongs to a research branch.

## New contributions

1. **Hand-boundary switching filter** (`pokerlab/qt_v4.py`):
   Bayesian update from terminal player observations, then Markov hazard
   `h=0.1` refresh toward the initial opponent-type prior before the next hand.
   Hazard zero reproduces the original stationary Bayesian update.
2. **Certified safe whole-hand mixture**: Sample CFR vs posterior-mixture best
   response once *before the hand*. For any opponent policy tau,
   `V((1-w)CFR + w BR, tau) = (1-w)V(CFR,tau) + wV(BR,tau)`.
   Hence worst-case value is at least the same mixture of the two individual
   worst-case values. Select w so this lower bound exceeds
   `Vmin(CFR) - epsilon` with `epsilon=0.1 chips/hand`.
3. **Exact-safe improvement** (`pokerlab/qt_safe_exact.py`):
   Evaluate the opponent's exact best response to the *hidden whole-hand*
   mixture and search for a less conservative safe weight by bisection.
   This is NOT equivalent to independently mixing action probabilities at
   every information set. Four bisection steps leave some unspent risk budget.
4. **Information economics** (`pokerlab/qt_information.py`):
   Exact EVSI/EVPI of a pre-hand noisy signal about opponent policy type.
   No private card or future action is revealed.
5. **Paired evaluation**: CFR, frozen-prior BR, original Bayesian, switching
   Bayesian, conservative safe switching, and exact-safe switching.
   The evaluator records realized chips and exact conditional expected value
   against the simulator's true opponent, which is **never** passed to the
   decision agent. Replicate averages across both seats are the independent
   bootstrap units.

## Prespecified exploratory protocol

- Frozen game and strategy: existing `strategy.json`, unchanged.
- Candidate types: baseline, folding, calling, aggressive.
- Scenarios: stationary calling; held-out rank-selective; calling-to-aggressive
  switch halfway through each match.
- Fixed hazard `0.1`, fixed single-hand worst-case risk budget `0.1`.
- Initial exploratory seed `20261014`, 4 independent replicates,
  8 hands/seat/replicate; 64 hands per arm per scenario.
- Independent-seed replication: seed `20261015`, 8 replicates,
  12 hands/seat/replicate; 192 hands per arm per scenario.
- Paired percentile bootstrap CIs with 1000 draws; very small initial
  replicate count means the exploratory CIs are not confirmatory.
- No hyperparameter selection on the independent-seed replication.

## Exploratory exact conditional EV (chips/hand)

| Policy | Stationary | Held-out | Switch |
|---|---:|---:|---:|
| CFR | +0.0628 | +0.0038 | +0.0801 |
| Frozen prior | -0.0929 | -0.0237 | +0.2880 |
| Original Bayesian | +0.1727 | -0.0117 | +0.2508 |
| Switching Bayesian | +0.1479 | -0.0106 | +0.2728 |
| Conservative Safe | +0.0706 | +0.0029 | +0.1003 |
| **Exact Safe** | **+0.0808** | **+0.0030** | **+0.1109** |

Exploratory evidence is **mixed**:
- Switching improves the switch scenario vs original Bayesian (+0.0219)
  but worsens stationary (+/-0.0247).
- Against an out-of-model opponent, both unrestricted learning policies have
  negative conditional EV, while safe policies stay near the CFR baseline.
- Exact Safe gains about +0.0102 (stationary) and +0.0106 (switch)
  over Conservative Safe, but remains far below unrestricted exploitation
  in these in-model scenarios. The extra safety has an opportunity cost.
- Conservative Safe mean exploit weights: 0.0807 stationary, 0.0554 held-out,
  0.0863 switch. Exact Safe: 0.1237, 0.0569, 0.1224 respectively.
- Realized PnL differs dramatically from exact conditional EV with this small
  sample. Do NOT report sampled PnL as stable skill or alpha.

Exploratory run and raw per-hand JSON artifacts:
https://github.com/minseogiiiii/adaptive-leduc-research/actions/runs/38663434820

## Exact value of information

With equal prior over calling/aggressive and an **80%-accurate** pre-hand
binary signal:
- Seat 0: uninformed 0.26170, informed 0.30476, EVSI **0.04305**,
  perfect-information upper bound EVPI **0.10740** chips/hand.
- Seat 1: uninformed 0.45813, informed 0.57207, EVSI **0.11394**,
  EVPI **0.21406** chips/hand.
- The value is an upper bound on rational willingness to pay for the
  specified signal **in this game**; the signal itself is hypothetical.

Information-value run:
https://github.com/minseogiiiii/adaptive-leduc-research/actions/runs/38663680320

## Validation

- **49 tests passed** on Python 3.10, 3.12, 3.13.
- Original CFR strategy artifact and historical report/figure reproducibility
  checks passed in GitHub Actions.
- Zero-hazard switching matches the existing Bayesian posterior.
- A true best-response opponent to the hidden whole-hand mixture confirms
  the exact worst-case risk floor in integration tests.
- Repeated seeded real-Leduc benchmark produces identical outcomes.

CI: https://github.com/minseogiiiii/adaptive-leduc-research/actions/runs/38663995486

## Independent-seed replication

Workflow: https://github.com/minseogiiiii/adaptive-leduc-research/actions/runs/38663995402

Results are reported separately from exploratory results. A new seed and
longer horizon do not automatically create real-world out-of-sample market
validation; all opponent policies are still synthetic.

## Reproduction

```bash
python3 -m pip install -e .
python3 -m unittest discover -s tests -v
python3 -m pokerlab.qt_v4 --scenario switch --hands 8 --replicates 4 \
  --seed 20261014 --hazard 0.1 --budget 0.1 --output qt-v4-switch.json
python3 -m pokerlab.qt_information --accuracy 0.8
```

## Interpretation and limitations

- **Single-hand expected-value** risk certificate, not a realized drawdown,
  CVaR, or multi-hand wealth guarantee.
- Exact-safe optimization is computationally expensive; this is an offline
  research oracle, not a low-latency production trader.
- The four modeled types are artificial perturbations of a CFR policy;
  held-out opponents are also hand-coded. External validity is limited.
- The exact conditional EV evaluator knows the simulator's true opponent
  policy, but only *after* the agent has committed to its pre-hand policy
  and exclusively for offline evaluation.
- Opponent observations are terminal player observations; unrevealed cards
  are marginalized by the likelihood code.
- A few independent replicates do not establish statistical significance.
- Selecting the best-performing model after seeing all scenarios would be
  data snooping. Negative results are retained.

**Next gate:** Inspect the independent-seed replication without changing the
predeclared hazard or budget. A risk-constrained policy should be judged on
both the explicit worst-case guarantee and the cost in expected reward;
neither a single positive mean nor a narrow bootstrap interval from a small
number of replicates is enough for a strong generalization claim.
