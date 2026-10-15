# QT V6 — Changepoint Evidence and the Price of Safety

**Research status: complete A1–A3 and B1–B2 protocol cycles.**

This is reproducible research in a **synthetic six-card Leduc variant**.
It is NOT a claim of profitable real poker play, financial market alpha,
actual market-making ability, or zero chance of bankroll ruin.

**Base:** [QT V4 research and validation](QT_V4_RESEARCH.md). This branch
preserves the original rules, CFR strategy.json and previous artifacts.

## Executive decision

**A — No replacement of V4 changepoint filter.** BOCPD with fixed hazard
provides exactly the SAME marginal predictive type probabilities as V4;
duration-sensitive hazard fails the preregistered joint adoption gate.
Follow-ups A2 and A3 show some more informative probe actions but their
expected chip cost or exact worst-case risk makes one-hand benefits tiny.

**B — Retain exact single-hand risk/reward frontier as substantive result.**
At identical conservative risk budgets, exact game-theoretic certification
permits higher exploitative weights and posterior expected chips than
the earlier loose safety bound. The cost of safety is quantifiable and
depends on seat, belief and allowed budget. A larger candidate policy
set with two specialist BRs did not improve the frontier on the
prespecified grid; model complexity was not justified.

## A0 — Mathematical identity: constant BOCPD == V4 marginal prediction

Joint state is P(run_age, type | all legal observations); update hand
likelihoods, then transition to a new segment with hazard H(run_age).
For any constant H(r)=h, summing over run_age yields exactly:

```text
type_prior_next[j] = (1-h)*type_posterior_after_hand[j]
                     + h*initial_type_prior[j]
```

This is exactly the V4 SwitchingBeliefs recursion. Keeping a run-length
posterior adds a diagnostic but *no new predictive information* under
the same fixed hazard generative model. Test suite checks five hazard choices (0, .03, .1, .5, 1); the
empirical identity discrepancies are at floating-point rounding scale
(<=~1e-15). Avoid selling BOCPD name alone as a new alpha feature.

Theory: Adams and MacKay (2007), Bayesian Online Changepoint Detection:
https://arxiv.org/abs/0710.3742

## A1 — Duration dependent changepoint, preregistered failure

Specified BEFORE outcome inspection:
- duration <6 -> hazard 0.03; duration >=6 -> hazard 0.20;
- V4 reference constant hazard 0.10;
- 7 opponent scenarios, 24 hands per seat, 48 independent paired
  replicate blocks each; seed **20261020**;
- 16,128 legally observed terminal hand trajectories;
- outcome: *prequential* log likelihood, nats/hand, scored before updating.
- Bonferroni-compatible 97.5% percentile cluster bootstrap interval for
  2 co-primary hypotheses stationary Calling and Calling->Aggressive.

| Condition | Duration minus V4 | 97.5% bootstrap |
|---|---:|---|
| Stationary Calling | **-0.004434** | [-0.005833,-0.002928] |
| Stationary Aggressive | -0.006298 | [-0.007816,-0.004817] |
| Calling -> Aggressive | +0.004570 | [+0.002621,+0.006443] |
| Aggressive -> Calling | +0.004255 | [+0.003119,+0.005483] |
| Early switch at 1/4 | -0.001977 | [-0.004080,+0.000031] |
| Heldout Rank Selective | -0.001661 | [-0.002560,-0.000687] |
| Heldout Pressure Reactive | -0.006272 | [-0.007773,-0.004722] |

The co-primary criterion required both >+0.005 nats/hand and strictly
positive corrected CI. It FAILED: stationary loses, switch effect below
economic target. Duration-dependent hazard helps at some later switches
but destroys evidence elsewhere. Early switch also fails.

At true switch, average same-hand posterior P(changepoint) only
~0.103 (calling->aggressive) and ~0.104 (reverse); on other hands
~0.087–0.088. Posteriors overlap strongly. This is a descriptive
calibration/identifiability warning, NOT a formally calibrated change
detector or causal attribution.

Pilot: https://github.com/minseogiiiii/adaptive-leduc-research/actions/runs/38669024787
Independent confirmation:
https://github.com/minseogiiiii/adaptive-leduc-research/actions/runs/38669209502

## A2 — Observation bottleneck and probing cost

NEW predeclared follow-up after A1. Fresh seed **20261022**; 24 independent
match blocks x 2 seats x 48 hands per opponent type x 2 opponent types
x 3 probe policies = **13,824 hands**. All classifier likelihood scores
come from legal player observations only, no private opponent cards
unless revealed at showdown.

| Probe | Binary type AUC | Binary type accuracy |
|---|---:|---:|
| CFR | 0.70642 | 0.64453 |
| Uniform | 0.71093 | 0.64865 |
| Aggressive-biased CFR probe | **0.73227** | **0.66298** |

Aggressive vs CFR AUC delta +0.02585,
paired cluster 95% CI [+0.01432,+0.03861]. Its preregistered
usefulness target was >=0.03, so *fails materiality* even before cost.
Two alternative probes were considered and nominal 95% CIs
are not a familywise multiplicity-corrected proof.

Expected one-hand chip payoff against 50/50 Calling/Aggressive:
- Seat0 CFR -0.00950 -> Aggressive Probe -0.27593:
  **-0.26642 chips/hand opportunity cost**.
- Seat1 CFR +0.16976 -> Aggressive Probe +0.01897:
  **-0.15078 chips/hand opportunity cost**.

Known-type *optimistic* change-detection test with prior P(change)=0.1:
a posterior>0.5 requires log Bayes factor >log(9)=2.197.
- CFR true positive ~1.22%, false positive ~0.17%.
- Aggressive probe true positive ~2.73%, false positive ~0.30%.
In one observed hand even KNOWN Calling vs Aggressive is usually
insufficient to declare a change with high confidence.
This is a plausibility/identifiability diagnostic, not proof of
impossibility to learn over many observations.

Protocol: [A2](QT_V6_A2_PROTOCOL.md)
Independent evidence:
https://github.com/minseogiiiii/adaptive-leduc-research/actions/runs/38669486091

## A3 — Risk-constrained information acquisition

A new locked experiment after A2. **Fresh seed 20261024**; 24 independent
blocks. A hand-level hidden lottery CFR vs Aggressive probe was constrained
by the EXACT global adversarial best response so that each seat's
worst *expected one-hand chip payoff* cannot worsen by more than
0.05 versus original CFR. The shared probed-hand chance was the lower
of seat-specific certified weights.

- Safe aggressive-probe mixture probability **0.066063 (6.61%)**.
- Per-seat unconstrained safe probe caps: 6.61% seat0, 12.40% seat1.
- One-hand worst payoff: seat0 -0.21251 vs safety floor -0.21275;
  seat1 -0.01709 vs floor -0.05077.
- Fresh seed CFR accuracy **0.63650**, mixed probe **0.63775**.
- Absolute improvement **+0.001247**, i.e. **+0.125 percentage points**,
  95% cluster bootstrap [+0.000774,+0.001706] (accuracy units).
- True change sensitivity for a known two-type detector only
  **1.302% -> 1.354%**.
- Fixed prior exact payoff: seat0 -0.00950 -> -0.02711, seat1
  +0.16976 -> +0.15979. No violation of worst-case guarantee.

The predefined usefulness gate required >=+1.00 percentage point
classification improvement, so **FAILED** despite positive sampling
interval. The result is a single-hand cost-benefit finding: potentially
valuable long-horizon information carryover is NOT evaluated.

Protocol: [A3](QT_V6_A3_PROTOCOL.md)
Fresh confirmation:
https://github.com/minseogiiiii/adaptive-leduc-research/actions/runs/38669693627

## B1 — Exact strategic risk/reward frontier (main positive result)

Policy is a *hidden prehand* mixture of existing CFR vs BR to a declared
opponent distribution. For any opponent policy sigma:

```text
V((1-w)*CFR + w*BR, sigma) =
    (1-w)*V(CFR,sigma) + w*V(BR,sigma)
```

We evaluate `min_sigma V(lottery, sigma)` by exact opponent BR, not
by independently mixing action probabilities at each information set.
The risk constraint is **worst-case one-hand EXPECTED chip payoff floor**
at baseline worst - epsilon, NOT a max-loss guarantee, CVaR, Kelly or
drawdown. Max feasible mixture w within this restricted one-dimensional
family is the optimal posterior-EV choice if BR improves posterior EV.

Two seats x two declared beliefs (uniform four types and 85% Calling) x
eight predeclared budgets ε=0, .02, .05, .1, .2, .4, .8, 2.
Fully enumerated finite game; deterministic exact result has NO sampling
p-value or stochastic CI.

Representative: Seat1, 85% Calling posterior:

| ε extra worst expected loss | exact BR weight | posterior EV chips/hand | exact worst EV |
|---|---:|---:|---:|
| 0 | 0.11621 | +0.18537 | -0.00062 |
| 0.02 | 0.17461 | +0.19855 | -0.02070 |
| 0.05 | 0.25374 | +0.21641 | -0.05067 |
| 0.10 | 0.32891 | +0.23337 | -0.10019 |
| 0.20 | 0.43536 | +0.25740 | -0.20062 |
| 0.40 | 0.63063 | +0.30147 | -0.40019 |
| 0.80 | 0.97219 | +0.37856 | -0.80072 |
| 2.0 | 1.0 | +0.38483 | -0.83333 |

Seat1 benchmark CFR posterior EV +0.15914, exact worst -0.00076575.
At ε=0, mixed strategy improves BOTH posterior EV (+0.02623)
and worst expected payoff (to -0.000624), because the frozen 300-iteration
CFR reference is an APPROXIMATE equilibrium, not the exact game equilibrium.
Do not present this as a guarantee of 'free alpha' or a general equilibrium
dominance theorem.

At ε=.1, conservative `safe_weight` allows ~0.12011 BR weight,
exact risk oracle allows **0.32891**; expected value increase relative to
conservative **+0.04713 chips/hand**, with verified worst payoff
~ -0.10019 >= baseline worst -0.1 ~ -0.10077.

Marginal price of extra risk is nonlinear:
- ε=.05→.10: EV +.01697 for +.05 budget (~.34 gain/budget).
- ε=.40→.80: EV +.07709 for +.40 budget (~.19 gain/budget).
- ε=.80→2.0: EV +.00628 for +1.20 of EXTRA ALLOWED budget,
  but the full-BR policy saturates at worst payoff -0.83333; actual
  worst-case loss rises only ~0.03261, giving ~0.1925 EV per unit of
  **actual** additional worst-case exposure.
Do not mistake the finite difference per *unused allowance* for the
marginal cost of *risk actually taken*. The high-risk frontier is
nearly linear between ε=.4 and the full-BR saturation point; the
lower-budget portion is more curved.

**Model risk still matters**: under uniform belief, Seat0's heldout
Rank-Selective EV moves ~ -0.08687 at ε=0 to **-0.10561** at ε=.8
and -0.13065 at full BR. Under other seats or beliefs heldout EV can
instead increase. A robust choice of ε needs risk preference and a
credible uncertainty set; no universal optimal ε follows.

Exact frontier runs and downloadable JSON:
https://github.com/minseogiiiii/adaptive-leduc-research/actions/runs/38669193065

## B2 — Richer specialist strategy set: no additional benefit

After seeing B1, specified [B2](QT_V6_B2_PROTOCOL.md):
global opponent best-response safety oracle with three prehand policies:
CFR, BR to Calling-concentrated posterior, BR to known Aggressive opponent.
Tested phi∈{0,.2,.4,.6,.8,1}, ε=.1/.4, both seats.
The expanded grid contains one-dimensional phi=1, so max cannot be worse.

**All four contexts preferred phi=1**. Improvement compared to the
same-grid one-dimensional target policy was precisely **0 chips/hand**.
A second specialist BR lowered target-belief EV more than it improved the
safety tradeoff in this declared candidate set. This is a NEGATIVE result
for the added complexity, NOT a proof all richer policy sets are useless.

Run: https://github.com/minseogiiiii/adaptive-leduc-research/actions/runs/38669908480

## Quality assurance, statistical scope, and honest interpretation

- No modifications to original baseline game, strategy.json, published
  historical experiments or their reproducible results.
- New tests cover marginal equivalence, legal observation timing,
  probabilistic normalization, all risk floors, monotonicity, deterministic
  replay and active-probe classifier semantics.
- A1 and A2/A3 reuse same *synthetic family* of opponents but use fresh
  experimental RNG seeds. This supports procedural replication and
  conditional inference, NOT external validity to real opponents.
- A1 corrected co-primary CI target failed. A2 sample AUC advantage did not
  meet effect threshold, and A3 small significant accuracy gain also failed
  precommitted practical threshold. No p-hacking/hazard retuning on locked data.
- B exact tree enumeration has no stochastic sampling uncertainty, but
  structural model uncertainty, restrictions to candidate policies, baseline
  CFR approximation and compute cost remain substantial.
- Safe hand-level mixture assumption: adversary cannot see the hidden lottery
  draw, although it can condition on observed within-hand behavior.
- No multi-hand ruin bound, hedging interpretation or finance alpha measured.

### Research recommendation

**KEEP** B1 exact risk/reward frontier as the most useful QT research result.
**RETAIN** A0 identity and A1–A3 negative/conditional results as the
supporting study about the cost of identifying strategic change.
**DO NOT** advertise A's duration hazard or A3 probing as superior deployed
policies. **DO NOT** increase the number of models hoping to mine a positive
backtest.

The next hard research question is *long-horizon net value of an extra
observation*: can a modest one-hand probing cost be repaid by better
future decisions over 20–100 hands against genuinely nonstationary,
held-out policy families? It requires a predeclared utility criterion,
realizable risk budgets, explicit computation-cost limits and fresh
out-of-family opponents.

Related theory: Adams & MacKay (2007) BOCPD
https://arxiv.org/abs/0710.3742 ; Ganzfried & Sandholm (2012),
Safe Opponent Exploitation:
https://www.cs.cmu.edu/~sandholm/safeExploitation.ec12.pdf ;
Ge et al. (ICML 2024) Safe and Robust Subgame Exploitation:
https://proceedings.mlr.press/v235/ge24b.html
