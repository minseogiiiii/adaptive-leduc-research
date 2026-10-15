"""QT V4: switching beliefs, certified safe whole-hand mixing, paired evaluation.

This is an additive research module. It does not change frozen Leduc rules,
CFR reference strategy, or published historical experiments.

The safety certificate applies to EXPECTED single-hand worst-case payoff in
the exact frozen zero-sum game, not to realized losses or a multi-hand bankroll.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

from .bayes import BayesianOpponentModel, _hand_likelihood, example_models
from .cfr import PolicyAgent, TabularPolicy, best_response, best_response_mixture, profile_value
from .experiment import HELD_OUT_POLICIES
from .simulate import run_hand


class SwitchingBeliefs:
    """Finite-state hand-boundary type filter, compatible with AdaptiveAgent.

    z_t is fixed within a hand. After the terminal observation, update via
    Bayes; before the next hand, mix filtered belief with initial prior.
    """
    def __init__(self, models, hazard: float = 0.1, prior=None):
        if not models or not math.isfinite(hazard) or not 0 <= hazard <= 1:
            raise ValueError("Invalid models or hazard")
        self.names = tuple(models)
        self.policies = tuple(models.values())
        self.initial_prior = tuple(prior if prior is not None else
                                   (1 / len(models) for _ in models))
        if (len(self.initial_prior) != len(models)
                or any(not math.isfinite(p) or p <= 0 for p in self.initial_prior)
                or abs(math.fsum(self.initial_prior) - 1) > 1e-10):
            raise ValueError("Invalid prior")
        self.hazard = hazard
        self._predictive = self.initial_prior
        self.hands_seen = 0
        self.last_log_score = None

    @property
    def posterior(self):
        """Pre-hand predictive type distribution; never post-hand filtered."""
        return self._predictive

    def named_posterior(self):
        return dict(zip(self.names, self.posterior))

    def update(self, terminal):
        likelihoods = [_hand_likelihood(terminal, p) for p in self.policies]
        logs = [math.log(w) + math.log(l) if w > 0 and l > 0 else -math.inf
                for w, l in zip(self._predictive, likelihoods)]
        m = max(logs)
        if m == -math.inf:
            raise ValueError("Impossible observation under all opponent types")
        raw = [math.exp(x - m) for x in logs]
        denom = math.fsum(raw)
        filtered = tuple(x / denom for x in raw)
        self.last_log_score = m + math.log(denom)
        self._predictive = tuple(
            (1 - self.hazard) * p + self.hazard * initial
            for p, initial in zip(filtered, self.initial_prior))
        self.hands_seen += 1
        return self.posterior


@dataclass(frozen=True)
class SafetyCertificate:
    weight: float
    baseline_worst: float
    exploit_worst: float
    certified_lower_bound: float
    allowed_floor: float


def safe_weight(baseline, exploit, seat: int, budget: float,
                baseline_worst: float | None = None) -> SafetyCertificate:
    """Safe EX-ANTE mixture: sample baseline vs exploit once per whole hand.

    For any opponent tau, V((1-w)*base + w*BR, tau) is linear in w.
    Therefore min_tau V(mixture,tau) >= (1-w)*min V(base,tau)
                                   + w*min V(BR,tau).
    Choosing w from this conservative bound guarantees floor=base_worst-budget.
    This does NOT hold for arbitrary per-information-set behavioral mixing.
    """
    if seat not in (0, 1) or not math.isfinite(budget) or budget < 0:
        raise ValueError("Invalid seat or budget")
    if baseline_worst is None:
        baseline_worst = -best_response(baseline, 1 - seat)[0]
    exploit_worst = -best_response(exploit, 1 - seat)[0]
    gap = max(0.0, baseline_worst - exploit_worst)
    weight = min(1.0, budget / gap) if gap > 1e-12 else 1.0
    certified = (1 - weight) * baseline_worst + weight * exploit_worst
    floor = baseline_worst - budget
    if certified < floor - 1e-9:
        raise AssertionError("Invalid safety certificate")
    return SafetyCertificate(weight, baseline_worst, exploit_worst,
                             certified, floor)


class DecisionAgent:
    """One strategy per whole hand. Never observes opponent hidden cards.

    mode: cfr, frozen, adaptive, switching, safe, safe_exact.
    safe samples the entire baseline or BR policy before the first action.
    """
    def __init__(self, mode, seat, baseline, models, *, hazard=0.1, budget=0.1):
        if mode not in ("cfr", "frozen", "adaptive", "switching", "safe", "safe_exact"):
            raise ValueError("Unknown mode")
        if seat not in (0, 1):
            raise ValueError("Invalid seat")
        self.mode, self.seat, self.baseline = mode, seat, baseline
        self.models = models
        self.budget = budget
        self.beliefs = (BayesianOpponentModel(models) if mode == "adaptive"
                        else SwitchingBeliefs(models, hazard) if mode in ("switching", "safe", "safe_exact")
                        else None)
        self.frozen_policy = (best_response_mixture(tuple(models.values()),
                              tuple(1 / len(models) for _ in models), seat)[1]
                              if mode == "frozen" else None)
        self.base_worst = (-best_response(baseline, 1 - seat)[0]
                           if mode in ("safe", "safe_exact") else None)
        self.policy = None
        self._selected = None
        self.weight = None
        self.certificate = None
        self._ev_cache = {}

    def prepare(self):
        """Call at hand boundary, before observing current hand."""
        self._selected = None
        self.certificate = None
        if self.mode == "cfr":
            self.policy, self.weight = self.baseline, 0.0
        elif self.mode == "frozen":
            self.policy, self.weight = self.frozen_policy, 1.0
        else:
            self.policy = best_response_mixture(
                self.beliefs.policies, self.beliefs.posterior, self.seat)[1]
            if self.mode in ("safe", "safe_exact"):
                if self.mode == "safe_exact":
                    from .qt_safe_exact import exact_safe_weight
                    self.certificate = exact_safe_weight(
                        self.baseline, self.policy, self.seat, self.budget,
                        baseline_worst=self.base_worst)
                else:
                    self.certificate = safe_weight(
                        self.baseline, self.policy, self.seat, self.budget,
                        baseline_worst=self.base_worst)
                self.weight = self.certificate.weight
            else:
                self.weight = 1.0

    def choose(self, observation, rng):
        if self.policy is None:
            raise RuntimeError("Call prepare before the hand")
        if self._selected is None:
            self._selected = (self.baseline if self.mode in ("safe", "safe_exact")
                              and rng.random() >= self.weight else self.policy)
        return PolicyAgent(self._selected).choose(observation, rng)

    def finish_hand(self, terminal):
        if self.beliefs is not None:
            self.beliefs.update(terminal)
        self.policy = None
        self._selected = None

    def exact_expected_value(self, opponent):
        """Oracle evaluation only: NEVER feed this into choose/prepare."""
        if self.policy is None:
            raise RuntimeError("Call prepare before evaluation")
        def v(p):
            # Cache fixed-policy oracle values only, never adaptive policies.
            key = id(opponent)
            if (p is self.baseline or self.mode == "frozen") and key in self._ev_cache:
                return self._ev_cache[key]
            value = (profile_value(p, opponent) if self.seat == 0
                     else -profile_value(opponent, p))
            if p is self.baseline or self.mode == "frozen":
                self._ev_cache[key] = value
            return value
        if self.mode in ("safe", "safe_exact"):
            return ((1 - self.weight) * v(self.baseline)
                    + self.weight * v(self.policy))
        return v(self.policy)


def _mean(xs):
    return statistics.fmean(xs)


def _paired_ci(a, b, seed, resamples=1000):
    if len(a) != len(b) or len(a) < 2:
        raise ValueError("Paired CI requires >=2 independent replicate blocks")
    diffs = [x - y for x, y in zip(a, b)]
    rng = random.Random(seed)
    draws = sorted(_mean(rng.choices(diffs, k=len(diffs)))
                   for _ in range(resamples))
    return {"mean": _mean(diffs), "ci95": [draws[int(.025 * resamples)],
                                           draws[int(.975 * resamples)]]}


def run_benchmark(*, scenario, hands=12, replicates=6, seed=20261014,
                  hazard=0.1, budget=0.1, strategy="strategy.json",
                  include_oracle=True):
    """Predeclared scenarios; replicate is the unit of statistical independence.

    The oracle's exact expected payoff is evaluated using the *true simulator*
    opponent policy but is NEVER used by the agents for decision-making.
    """
    if scenario not in ("stationary", "heldout", "switch"):
        raise ValueError("Unknown scenario")
    if hands < 2 or replicates < 2:
        raise ValueError("Need >=2 hands and independent replicates")
    if not 0 <= hazard <= 1 or budget < 0:
        raise ValueError("Invalid hyperparameters")
    baseline = TabularPolicy.load(strategy)
    models = example_models(baseline)
    opponent_a = models["calling"] if scenario != "heldout" else HELD_OUT_POLICIES["rank_selective"](baseline)
    opponent_b = models["aggressive"] if scenario == "switch" else opponent_a
    switch_at = hands // 2 if scenario == "switch" else None
    arms = ("cfr", "frozen", "adaptive", "switching", "safe", "safe_exact")
    rows = []
    blocks = {arm: [] for arm in arms}
    oracle_blocks = {arm: [] for arm in arms}
    phase_blocks = {phase: {arm: [] for arm in arms} for phase in ("pre", "post")}
    phase_oracle_blocks = {phase: {arm: [] for arm in arms} for phase in ("pre", "post")}
    safe_certificates = []
    rng = random.Random(seed)
    start = time.monotonic()

    for replicate in range(replicates):
        by_arm = {arm: [] for arm in arms}
        by_arm_oracle = {arm: [] for arm in arms}
        by_phase = {p: {arm: [] for arm in arms} for p in phase_blocks}
        by_phase_oracle = {p: {arm: [] for arm in arms} for p in phase_blocks}
        for seat in (0, 1):
            agents = {arm: DecisionAgent(arm, seat, baseline, models,
                                        hazard=hazard, budget=budget) for arm in arms}
            for hand in range(hands):
                hand_seed = rng.randrange(1 << 63)
                phase = "post" if switch_at is not None and hand >= switch_at else "pre"
                opponent = opponent_b if phase == "post" else opponent_a
                for arm in arms:
                    agent = agents[arm]
                    prior = (list(agent.beliefs.posterior)
                             if agent.beliefs is not None else None)
                    agent.prepare()
                    oracle = agent.exact_expected_value(opponent) if include_oracle else None
                    weight = agent.weight
                    if agent.certificate is not None:
                        safe_certificates.append({
                            "replicate": replicate, "seat": seat, "hand": hand,
                            "mode": arm, "weight": weight,
                            "certified_lower_bound": agent.certificate.certified_lower_bound,
                            "allowed_floor": agent.certificate.allowed_floor})
                    players = ((agent, PolicyAgent(opponent)) if seat == 0
                               else (PolicyAgent(opponent), agent))
                    terminal = run_hand(hand_seed, players)
                    reward = terminal.rewards[seat]
                    agent.finish_hand(terminal.observe(seat))
                    by_arm[arm].append(reward)
                    by_phase[phase][arm].append(reward)
                    if oracle is not None:
                        by_arm_oracle[arm].append(oracle)
                        by_phase_oracle[phase][arm].append(oracle)
                    rows.append({"replicate": replicate, "seat": seat,
                                 "hand": hand, "phase": phase, "arm": arm,
                                 "reward": reward, "oracle_ev": oracle,
                                 "weight": weight, "prior": prior})
        for arm in arms:
            blocks[arm].append(_mean(by_arm[arm]))
            if include_oracle:
                oracle_blocks[arm].append(_mean(by_arm_oracle[arm]))
        for phase in phase_blocks:
            if by_phase[phase]["cfr"]:
                for arm in arms:
                    phase_blocks[phase][arm].append(_mean(by_phase[phase][arm]))
                    if include_oracle:
                        phase_oracle_blocks[phase][arm].append(
                            _mean(by_phase_oracle[phase][arm]))

    paired = {f"{arm}_minus_{control}": _paired_ci(
        blocks[arm], blocks[control], seed + 19)
        for arm, control in (("switching", "adaptive"), ("safe", "adaptive"),
                             ("safe", "switching"), ("safe", "cfr"),
                             ("safe_exact", "safe"), ("safe_exact", "adaptive"),
                             ("safe_exact", "cfr"))}
    oracle_paired = ({f"{arm}_minus_{control}": _paired_ci(
        oracle_blocks[arm], oracle_blocks[control], seed + 29)
        for arm, control in (("switching", "adaptive"), ("safe", "adaptive"),
                             ("safe", "switching"), ("safe", "cfr"),
                             ("safe_exact", "safe"), ("safe_exact", "adaptive"),
                             ("safe_exact", "cfr"))}
                     if include_oracle else {})
    phase_means = {p: {arm: _mean(values)
                       for arm, values in group.items() if values}
                   for p, group in phase_blocks.items()}
    result = {
        "protocol": {"scenario": scenario, "hands_per_seat": hands,
                     "replicates": replicates, "seed": seed,
                     "hazard": hazard, "risk_budget": budget,
                     "arms": list(arms), "switch_after": switch_at,
                     "oracle_ex_ante": include_oracle,
                     "independent_unit": "replicate averaged over both seats",
                     "note": "Synthetic Leduc opponent models; not financial returns."},
        "mean_realized_chips_per_hand": {arm: _mean(v) for arm, v in blocks.items()},
        "paired_realized": paired,
        "mean_exact_conditional_ev": ({arm: _mean(v) for arm, v in oracle_blocks.items()}
                                      if include_oracle else None),
        "paired_exact_conditional_ev": oracle_paired,
        "phase_mean_realized": phase_means,
        "phase_mean_exact_ev": ({p: {arm: _mean(vals) for arm, vals in group.items() if vals}
                                  for p, group in phase_oracle_blocks.items()}
                                 if include_oracle else None),
        "safe_certificate": {
            "n": len(safe_certificates),
            "min_margin": min((x["certified_lower_bound"] - x["allowed_floor"]
                               for x in safe_certificates), default=None),
            "mean_weight": (_mean([x["weight"] for x in safe_certificates])
                            if safe_certificates else None),
            "by_mode": {arm: {
                "n": len([x for x in safe_certificates if x["mode"] == arm]),
                "mean_weight": _mean([x["weight"] for x in safe_certificates
                                      if x["mode"] == arm]),
                "min_margin": min(x["certified_lower_bound"] - x["allowed_floor"]
                                  for x in safe_certificates if x["mode"] == arm)
            } for arm in ("safe", "safe_exact")}},
        "runtime_seconds": round(time.monotonic() - start, 3),
        "rows": rows,
    }
    return result


def main():
    parser = argparse.ArgumentParser(description="QT V4 real-Leduc benchmark")
    parser.add_argument("--scenario", choices=("stationary", "heldout", "switch"), default="switch")
    parser.add_argument("--hands", type=int, default=12)
    parser.add_argument("--replicates", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20261014)
    parser.add_argument("--hazard", type=float, default=0.1)
    parser.add_argument("--budget", type=float, default=0.1)
    parser.add_argument("--strategy", default="strategy.json")
    parser.add_argument("--output", default=None)
    parser.add_argument("--no-oracle", action="store_true")
    args = parser.parse_args()
    result = run_benchmark(scenario=args.scenario, hands=args.hands,
                           replicates=args.replicates, seed=args.seed,
                           hazard=args.hazard, budget=args.budget,
                           strategy=args.strategy, include_oracle=not args.no_oracle)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    summary = {k: v for k, v in result.items() if k != "rows"}
    print(json.dumps(summary, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
