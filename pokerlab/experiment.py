"""Paired, seat-balanced experiments for stationary or switching opponents.

The fixed-prior control optimizes against the *same initial mixture* as the
learner. Its only difference is that it never uses observations from earlier
hands to change its policy. All three arms reuse each hand's chance seed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

from .bayes import (ActionBiasPolicy, AdaptiveAgent, BayesianOpponentModel,
                    example_models)
from .cfr import Policy, PolicyAgent, TabularPolicy, best_response_mixture
from .game import Action, PlayerObservation
from .simulate import run_hand

ARMS = ("learning", "frozen_prior", "cfr")


@dataclass(frozen=True)
class RankSelectivePolicy:
    """Prespecified opponent outside the four stationary candidate types.

    This type changes its betting bias with its private card and public pair.
    It is an illustrative assumption, not a fitted human behavior model.
    """

    base: Policy

    def probabilities(self, observation: PlayerObservation) -> tuple[float, ...]:
        if (observation.public_card is not None
                and observation.own_card.rank == observation.public_card.rank):
            factors = {Action.BET: 4.0, Action.RAISE: 4.0,
                       Action.CALL: 2.0, Action.FOLD: 0.2}
        elif observation.public_card is None and observation.own_card.rank == "K":
            factors = {Action.BET: 3.0, Action.RAISE: 3.0,
                       Action.CALL: 1.5, Action.FOLD: 0.3}
        elif observation.own_card.rank == "J":
            factors = {Action.BET: 0.5, Action.RAISE: 0.5,
                       Action.CALL: 0.6, Action.FOLD: 3.5}
        else:
            factors = {}
        return ActionBiasPolicy(self.base, factors).probabilities(observation)


def _mean(values: list[float]) -> float:
    return statistics.fmean(values)


def _paired_interval(differences: list[float], seed: int,
                     resamples: int = 2000) -> tuple[float, float]:
    """Percentile bootstrap over independent, two-seat match replicates."""
    if len(differences) < 2:
        raise ValueError("At least two replicates are needed for an interval")
    rng = random.Random(seed)
    draws = sorted(_mean(rng.choices(differences, k=len(differences)))
                   for _ in range(resamples))
    return draws[int(.025 * resamples)], draws[int(.975 * resamples)]


def _summarize(blocks: list[dict[str, float]], seed: int) -> dict:
    means = {arm: _mean([block[arm] for block in blocks]) for arm in ARMS}
    contrasts = {}
    for control in ("frozen_prior", "cfr"):
        differences = [block["learning"] - block[control]
                       for block in blocks]
        low, high = _paired_interval(differences, seed)
        contrasts[f"learning_minus_{control}"] = {
            "mean": _mean(differences), "bootstrap_95_percent": [low, high],
        }
    return {"mean_reward": means, "paired_contrasts": contrasts,
            "replicate_means": blocks}


def run_experiment(baseline: Policy, opponent: Policy,
                   models: dict[str, Policy], *, hands: int,
                   replicates: int, seed: int, switch_to: Policy | None = None,
                   switch_after: int | None = None) -> dict:
    """Compare learning, fixed mixture response and CFR on shared deals.

    A replicate contains one match at each seat, with fresh learner beliefs at
    the start of each match. Seeds are shared across arms within a seat/hand,
    but distinct between seats and replicates. If requested, the opponent
    changes policy at hand index switch_after, before that hand is dealt.
    Learner beliefs persist across the switch; the learner is not told its
    position. Only terminal player observations reach Bayesian updates.
    """
    if hands < 1 or replicates < 2:
        raise ValueError("hands must be positive and replicates at least two")
    if (switch_to is None) != (switch_after is None):
        raise ValueError("switch_to and switch_after must be specified together")
    if switch_after is not None and not 0 < switch_after < hands:
        raise ValueError("switch_after must be between 1 and hands - 1")
    if not models:
        raise ValueError("At least one candidate model is required")
    names = tuple(models)
    policies = tuple(models.values())
    prior = (1 / len(policies),) * len(policies)
    frozen = {seat: PolicyAgent(best_response_mixture(policies, prior, seat)[1])
              for seat in (0, 1)}
    cfr = PolicyAgent(baseline)
    seed_rng = random.Random(seed)
    rows: list[dict] = []
    blocks: list[dict[str, float]] = []
    phases = ({"pre_switch": switch_after,
               "post_switch": hands - switch_after} if switch_after is not None
              else {"stationary": hands})
    phase_blocks: dict[str, list[dict[str, float]]] = {phase: [] for phase in phases}

    for replicate in range(replicates):
        seat_rewards: dict[str, list[float]] = {arm: [] for arm in ARMS}
        phase_seat_rewards: dict[str, dict[str, list[float]]] = {
            phase: {arm: [] for arm in ARMS} for phase in phases}
        for seat in (0, 1):
            learner = AdaptiveAgent(seat, BayesianOpponentModel(models))
            seat_totals = {arm: 0.0 for arm in ARMS}
            phase_totals = {phase: {arm: 0.0 for arm in ARMS} for phase in phases}
            for hand in range(hands):
                hand_seed = seed_rng.randrange(1 << 63)
                if switch_after is None:
                    phase = "stationary"
                else:
                    phase = "pre_switch" if hand < switch_after else "post_switch"
                current_opponent = (switch_to if phase == "post_switch" else opponent)
                posterior_before = learner.beliefs.named_posterior()
                rewards: dict[str, float] = {}
                for arm, agent in (("learning", learner),
                                   ("frozen_prior", frozen[seat]),
                                   ("cfr", cfr)):
                    players = ((agent, PolicyAgent(current_opponent))
                               if seat == 0 else (PolicyAgent(current_opponent), agent))
                    terminal = run_hand(hand_seed, players)
                    assert terminal.rewards is not None
                    rewards[arm] = terminal.rewards[seat]
                    seat_totals[arm] += rewards[arm]
                    phase_totals[phase][arm] += rewards[arm]
                    if arm == "learning":
                        learner.finish_hand(terminal.observe(seat))
                rows.append({"replicate": replicate, "seat": seat,
                             "hand": hand, "seed": hand_seed, "phase": phase,
                             "rewards": rewards,
                             "posterior_before": posterior_before,
                             "posterior": learner.beliefs.named_posterior()})
            for arm in ARMS:
                seat_rewards[arm].append(seat_totals[arm] / hands)
                for phase, count in phases.items():
                    phase_seat_rewards[phase][arm].append(
                        phase_totals[phase][arm] / count)
        blocks.append({arm: _mean(seat_rewards[arm]) for arm in ARMS})
        for phase in phases:
            phase_blocks[phase].append({
                arm: _mean(phase_seat_rewards[phase][arm]) for arm in ARMS})

    result = _summarize(blocks, seed + 1)
    result.update({"design": {"hands_per_seat": hands, "replicates": replicates,
                              "seat_order": [0, 1], "seed": seed,
                              "candidate_types": list(names),
                              "initial_prior": list(prior),
                              "switch_after": switch_after,
                              "bootstrap_resamples": 2000,
                              "unit": "net chips per hand, averaged across seats"},
                   "phases": {phase: {"hands_per_seat": count,
                                      **_summarize(phase_blocks[phase], seed + 1)}
                              for phase, count in phases.items()},
                   "hands": rows})
    if switch_after is not None:
        change = {}
        for control in ("frozen_prior", "cfr"):
            key = f"learning_minus_{control}"
            differences = [
                (post["learning"] - post[control])
                - (pre["learning"] - pre[control])
                for pre, post in zip(phase_blocks["pre_switch"],
                                     phase_blocks["post_switch"])]
            low, high = _paired_interval(differences, seed + 1)
            change[key] = {"mean": _mean(differences),
                           "bootstrap_95_percent": [low, high]}
        result["change_in_paired_contrasts"] = change
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Paired Leduc opponent experiment")
    parser.add_argument("--strategy", type=Path, default=Path("strategy.json"))
    parser.add_argument("--rules", type=Path, default=Path("rules.md"))
    parser.add_argument("--opponent", choices=("baseline", "folding", "calling",
                                                "aggressive", "rank_selective"),
                        default="calling")
    parser.add_argument("--switch-to", choices=("baseline", "folding", "calling",
                                                "aggressive", "rank_selective"))
    parser.add_argument("--switch-after", type=int)
    parser.add_argument("--hands", type=int, default=12)
    parser.add_argument("--replicates", type=int, default=2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if (args.switch_to is None) != (args.switch_after is None):
        parser.error("--switch-to and --switch-after must be specified together")
    if args.switch_to is not None and args.switch_to == args.opponent:
        parser.error("--switch-to must differ from --opponent")
    if args.switch_after is not None and not 0 < args.switch_after < args.hands:
        parser.error("--switch-after must be between 1 and --hands - 1")
    baseline = TabularPolicy.load(args.strategy)
    models = example_models(baseline)

    def selected_policy(name: str) -> Policy:
        return RankSelectivePolicy(baseline) if name == "rank_selective" else models[name]

    opponent = selected_policy(args.opponent)
    switch_to = selected_policy(args.switch_to) if args.switch_to else None
    result = run_experiment(baseline, opponent, models,
                            hands=args.hands, replicates=args.replicates,
                            seed=args.seed, switch_to=switch_to,
                            switch_after=args.switch_after)
    result["design"].update({
        "opponent": args.opponent,
        "switch_to": args.switch_to,
        "strategy_sha256": hashlib.sha256(args.strategy.read_bytes()).hexdigest(),
        "rules_sha256": hashlib.sha256(args.rules.read_bytes()).hexdigest(),
    })
    data = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(data, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(data)
        print(f"saved={args.output}")
    for contrast, summary in result["paired_contrasts"].items():
        print(f"{contrast}={summary['mean']:+.4f} "
              f"bootstrap_95_percent={summary['bootstrap_95_percent']}",
              file=sys.stderr)


if __name__ == "__main__":
    main()
