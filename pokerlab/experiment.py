"""Paired, seat-balanced experiments for stationary Leduc opponents.

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
from pathlib import Path

from .bayes import AdaptiveAgent, BayesianOpponentModel, example_models
from .cfr import Policy, PolicyAgent, TabularPolicy, best_response_mixture
from .simulate import run_hand

ARMS = ("learning", "frozen_prior", "cfr")


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


def run_experiment(baseline: Policy, opponent: Policy,
                   models: dict[str, Policy], *, hands: int,
                   replicates: int, seed: int) -> dict:
    """Compare learning, fixed mixture response and CFR on shared deals.

    A replicate contains one match at each seat, with fresh learner beliefs at
    the start of each match. Seeds are shared across arms within a seat/hand,
    but distinct between seats and replicates. The opponent type is fixed for
    the match. Only terminal player observations reach Bayesian updates.
    """
    if hands < 1 or replicates < 2:
        raise ValueError("hands must be positive and replicates at least two")
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

    for replicate in range(replicates):
        seat_rewards: dict[str, list[float]] = {arm: [] for arm in ARMS}
        for seat in (0, 1):
            learner = AdaptiveAgent(seat, BayesianOpponentModel(models))
            seat_totals = {arm: 0.0 for arm in ARMS}
            for hand in range(hands):
                hand_seed = seed_rng.randrange(1 << 63)
                rewards: dict[str, float] = {}
                for arm, agent in (("learning", learner),
                                   ("frozen_prior", frozen[seat]),
                                   ("cfr", cfr)):
                    players = ((agent, PolicyAgent(opponent)) if seat == 0 else
                               (PolicyAgent(opponent), agent))
                    terminal = run_hand(hand_seed, players)
                    assert terminal.rewards is not None
                    rewards[arm] = terminal.rewards[seat]
                    seat_totals[arm] += rewards[arm]
                    if arm == "learning":
                        learner.finish_hand(terminal.observe(seat))
                rows.append({"replicate": replicate, "seat": seat,
                             "hand": hand, "seed": hand_seed,
                             "rewards": rewards,
                             "posterior": learner.beliefs.named_posterior()})
            for arm in ARMS:
                seat_rewards[arm].append(seat_totals[arm] / hands)
        blocks.append({arm: _mean(seat_rewards[arm]) for arm in ARMS})

    means = {arm: _mean([block[arm] for block in blocks]) for arm in ARMS}
    contrasts = {}
    for control in ("frozen_prior", "cfr"):
        differences = [block["learning"] - block[control]
                       for block in blocks]
        low, high = _paired_interval(differences, seed + 1)
        contrasts[f"learning_minus_{control}"] = {
            "mean": _mean(differences), "bootstrap_95_percent": [low, high],
        }
    return {"design": {"hands_per_seat": hands, "replicates": replicates,
                       "seat_order": [0, 1], "seed": seed,
                       "candidate_types": list(names), "initial_prior": list(prior),
                       "bootstrap_resamples": 2000,
                       "unit": "net chips per hand, averaged across seats"},
            "mean_reward": means, "paired_contrasts": contrasts,
            "replicate_means": blocks, "hands": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description="Paired stationary-opponent Leduc experiment")
    parser.add_argument("--strategy", type=Path, default=Path("strategy.json"))
    parser.add_argument("--rules", type=Path, default=Path("rules.md"))
    parser.add_argument("--opponent", choices=("baseline", "folding", "calling",
                                                "aggressive"), default="calling")
    parser.add_argument("--hands", type=int, default=12)
    parser.add_argument("--replicates", type=int, default=2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    baseline = TabularPolicy.load(args.strategy)
    models = example_models(baseline)
    result = run_experiment(baseline, models[args.opponent], models,
                            hands=args.hands, replicates=args.replicates,
                            seed=args.seed)
    result["design"].update({
        "opponent": args.opponent,
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
