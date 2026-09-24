"""Independent rollout check of the exact strategy-profile evaluator.

The reference is the hand simulator with independent card and action streams.
This checks chance weighting and action probabilities in ``profile_value``;
both paths still share the game engine, so it is not an external rules audit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

from .bayes import example_models
from .cfr import Policy, PolicyAgent, TabularPolicy, UniformPolicy, profile_value
from .simulate import run_hand


def validate_profile(player0: Policy, player1: Policy, *,
                     hands: int, seed: int) -> dict[str, float | int | bool]:
    """Compare exact expectation to independently sampled completed hands.

    The four-standard-error threshold is a diagnostic, not proof of equality.
    A nonzero gap can be sampling noise; reproduce a failure with more hands.
    """
    if hands < 2:
        raise ValueError("hands must be at least two to estimate uncertainty")
    expected = profile_value(player0, player1)
    players = (PolicyAgent(player0), PolicyAgent(player1))
    rewards = []
    for hand in range(hands):
        terminal = run_hand(seed + hand, players)
        assert terminal.rewards is not None
        rewards.append(terminal.rewards[0])
    observed = statistics.fmean(rewards)
    se = statistics.stdev(rewards) / math.sqrt(hands)
    error = observed - expected
    return {"hands": hands, "seed": seed, "exact_value_p0": expected,
            "sample_mean_p0": observed, "sample_standard_error": se,
            "sample_minus_exact": error,
            "within_four_standard_errors": abs(error) <= 4 * se + 1e-12}


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-check exact Leduc value with seeded hands")
    parser.add_argument("--strategy", type=Path, default=Path("strategy.json"))
    parser.add_argument("--rules", type=Path, default=Path("rules.md"))
    choices = ("baseline", "folding", "calling", "aggressive", "uniform")
    parser.add_argument("--player0", choices=choices, default="baseline")
    parser.add_argument("--player1", choices=choices, default="calling")
    parser.add_argument("--hands", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    baseline = TabularPolicy.load(args.strategy)
    policies: dict[str, Policy] = {**example_models(baseline), "uniform": UniformPolicy()}
    result = validate_profile(policies[args.player0], policies[args.player1],
                              hands=args.hands, seed=args.seed)
    result.update({"player0": args.player0, "player1": args.player1,
                   "strategy_sha256": hashlib.sha256(args.strategy.read_bytes()).hexdigest(),
                   "rules_sha256": hashlib.sha256(args.rules.read_bytes()).hexdigest()})
    data = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(data, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(data)
        print(f"saved={args.output}")
    if not result["within_four_standard_errors"]:
        parser.exit(1, "Rollout mean differs from exact value by more than four standard errors.\n")


if __name__ == "__main__":
    main()
