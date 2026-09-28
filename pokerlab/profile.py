"""Reproducible workload and deterministic-result checks for CPU profiling.

Elapsed times depend on the machine and include cProfile overhead. The digest
covers only the underlying experiment output, which should not depend on the
profiler. No timings are used to assert poker performance.
"""

from __future__ import annotations

import argparse
import cProfile
import hashlib
import json
import platform
import pstats
import sys
import time
from pathlib import Path

from .bayes import example_models
from .cfr import Policy, TabularPolicy
from .experiment import RankSelectivePolicy, run_experiment

OPPONENTS = ("baseline", "folding", "calling", "aggressive", "rank_selective")
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def result_digest(result: dict) -> str:
    """Hash the entire deterministic experiment, excluding profile timings."""
    canonical = json.dumps(result, sort_keys=True, separators=(",", ":"),
                           allow_nan=False).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _display_file(filename: str) -> str:
    path = Path(filename)
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return path.name


def profile_experiment(baseline: Policy, opponent: Policy,
                       models: dict[str, Policy], *, hands: int,
                       replicates: int, seed: int,
                       switch_to: Policy | None = None,
                       switch_after: int | None = None,
                       top: int = 15) -> dict:
    """Profile a cold full experiment, with the same inputs as run_experiment.

    Strategy loading and report serialization are outside the timed region.
    The function returns no hand trace; its digest can be compared with a
    separately executed experiment using the same inputs.
    """
    if top < 1:
        raise ValueError("top must be positive")
    profiler = cProfile.Profile()
    start = time.perf_counter()
    profiler.enable()
    try:
        result = run_experiment(baseline, opponent, models,
                                hands=hands, replicates=replicates, seed=seed,
                                switch_to=switch_to, switch_after=switch_after)
    finally:
        profiler.disable()
    elapsed = time.perf_counter() - start
    stats = pstats.Stats(profiler)
    functions = [{"file": _display_file(filename), "line": line,
                  "function": function, "calls": calls,
                  "primitive_calls": primitive_calls,
                  "self_seconds": self_time,
                  "cumulative_seconds": cumulative_time}
                 for (filename, line, function),
                 (primitive_calls, calls, self_time, cumulative_time, _)
                 in stats.stats.items()]
    functions.sort(key=lambda row: (-row["cumulative_seconds"],
                                    -row["self_seconds"], row["file"],
                                    row["line"], row["function"]))
    return {"workload": result["design"],
            "result_sha256": result_digest(result),
            "mean_reward": result["mean_reward"],
            "paired_contrasts": result["paired_contrasts"],
            "profile": {"elapsed_seconds": elapsed,
                        "total_function_calls": stats.total_calls,
                        "primitive_function_calls": stats.prim_calls,
                        "top_by_cumulative_seconds": functions[:top]}}


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile a paired Leduc experiment")
    parser.add_argument("--strategy", type=Path, default=Path("strategy.json"))
    parser.add_argument("--rules", type=Path, default=Path("rules.md"))
    parser.add_argument("--opponent", choices=OPPONENTS, default="calling")
    parser.add_argument("--switch-to", choices=OPPONENTS)
    parser.add_argument("--switch-after", type=int)
    parser.add_argument("--hands", type=int, default=12)
    parser.add_argument("--replicates", type=int, default=2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--top", type=int, default=15)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if (args.switch_to is None) != (args.switch_after is None):
        parser.error("--switch-to and --switch-after must be specified together")
    if args.switch_to == args.opponent:
        parser.error("--switch-to must differ from --opponent")
    if args.switch_after is not None and not 0 < args.switch_after < args.hands:
        parser.error("--switch-after must be between 1 and --hands - 1")
    if args.hands < 1 or args.replicates < 2 or args.top < 1:
        parser.error("hands and top must be positive; replicates at least two")

    baseline = TabularPolicy.load(args.strategy)
    models = example_models(baseline)

    def policy(name: str) -> Policy:
        return RankSelectivePolicy(baseline) if name == "rank_selective" else models[name]

    report = profile_experiment(baseline, policy(args.opponent), models,
                                hands=args.hands, replicates=args.replicates,
                                seed=args.seed,
                                switch_to=policy(args.switch_to) if args.switch_to else None,
                                switch_after=args.switch_after, top=args.top)
    report["workload"].update({
        "opponent": args.opponent, "switch_to": args.switch_to,
        "strategy_sha256": hashlib.sha256(args.strategy.read_bytes()).hexdigest(),
        "rules_sha256": hashlib.sha256(args.rules.read_bytes()).hexdigest(),
    })
    report["runtime"] = {"python": platform.python_version(),
                         "implementation": platform.python_implementation(),
                         "platform": platform.platform()}
    data = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output is None:
        print(data, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(data)
        print(f"saved={args.output}")
    top = report["profile"]["top_by_cumulative_seconds"]
    nested = next((row for row in top if row["function"] != "run_experiment"),
                  top[0])
    print(f"elapsed_seconds={report['profile']['elapsed_seconds']:.3f} "
          f"top_nested_cumulative={nested['file']}:{nested['line']}:{nested['function']}",
          file=sys.stderr)


if __name__ == "__main__":
    main()
