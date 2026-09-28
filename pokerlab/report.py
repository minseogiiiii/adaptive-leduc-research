"""Validate saved experiment records and render a descriptive pilot report.

This module reads existing runs. It never reruns or selects experiments based
on their payoffs, and checks every reported point estimate against hand rows.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
from pathlib import Path

from .experiment import ARMS

SCENARIOS = (
    ("Calling (candidate type)", "calling", None, None),
    ("Rank selective (held out)", "rank_selective", None, None),
    ("Calling to aggressive", "calling", "aggressive", 6),
)
CONTROLS = ("frozen_prior", "cfr")


def _equal(observed: float, expected: float, label: str) -> None:
    if not (isinstance(observed, (int, float)) and math.isfinite(observed)
            and math.isclose(observed, expected, rel_tol=1e-12, abs_tol=1e-12)):
        raise ValueError(f"Inconsistent {label}: {observed!r} != {expected!r}")


def _interval(values: list[float], seed: int) -> tuple[float, float]:
    rng = random.Random(seed)
    draws = sorted(statistics.fmean(rng.choices(values, k=len(values)))
                   for _ in range(2000))
    return draws[50], draws[1950]


def _validate_summary(summary: dict, rows: list[dict], hands: int,
                      replicates: int, seed: int, label: str) -> None:
    blocks = []
    for replicate in range(replicates):
        block = {}
        for arm in ARMS:
            by_seat = [statistics.fmean(
                row["rewards"][arm] for row in rows
                if row["replicate"] == replicate and row["seat"] == seat)
                for seat in (0, 1)]
            block[arm] = statistics.fmean(by_seat)
        blocks.append(block)
    if len(rows) != hands * replicates * 2:
        raise ValueError(f"Wrong number of rows for {label}")
    if len(summary["replicate_means"]) != replicates:
        raise ValueError(f"Wrong number of replicate means for {label}")
    for index, block in enumerate(blocks):
        for arm in ARMS:
            _equal(summary["replicate_means"][index][arm], block[arm],
                   f"{label} replicate {index} {arm}")
            _equal(summary["mean_reward"][arm],
                   statistics.fmean(b[arm] for b in blocks), f"{label} {arm}")
    for control in CONTROLS:
        key = f"learning_minus_{control}"
        differences = [block["learning"] - block[control] for block in blocks]
        _equal(summary["paired_contrasts"][key]["mean"],
               statistics.fmean(differences), f"{label} {key}")
        low, high = summary["paired_contrasts"][key]["bootstrap_95_percent"]
        if not (math.isfinite(low) and math.isfinite(high) and low <= high):
            raise ValueError(f"Invalid interval for {label} {key}")
        expected_low, expected_high = _interval(differences, seed + 1)
        _equal(low, expected_low, f"{label} {key} interval lower")
        _equal(high, expected_high, f"{label} {key} interval upper")


def validate_result(result: dict, *, strategy_sha256: str,
                    rules_sha256: str, opponent: str,
                    switch_to: str | None, switch_after: int | None) -> None:
    """Check provenance, row coverage, phases and all published point estimates."""
    design = result["design"]
    for name, expected in (("strategy_sha256", strategy_sha256),
                           ("rules_sha256", rules_sha256),
                           ("opponent", opponent), ("switch_to", switch_to),
                           ("switch_after", switch_after),
                           ("seat_order", [0, 1])):
        if design[name] != expected:
            raise ValueError(f"Unexpected design {name}: {design[name]!r}")
    hands, replicates = design["hands_per_seat"], design["replicates"]
    seed = design["seed"]
    if not (type(hands) is int and hands >= 2 and type(replicates) is int
            and replicates >= 2):
        raise ValueError("Pilot needs at least two hands and two replicates")
    if switch_after is not None and not 0 < switch_after < hands:
        raise ValueError("Switch must leave both phases nonempty")
    rows = result["hands"]
    expected_keys = {(replicate, seat, hand)
                     for replicate in range(replicates)
                     for seat in (0, 1) for hand in range(hands)}
    keys = [(row["replicate"], row["seat"], row["hand"]) for row in rows]
    if len(keys) != len(expected_keys) or set(keys) != expected_keys:
        raise ValueError("Missing or duplicate replicate/seat/hand row")
    by_key = {key: row for key, row in zip(keys, rows)}
    seeds = {row["seed"] for row in rows}
    if len(seeds) != len(rows):
        raise ValueError("Hand seeds repeat")
    for row in rows:
        phase = ("stationary" if switch_after is None else
                 "pre_switch" if row["hand"] < switch_after else "post_switch")
        if row["phase"] != phase:
            raise ValueError("Hand phase disagrees with switch schedule")
        for arm in ARMS:
            if not math.isfinite(row["rewards"][arm]):
                raise ValueError("Nonfinite hand reward")
        for key in ("posterior_before", "posterior"):
            probabilities = row[key]
            if (set(probabilities) != set(design["candidate_types"])
                    or any(not math.isfinite(p) or p < 0 or p > 1
                           for p in probabilities.values())):
                raise ValueError(f"Invalid {key}")
            _equal(sum(probabilities.values()), 1.0, key)
        previous = (dict(zip(design["candidate_types"], design["initial_prior"]))
                    if row["hand"] == 0 else
                    by_key[(row["replicate"], row["seat"], row["hand"] - 1)]["posterior"])
        for name, probability in previous.items():
            _equal(row["posterior_before"][name], probability,
                   f"posterior continuity {name}")
    _validate_summary(result, rows, hands, replicates, seed, "overall")
    phase_counts = ({"stationary": hands} if switch_after is None else
                    {"pre_switch": switch_after,
                     "post_switch": hands - switch_after})
    if set(result["phases"]) != set(phase_counts):
        raise ValueError("Unexpected phases")
    for phase, count in phase_counts.items():
        summary = result["phases"][phase]
        if summary["hands_per_seat"] != count:
            raise ValueError(f"Wrong length of {phase}")
        _validate_summary(summary, [row for row in rows if row["phase"] == phase],
                          count, replicates, seed, phase)
    if switch_after is not None:
        for control in CONTROLS:
            key = f"learning_minus_{control}"
            pre = result["phases"]["pre_switch"]["replicate_means"]
            post = result["phases"]["post_switch"]["replicate_means"]
            differences = [
                (b["learning"] - b[control]) - (a["learning"] - a[control])
                for a, b in zip(pre, post)]
            change = result["change_in_paired_contrasts"][key]
            _equal(change["mean"], statistics.fmean(differences),
                   f"phase change {key}")
            low, high = change["bootstrap_95_percent"]
            if not (math.isfinite(low) and math.isfinite(high) and low <= high):
                raise ValueError(f"Invalid phase change interval for {key}")
            expected_low, expected_high = _interval(differences, seed + 1)
            _equal(low, expected_low, f"phase change {key} interval lower")
            _equal(high, expected_high, f"phase change {key} interval upper")


def _metric(summary: dict, control: str) -> str:
    contrast = summary["paired_contrasts"][f"learning_minus_{control}"]
    low, high = contrast["bootstrap_95_percent"]
    return f"{contrast['mean']:+.4f} [{low:+.4f}, {high:+.4f}]"


def render_report(paths: tuple[Path, Path, Path], *, strategy: Path,
                  rules: Path) -> str:
    """Render fixed, prespecified scenarios after validating their raw rows."""
    strategy_hash = hashlib.sha256(strategy.read_bytes()).hexdigest()
    rules_hash = hashlib.sha256(rules.read_bytes()).hexdigest()
    records = []
    for path, (label, opponent, switch_to, switch_after) in zip(paths, SCENARIOS):
        raw = path.read_bytes()
        result = json.loads(raw)
        validate_result(result, strategy_sha256=strategy_hash,
                        rules_sha256=rules_hash, opponent=opponent,
                        switch_to=switch_to, switch_after=switch_after)
        records.append((label, path, hashlib.sha256(raw).hexdigest(), result))
    designs = [record[3]["design"] for record in records]
    for name in ("hands_per_seat", "replicates", "seed", "candidate_types",
                 "initial_prior", "bootstrap_resamples", "seat_order"):
        if any(design[name] != designs[0][name] for design in designs[1:]):
            raise ValueError(f"Scenarios have different {name}")
    design = designs[0]
    if (design["hands_per_seat"], design["replicates"], design["seed"],
            design["bootstrap_resamples"]) != (12, 10, 7, 2000):
        raise ValueError("Unexpected prespecified pilot workload")
    if (design["candidate_types"] != ["baseline", "folding", "calling", "aggressive"]
            or design["initial_prior"] != [0.25] * 4):
        raise ValueError("Unexpected candidate models or initial prior")
    reference_seeds = {(row["replicate"], row["seat"], row["hand"]): row["seed"]
                       for row in records[0][3]["hands"]}
    for _, _, _, result in records[1:]:
        if any(reference_seeds[(row["replicate"], row["seat"], row["hand"])]
               != row["seed"] for row in result["hands"]):
            raise ValueError("Scenarios use different hand seeds")
    calling_prefix = {(row["replicate"], row["seat"], row["hand"]): row
                      for row in records[0][3]["hands"] if row["hand"] < 6}
    for row in records[2][3]["hands"]:
        if row["hand"] < 6:
            key = (row["replicate"], row["seat"], row["hand"])
            expected = calling_prefix[key]
            if any(row[field] != expected[field] for field in
                   ("seed", "rewards", "posterior_before", "posterior")):
                raise ValueError("Pre-switch hands differ from calling control")
    lines = ["# Leduc opponent-learning pilot report", "",
             "## Research question and fixed design", "",
             "Does updating a posterior over opponent types improve net chips "
             "per hand relative to a policy optimized for the same initial "
             "mixture but never updated? This pilot compares that learning "
             "arm with a frozen-prior response and the 300-iteration CFR "
             "policy. The three conditions were selected before these runs: "
             "an in-model calling opponent, a held-out rank-selective opponent, "
             "and a calling-to-aggressive switch halfway through a match.", "",
             f"Each condition has {design['replicates']} independent replicates. "
             f"A replicate contains {design['hands_per_seat']} hands in each "
             "seat, with a fresh uniform prior for each match. Arms share "
             "deals and action random streams within each hand. The opponent "
             "schedule is shared across arms, while their public histories "
             "may diverge. The learning policy uses its own legal "
             "observations; it is not told the switch point. All conditions "
             f"use seed {design['seed']}, and intervals use "
             f"{design['bootstrap_resamples']} percentile bootstrap resamples "
             "of whole two-seat replicates.", "",
             "## Observed rewards", "",
             "All values are net chips per hand, averaged over seats. A contrast "
             "is learning minus its control; brackets are descriptive 95% "
             "bootstrap intervals over replicates.", "",
             "| Condition | Learning | Frozen prior | CFR | Learning − frozen [interval] | Learning − CFR [interval] |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for label, _, _, result in records:
        rewards = result["mean_reward"]
        lines.append(f"| {label} | {rewards['learning']:+.4f} | "
                     f"{rewards['frozen_prior']:+.4f} | {rewards['cfr']:+.4f} | "
                     f"{_metric(result, 'frozen_prior')} | {_metric(result, 'cfr')} |")
    switch = records[2][3]
    lines.extend(["", "## Policy-switch phases", "",
                  "The switch occurs before hand 6 (zero-based). The change "
                  "compares post-switch minus pre-switch paired contrasts "
                  "within each replicate. It is descriptive, not an estimate "
                  "of a change-point detector's performance.", "",
                  "| Phase | Learning | Frozen prior | CFR | Learning − frozen [interval] | Learning − CFR [interval] |",
                  "| --- | ---: | ---: | ---: | ---: | ---: |"])
    for phase in ("pre_switch", "post_switch"):
        summary = switch["phases"][phase]
        rewards = summary["mean_reward"]
        lines.append(f"| {phase} | {rewards['learning']:+.4f} | "
                     f"{rewards['frozen_prior']:+.4f} | {rewards['cfr']:+.4f} | "
                     f"{_metric(summary, 'frozen_prior')} | "
                     f"{_metric(summary, 'cfr')} |")
    changes = switch["change_in_paired_contrasts"]
    for control in CONTROLS:
        summary = changes[f"learning_minus_{control}"]
        low, high = summary["bootstrap_95_percent"]
        lines.append(f"- Change in learning − {control}: "
                     f"{summary['mean']:+.4f} [{low:+.4f}, {high:+.4f}].")
    lines.extend(["", "## Interpretation and limits", "",
                  "The first contrast isolates the value of between-hand "
                  "posterior updates within this finite model family, since "
                  "the frozen control makes the same initial mixture response. "
                  "The CFR contrast also changes the initial policy, so it "
                  "does not isolate learning. The rank-selective opponent is "
                  "one constructed held-out rule, not a sample of real players. "
                  "The switch violates the learner's fixed-type assumption; "
                  "its posterior is not reset. These seed-dependent outcomes "
                  "are a small pilot with only ten independent replicate units "
                  "per condition. Bootstrap intervals are imprecise at that "
                  "size. Neither phase contrasts nor between-condition gaps "
                  "identify a causal effect of learning speed or establish "
                  "generalization to other opponents. No claim of superior "
                  "play or statistical significance follows from this pilot.", "",
                  "The simulator and exact evaluator share game rules. An "
                  "independent implementation with reconciled betting rules "
                  "remains an external cross-check. Before a substantive "
                  "claim, increase independent replicates, define a power "
                  "target and broader held-out opponent family, and "
                  "prespecify the comparison and stopping rule.", "",
                  "## Reproduction and provenance", "",
                  f"- `strategy.json` SHA-256: `{strategy_hash}`.",
                  f"- `rules.md` SHA-256: `{rules_hash}`."])
    for name in ("game.py", "cfr.py", "bayes.py", "experiment.py", "report.py"):
        digest = hashlib.sha256((Path(__file__).resolve().parent / name).read_bytes())
        lines.append(f"- `pokerlab/{name}` SHA-256: `{digest.hexdigest()}`.")
    lines.extend(["",
                  "- From the repository root, run:", "",
                  "```bash",
                  f"python3 -m pokerlab.experiment --opponent calling --hands {design['hands_per_seat']} --replicates {design['replicates']} --seed {design['seed']} --output results/calling-pilot.json",
                  f"python3 -m pokerlab.experiment --opponent rank_selective --hands {design['hands_per_seat']} --replicates {design['replicates']} --seed {design['seed']} --output results/rank-selective-pilot.json",
                  f"python3 -m pokerlab.experiment --opponent calling --switch-to aggressive --switch-after 6 --hands {design['hands_per_seat']} --replicates {design['replicates']} --seed {design['seed']} --output results/switch-pilot.json",
                  "python3 -m pokerlab.report --output REPORT.md",
                  "```", "",
                  "The pre-switch hand records exactly match the calling-only "
                  "run for the same seed, seat and hand index. The report "
                  "builder rejects incomplete seat/hand grids, "
                  "repeated seeds, wrong phase labels, mismatched input hashes "
                  "and means or contrasts inconsistent with raw hand rewards. "
                  "Timing-dependent profiler metadata is not used here.", "",
                  "| Condition | Input JSON SHA-256 |", "| --- | --- |"])
    for label, path, digest, _ in records:
        lines.append(f"| {label} (`{path.as_posix()}`) | `{digest}` |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate and render Leduc pilot report")
    parser.add_argument("--calling", type=Path, default=Path("results/calling-pilot.json"))
    parser.add_argument("--held-out", type=Path,
                        default=Path("results/rank-selective-pilot.json"))
    parser.add_argument("--switch", type=Path, default=Path("results/switch-pilot.json"))
    parser.add_argument("--strategy", type=Path, default=Path("strategy.json"))
    parser.add_argument("--rules", type=Path, default=Path("rules.md"))
    parser.add_argument("--output", type=Path, default=Path("REPORT.md"))
    args = parser.parse_args()
    data = render_report((args.calling, args.held_out, args.switch),
                         strategy=args.strategy, rules=args.rules)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(data, encoding="utf-8")
    print(f"saved={args.output}")


if __name__ == "__main__":
    main()
