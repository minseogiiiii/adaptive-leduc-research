"""Reconstruct the prespecified held-out family analysis from saved hand rows."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path

from .report import _interval, _metric, validate_result

OPPONENTS = ("rank_selective", "round_polarized", "pressure_reactive")
SEED = 20261001
REPLICATES = 24
HANDS = 12
SWITCH_AFTER = 6


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def render_followup(paths: tuple[Path, Path, Path, Path], *, strategy: Path,
                    rules: Path, protocol: Path) -> str:
    """Validate all four prespecified records before calculating the family mean."""
    if len(paths) != 4 or len(set(paths)) != 4:
        raise ValueError("Exactly four distinct scenario paths are required")
    records = []
    for path, opponent, switch_to, switch_after in zip(
            paths, (*OPPONENTS, "calling"),
            (None, None, None, "aggressive"),
            (None, None, None, SWITCH_AFTER)):
        result = json.loads(path.read_text(encoding="utf-8"))
        validate_result(result, strategy_sha256=_sha(strategy),
                        rules_sha256=_sha(rules), opponent=opponent,
                        switch_to=switch_to, switch_after=switch_after)
        design = result["design"]
        if any(design[key] != value for key, value in (
                ("seed", SEED), ("replicates", REPLICATES),
                ("hands_per_seat", HANDS), ("bootstrap_resamples", 2000),
                ("candidate_types", ["baseline", "folding", "calling", "aggressive"]),
                ("initial_prior", [.25] * 4))):
            raise ValueError(f"Unexpected follow-up workload in {path}")
        records.append((path, result))

    # A joint bootstrap over replicate indices requires the same exogenous
    # hand draws in every scenario. Opponent actions can still differ.
    schedules = [{(row["replicate"], row["seat"], row["hand"]): row["seed"]
                  for row in result["hands"]} for _, result in records]
    if any(schedule != schedules[0] for schedule in schedules[1:]):
        raise ValueError("Conditions have different hand seed schedules")

    per_replicate = [statistics.fmean(
        result["replicate_means"][rep]["learning"]
        - result["replicate_means"][rep]["frozen_prior"]
        for _, result in records[:3]) for rep in range(REPLICATES)]
    family_mean = statistics.fmean(per_replicate)
    family_low, family_high = _interval(per_replicate, SEED + 1)
    lines = ["# Leduc held-out opponent follow-up", "",
             "## Fixed design and primary result", "",
             "The conditions, primary contrast, replicate unit, seed and stopping "
             "rule were fixed in `FOLLOWUP_PROTOCOL.md` before these outcomes. "
             "Each condition uses 24 independent two-seat replicates with "
             "12 hands per seat. All share the same hand seed schedule; each "
             "match starts with a fresh prior. The three held-out policies "
             "are absent from the four candidate types.", "",
             "Primary equally weighted learning minus frozen-prior contrast "
             "over the three held-out opponents: "
             f"**{family_mean:+.4f} [{family_low:+.4f}, {family_high:+.4f}] "
             "net chips/hand** (95% descriptive joint-replicate bootstrap, "
             "2,000 resamples; seed 20261002).", "",
             "## Condition results", "",
             "All rewards are chips per hand averaged over seats. Brackets "
             "are descriptive 95% replicate bootstrap intervals.", "",
             "| Condition | Learning | Frozen prior | CFR | Learning − frozen [interval] | Learning − CFR [interval] |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for label, (_, result) in zip((*OPPONENTS, "calling to aggressive"), records):
        rewards = result["mean_reward"]
        lines.append(f"| {label} | {rewards['learning']:+.4f} | "
                     f"{rewards['frozen_prior']:+.4f} | {rewards['cfr']:+.4f} | "
                     f"{_metric(result, 'frozen_prior')} | {_metric(result, 'cfr')} |")
    switch = records[3][1]
    lines.extend(["", "## Switch stress condition", "",
                  "The opponent changes before zero-based hand 6; the learner's "
                  "beliefs persist across the switch. This condition is "
                  "secondary and excluded from the family contrast.", "",
                  "| Phase | Learning | Frozen prior | CFR | Learning − frozen [interval] | Learning − CFR [interval] |",
                  "| --- | ---: | ---: | ---: | ---: | ---: |"])
    for phase in ("pre_switch", "post_switch"):
        result = switch["phases"][phase]
        rewards = result["mean_reward"]
        lines.append(f"| {phase} | {rewards['learning']:+.4f} | "
                     f"{rewards['frozen_prior']:+.4f} | {rewards['cfr']:+.4f} | "
                     f"{_metric(result, 'frozen_prior')} | {_metric(result, 'cfr')} |")
    for control in ("frozen_prior", "cfr"):
        change = switch["change_in_paired_contrasts"][f"learning_minus_{control}"]
        low, high = change["bootstrap_95_percent"]
        lines.append(f"- Post minus pre learning − {control}: "
                     f"{change['mean']:+.4f} [{low:+.4f}, {high:+.4f}].")
    lines.extend(["", "## Interpretation and limits", "",
                  "The frozen-prior response starts from the same policy as "
                  "the learner; their between-hand posterior updates are the "
                  "intended difference. The three held-out policies are "
                  "constructed rules rather than a representative sample of "
                  "people. Common random numbers reduce some comparison "
                  "noise, but the trajectories can diverge. The experiment "
                  "builds on a small earlier pilot, and 24 replicate units "
                  "leave substantial uncertainty. The bootstrap interval "
                  "is descriptive; it is not a claim of general performance "
                  "or a multiplicity-adjusted significance test. The switch "
                  "violates the learner's stationary-type assumption. "
                  "The simulator and evaluator still share implementation; "
                  "external rule-matched validation remains outstanding.", "",
                  "## Reproduction and provenance", "",
                  "Run from the repository root, then rerun "
                  "`python3 -m unittest discover -s tests -v`.", "",
                  "```bash"])
    for opponent, (path, _) in zip(OPPONENTS, records):
        lines.append("python3 -m pokerlab.experiment "
                     f"--opponent {opponent} --hands 12 --replicates 24 "
                     f"--seed 20261001 --output {path.as_posix()}")
    lines.extend(["python3 -m pokerlab.experiment --opponent calling "
                  "--switch-to aggressive --switch-after 6 --hands 12 "
                  "--replicates 24 --seed 20261001 "
                  f"--output {records[3][0].as_posix()}",
                  "python3 -m pokerlab.followup --output FOLLOWUP.md", "```", "",
                  "The generator checks design hashes, complete hand grids, "
                  "posterior continuity, rewards, bootstrap intervals, and "
                  "the identical seed schedule. It recomputes the primary "
                  "contrast from replicate means.", "",
                  "| Input | SHA-256 |", "| --- | --- |"])
    for path in (strategy, rules, protocol,
                 *(Path(__file__).resolve().parent / name for name in
                   ("game.py", "cfr.py", "bayes.py", "experiment.py",
                    "report.py", "followup.py")),
                 *(path for path, _ in records)):
        lines.append(f"| `{path.as_posix()}` | `{_sha(path)}` |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate and report held-out follow-up")
    for opponent in OPPONENTS:
        parser.add_argument(f"--{opponent.replace('_', '-')}", type=Path,
                            default=Path(f"results/followup-{opponent}.json"))
    parser.add_argument("--switch", type=Path,
                        default=Path("results/followup-switch.json"))
    parser.add_argument("--strategy", type=Path, default=Path("strategy.json"))
    parser.add_argument("--rules", type=Path, default=Path("rules.md"))
    parser.add_argument("--protocol", type=Path, default=Path("FOLLOWUP_PROTOCOL.md"))
    parser.add_argument("--output", type=Path, default=Path("FOLLOWUP.md"))
    args = parser.parse_args()
    paths = tuple(getattr(args, name) for name in OPPONENTS) + (args.switch,)
    data = render_followup(paths, strategy=args.strategy, rules=args.rules,
                           protocol=args.protocol)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(data, encoding="utf-8")
    print(f"saved={args.output}")


if __name__ == "__main__":
    main()
