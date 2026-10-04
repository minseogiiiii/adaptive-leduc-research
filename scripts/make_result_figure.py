#!/usr/bin/env python3
"""Generate the recruiter-facing payoff-contrast figure from validated records."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from xml.sax.saxutils import escape

from pokerlab.followup import OPPONENTS, REPLICATES, SEED, render_followup
from pokerlab.report import _interval

STATIONARY = (
    ("Rank selective", "rank_selective", Path("results/followup-rank_selective.json")),
    ("Round polarized", "round_polarized", Path("results/followup-round_polarized.json")),
    ("Pressure reactive", "pressure_reactive", Path("results/followup-pressure_reactive.json")),
)
SWITCH_PATH = Path("results/followup-switch.json")


def _rounded_contrast(summary: dict) -> dict[str, float]:
    contrast = summary["paired_contrasts"]["learning_minus_frozen_prior"]
    low, high = contrast["bootstrap_95_percent"]
    return {
        "mean": round(float(contrast["mean"]), 4),
        "low": round(float(low), 4),
        "high": round(float(high), 4),
    }


def _load_rows() -> tuple[list[dict[str, object]], dict[str, float]]:
    if tuple(name for _, name, _ in STATIONARY) != OPPONENTS:
        raise AssertionError("Figure scenarios must follow the prespecified opponent order")
    paths = tuple(path for _, _, path in STATIONARY) + (SWITCH_PATH,)
    # Do not plot a saved number until the full follow-up validator accepts it.
    render_followup(
        paths,
        strategy=Path("strategy.json"),
        rules=Path("rules.md"),
        protocol=Path("FOLLOWUP_PROTOCOL.md"),
    )
    stationary = [
        json.loads(path.read_text(encoding="utf-8"))
        for _, _, path in STATIONARY
    ]
    switch = json.loads(SWITCH_PATH.read_text(encoding="utf-8"))

    per_replicate = [
        statistics.fmean(
            result["replicate_means"][rep]["learning"]
            - result["replicate_means"][rep]["frozen_prior"]
            for result in stationary
        )
        for rep in range(REPLICATES)
    ]
    family_low, family_high = _interval(per_replicate, SEED + 1)
    rows: list[dict[str, object]] = [{
        "label": "Held-out family (primary)",
        "kind": "primary",
        "mean": round(statistics.fmean(per_replicate), 4),
        "low": round(family_low, 4),
        "high": round(family_high, 4),
    }]
    for (label, _, _), result in zip(STATIONARY, stationary):
        rows.append({"label": label, "kind": "stationary",
                     **_rounded_contrast(result)})
    for label, phase in (("Switch: pre", "pre_switch"),
                         ("Switch: post", "post_switch")):
        rows.append({"label": label, "kind": "switch",
                     **_rounded_contrast(switch["phases"][phase])})

    raw_change = switch["change_in_paired_contrasts"]["learning_minus_frozen_prior"]
    change_low, change_high = raw_change["bootstrap_95_percent"]
    change = {
        "mean": round(float(raw_change["mean"]), 4),
        "low": round(float(change_low), 4),
        "high": round(float(change_high), 4),
    }
    return rows, change


def _fmt(value: float) -> str:
    return f"{value:+.3f}"


def build_svg(rows: list[dict[str, object]], change: dict[str, float]) -> str:
    width, height = 1120, 570
    left, right, top = 265, 285, 105
    plot_width = width - left - right
    row_gap = 58

    lows = [float(row["low"]) for row in rows]
    highs = [float(row["high"]) for row in rows]
    raw_low = min(0.0, *lows)
    raw_high = max(0.0, *highs)
    x_min = math.floor((raw_low - 0.02) * 10) / 10
    x_max = math.ceil((raw_high + 0.02) * 10) / 10

    def x(value: float) -> float:
        return left + (value - x_min) / (x_max - x_min) * plot_width

    tick_step = 0.2 if x_max - x_min >= 1.0 else 0.1
    ticks: list[float] = []
    tick = math.ceil((x_min - 1e-12) / tick_step) * tick_step
    while tick <= x_max + 1e-9:
        ticks.append(round(tick, 10))
        tick += tick_step

    axis_y = top + row_gap * len(rows) + 12
    out = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        '<title id="title">Adaptive learning versus frozen-prior payoff</title>',
        '<desc id="desc">Learning minus frozen-prior net chips per hand. The primary held-out family result is near zero, while the switching-opponent stress test moves from positive before the switch to negative after the switch.</desc>',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Arial,sans-serif;fill:#17202a}.title{font-size:22px;font-weight:700}.subtitle{font-size:14px;fill:#4b5563}.label{font-size:14px;font-weight:600}.tick{font-size:12px;fill:#6b7280}.value{font-size:12px;fill:#374151}.foot{font-size:12px;fill:#6b7280}</style>',
        '<text class="title" x="32" y="36">Adaptive learning vs frozen-prior baseline</text>',
        '<text class="subtitle" x="32" y="60">Point = mean payoff difference; line = descriptive 95% bootstrap interval</text>',
        '<text class="subtitle" x="32" y="80">Positive values favor updating from observations; negative values favor the frozen-prior control.</text>',
    ]
    for tick in ticks:
        tx = x(tick)
        stroke = "#9ca3af" if abs(tick) < 1e-12 else "#e5e7eb"
        sw = "2" if abs(tick) < 1e-12 else "1"
        out.append(
            f'<line x1="{tx:.2f}" y1="{top-18}" x2="{tx:.2f}" y2="{axis_y}" '
            f'stroke="{stroke}" stroke-width="{sw}"/>'
        )
        tick_label = "0.0" if abs(tick) < 1e-12 else f"{tick:+.1f}"
        out.append(
            f'<text class="tick" x="{tx:.2f}" y="{axis_y+22}" '
            f'text-anchor="middle">{tick_label}</text>'
        )

    for index, row in enumerate(rows):
        y = top + index * row_gap
        if index == 4:
            separator_y = y - row_gap / 2
            out.append(
                f'<line x1="32" y1="{separator_y:.2f}" x2="{width-32}" '
                f'y2="{separator_y:.2f}" stroke="#d1d5db" stroke-width="1"/>'
            )
        label = escape(str(row["label"]))
        mean, low, high = (float(row["mean"]), float(row["low"]),
                           float(row["high"]))
        color = {
            "primary": "#111827",
            "stationary": "#0f766e",
            "switch": "#b45309",
        }[str(row["kind"])]
        out.append(f'<text class="label" x="32" y="{y+5}">{label}</text>')
        out.append(
            f'<line x1="{x(low):.2f}" y1="{y:.2f}" x2="{x(high):.2f}" '
            f'y2="{y:.2f}" stroke="{color}" stroke-width="4" stroke-linecap="round"/>'
        )
        out.append(
            f'<line x1="{x(low):.2f}" y1="{y-7:.2f}" x2="{x(low):.2f}" '
            f'y2="{y+7:.2f}" stroke="{color}" stroke-width="2"/>'
        )
        out.append(
            f'<line x1="{x(high):.2f}" y1="{y-7:.2f}" x2="{x(high):.2f}" '
            f'y2="{y+7:.2f}" stroke="{color}" stroke-width="2"/>'
        )
        radius = 7 if row["kind"] == "primary" else 6
        out.append(
            f'<circle cx="{x(mean):.2f}" cy="{y:.2f}" r="{radius}" '
            f'fill="{color}" stroke="#ffffff" stroke-width="2"/>'
        )
        out.append(
            f'<text class="value" x="{width-right+25}" y="{y+4}">'
            f'{_fmt(mean)} [{_fmt(low)}, {_fmt(high)}]</text>'
        )

    out.extend([
        f'<text class="tick" x="{left+plot_width/2:.2f}" y="{axis_y+52}" '
        'text-anchor="middle">← frozen-prior higher payoff     learning higher payoff →</text>',
        f'<text class="foot" x="32" y="{height-18}">Primary = equally weighted mean across 3 held-out stationary policies. Switch post − pre change: {_fmt(change["mean"])} [{_fmt(change["low"])}, {_fmt(change["high"])}] chips/hand.</text>',
        '</svg>',
        '',
    ])
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate the validated follow-up contrast figure")
    parser.add_argument("--output", type=Path,
                        default=Path("results/adaptive-vs-frozen.svg"))
    args = parser.parse_args()
    rows, change = _load_rows()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(build_svg(rows, change), encoding="utf-8")
    print(f"saved={args.output}")


if __name__ == "__main__":
    main()
