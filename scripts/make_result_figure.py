#!/usr/bin/env python3
"""Generate the recruiter-facing payoff-contrast figure from validated records."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from xml.sax.saxutils import escape

from pokerlab.followup import render_followup

SCENARIOS = (
    ("Rank selective", Path("results/followup-rank_selective.json"), "stationary"),
    ("Round polarized", Path("results/followup-round_polarized.json"), "stationary"),
    ("Pressure reactive", Path("results/followup-pressure_reactive.json"), "stationary"),
    ("Calling → aggressive", Path("results/followup-switch.json"), "switch"),
)


def _load_rows() -> list[dict[str, object]]:
    paths = tuple(path for _, path, _ in SCENARIOS)
    # Reuse the research report validator before plotting any saved number.
    render_followup(
        paths,
        strategy=Path("strategy.json"),
        rules=Path("rules.md"),
        protocol=Path("FOLLOWUP_PROTOCOL.md"),
    )
    rows: list[dict[str, object]] = []
    for label, path, kind in SCENARIOS:
        result = json.loads(path.read_text(encoding="utf-8"))
        contrast = result["paired_contrasts"]["learning_minus_frozen_prior"]
        low, high = contrast["bootstrap_95_percent"]
        rows.append({
            "label": label,
            "kind": kind,
            "mean": float(contrast["mean"]),
            "low": float(low),
            "high": float(high),
        })
    return rows


def _fmt(value: float) -> str:
    return f"{value:+.3f}"


def build_svg(rows: list[dict[str, object]]) -> str:
    width, height = 1040, 450
    left, right, top, bottom = 250, 245, 92, 78
    plot_width = width - left - right
    row_gap = 58

    lows = [float(row["low"]) for row in rows]
    highs = [float(row["high"]) for row in rows]
    raw_low = min(0.0, *lows)
    raw_high = max(0.0, *highs)
    x_min = math.floor((raw_low - 0.02) * 10) / 10
    x_max = math.ceil((raw_high + 0.02) * 10) / 10
    if x_max - x_min < 0.4:
        midpoint = (x_min + x_max) / 2
        x_min = math.floor((midpoint - 0.2) * 10) / 10
        x_max = math.ceil((midpoint + 0.2) * 10) / 10

    def x(value: float) -> float:
        return left + (value - x_min) / (x_max - x_min) * plot_width

    ticks: list[float] = []
    tick = math.ceil(x_min * 10 - 1e-9) / 10
    while tick <= x_max + 1e-9:
        ticks.append(round(tick, 10))
        tick += 0.1

    out = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        '<title id="title">Adaptive policy versus frozen-prior payoff difference</title>',
        '<desc id="desc">Learning minus frozen-prior net chips per hand with descriptive 95 percent bootstrap intervals for three held-out stationary opponents and one switching-opponent stress condition.</desc>',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Arial,sans-serif;fill:#17202a}.title{font-size:22px;font-weight:700}.subtitle{font-size:14px;fill:#4b5563}.label{font-size:14px;font-weight:600}.tick{font-size:12px;fill:#6b7280}.value{font-size:12px;fill:#374151}.foot{font-size:12px;fill:#6b7280}</style>',
        '<text id="chart-title" class="title" x="32" y="36">Adaptive learning vs frozen-prior baseline</text>',
        '<text class="subtitle" x="32" y="60">Learning − frozen prior, net chips per hand; bars show descriptive 95% bootstrap intervals</text>',
    ]

    axis_y = top + row_gap * len(rows) + 12
    for tick in ticks:
        tx = x(tick)
        stroke = "#9ca3af" if abs(tick) < 1e-12 else "#e5e7eb"
        sw = "2" if abs(tick) < 1e-12 else "1"
        out.append(f'<line x1="{tx:.2f}" y1="{top-18}" x2="{tx:.2f}" y2="{axis_y}" stroke="{stroke}" stroke-width="{sw}"/>')
        out.append(f'<text class="tick" x="{tx:.2f}" y="{axis_y+22}" text-anchor="middle">{tick:+.1f}</text>')

    for index, row in enumerate(rows):
        y = top + index * row_gap
        label = escape(str(row["label"]))
        mean, low, high = float(row["mean"]), float(row["low"]), float(row["high"])
        color = "#0f766e" if row["kind"] == "stationary" else "#b45309"
        out.append(f'<text class="label" x="32" y="{y+5}">{label}</text>')
        if row["kind"] == "switch":
            out.append(f'<text class="tick" x="32" y="{y+23}">switch stress</text>')
        out.append(f'<line x1="{x(low):.2f}" y1="{y:.2f}" x2="{x(high):.2f}" y2="{y:.2f}" stroke="{color}" stroke-width="4" stroke-linecap="round"/>')
        out.append(f'<line x1="{x(low):.2f}" y1="{y-7:.2f}" x2="{x(low):.2f}" y2="{y+7:.2f}" stroke="{color}" stroke-width="2"/>')
        out.append(f'<line x1="{x(high):.2f}" y1="{y-7:.2f}" x2="{x(high):.2f}" y2="{y+7:.2f}" stroke="{color}" stroke-width="2"/>')
        out.append(f'<circle cx="{x(mean):.2f}" cy="{y:.2f}" r="6" fill="{color}" stroke="#ffffff" stroke-width="2"/>')
        out.append(f'<text class="value" x="{width-right+28}" y="{y+4}">{_fmt(mean)} [{_fmt(low)}, {_fmt(high)}]</text>')

    out.extend([
        f'<text class="tick" x="{left + plot_width/2:.2f}" y="{height-43}" text-anchor="middle">Adaptive payoff advantage →</text>',
        f'<text class="foot" x="32" y="{height-16}">First three rows are prespecified held-out stationary policies. The switch row is secondary and excluded from the family-level primary contrast.</text>',
        '</svg>',
        '',
    ])
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the validated follow-up contrast figure")
    parser.add_argument("--output", type=Path, default=Path("results/adaptive-vs-frozen.svg"))
    args = parser.parse_args()
    rows = _load_rows()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(build_svg(rows), encoding="utf-8")
    print(f"saved={args.output}")


if __name__ == "__main__":
    main()
