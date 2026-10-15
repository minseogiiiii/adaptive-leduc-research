"""Exact worst-case optimized safe whole-hand mixing for Leduc QT research.

The opponent does NOT observe the policy coin flip. Therefore exact
best_response_mixture((baseline, exploit), (1-w, w), responder=other)
evaluates the correct hidden ex-ante mixture, unlike a per-infoset mix.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .cfr import best_response_mixture
from .qt_v4 import SafetyCertificate, safe_weight


def exact_safe_weight(baseline, exploit, seat: int, budget: float,
                      baseline_worst: float | None = None,
                      bisection_steps: int = 4) -> SafetyCertificate:
    """Optimize posterior EV over w, subject to EXACT game worst-case floor.

    Since exploit is a posterior best response, expected reward increases
    linearly in w. Worst-case value min_tau E[V(pi_w,tau)] is concave in w,
    hence the feasible set containing w=0 is an interval. Bisection is safe.
    A conservative certified feasible w from safe_weight initializes search.
    """
    if not isinstance(bisection_steps, int) or bisection_steps < 0:
        raise ValueError("bisection_steps must be a nonnegative integer")
    conservative = safe_weight(
        baseline, exploit, seat, budget, baseline_worst=baseline_worst)
    floor = conservative.allowed_floor

    def worst(w):
        value, _ = best_response_mixture(
            (baseline, exploit), (1.0 - w, w), responder=1 - seat)
        if not math.isfinite(value):
            raise ValueError("Nonfinite exact best-response value")
        return -value

    if conservative.weight >= 1.0 - 1e-12:
        return conservative

    if worst(1.0) >= floor - 1e-10:
        return SafetyCertificate(1.0, conservative.baseline_worst,
                                 conservative.exploit_worst, worst(1.0), floor)

    low, high = conservative.weight, 1.0
    for _ in range(bisection_steps):
        mid = (low + high) / 2
        if worst(mid) >= floor - 1e-10:
            low = mid
        else:
            high = mid
    exact_lower_bound = worst(low)
    if exact_lower_bound < floor - 1e-9:
        raise AssertionError("Exact safe optimizer returned unsafe policy")
    return SafetyCertificate(low, conservative.baseline_worst,
                             conservative.exploit_worst, exact_lower_bound, floor)
