"""Expected Value of Sample Information (EVSI) in the actual Leduc game.

A signal is observed BEFORE the hand, predicts opponent policy type, and has
a declared likelihood P(signal | opponent_type). It never reveals private cards.
Values use the exact information-set best response and are not market returns.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass

from .cfr import TabularPolicy, best_response, best_response_mixture
from .bayes import example_models


@dataclass(frozen=True)
class SignalValue:
    uninformed_ev: float
    informed_ev: float
    perfect_info_ev: float
    evsi: float
    evpi: float
    posterior_by_signal: dict


def value_of_signal(*, opponents, prior, signal_likelihoods, seat: int):
    """Return ex-ante EVSI and EVPI in chips/hand.

    signal_likelihoods maps signal name to P(signal|type_j) in type order.
    Each type's probabilities must sum to one across signals.
    """
    opponents = tuple(opponents)
    prior = tuple(float(x) for x in prior)
    if (seat not in (0, 1) or not opponents or len(prior) != len(opponents)
            or not signal_likelihoods):
        raise ValueError("Invalid opponents, prior, seat, or signals")
    if any(not math.isfinite(x) or x < 0 for x in prior) or abs(math.fsum(prior)-1) > 1e-10:
        raise ValueError("Prior must be a probability distribution")
    signals = {str(name): tuple(float(x) for x in row)
               for name, row in signal_likelihoods.items()}
    if any(len(row) != len(prior) or any(not math.isfinite(x) or x < 0 for x in row)
           for row in signals.values()):
        raise ValueError("Invalid signal likelihoods")
    for j in range(len(prior)):
        if abs(math.fsum(row[j] for row in signals.values()) - 1) > 1e-10:
            raise ValueError("Each type's signal likelihoods must sum to one")

    uninformed, _ = best_response_mixture(opponents, prior, seat)
    informed = 0.0
    posteriors = {}
    for name, likelihoods in signals.items():
        p_signal = math.fsum(p * l for p, l in zip(prior, likelihoods))
        if p_signal == 0:
            continue
        posterior = tuple(p * l / p_signal for p, l in zip(prior, likelihoods))
        value, _ = best_response_mixture(opponents, posterior, seat)
        informed += p_signal * value
        posteriors[name] = {"probability": p_signal,
                            "posterior": list(posterior),
                            "optimal_value": value}
    perfect = math.fsum(p * best_response(opp, seat)[0]
                        for p, opp in zip(prior, opponents) if p > 0)
    evsi = informed - uninformed
    evpi = perfect - uninformed
    if evsi < -1e-8 or evpi < evsi - 1e-8:
        raise AssertionError("Information-value monotonicity violated")
    return SignalValue(uninformed, informed, perfect, max(0.0, evsi),
                       max(0.0, evpi), posteriors)


def main():
    parser = argparse.ArgumentParser(description="Exact Leduc information value")
    parser.add_argument("--strategy", default="strategy.json")
    parser.add_argument("--accuracy", type=float, default=0.8)
    args = parser.parse_args()
    if not 0.5 <= args.accuracy <= 1:
        parser.error("Accuracy must be in [0.5,1]")
    models = example_models(TabularPolicy.load(args.strategy))
    opponents = (models["calling"], models["aggressive"])
    p = args.accuracy
    signals = {"signal_calling": (p, 1-p),
               "signal_aggressive": (1-p, p)}
    result = {f"seat_{seat}": asdict(value_of_signal(
        opponents=opponents, prior=(0.5, 0.5),
        signal_likelihoods=signals, seat=seat)) for seat in (0, 1)}
    print(json.dumps({"accuracy": p, "opponents": ["calling", "aggressive"],
                      "results": result}, indent=2))


if __name__ == "__main__":
    main()
