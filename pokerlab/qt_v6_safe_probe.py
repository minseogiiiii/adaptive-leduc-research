"""Exact single-hand safe-exploration policy and fresh-seed information audit.

A fixed hidden whole-hand lottery between CFR and a probing policy.
Within the budget the largest feasible probe weight maximizes a linear
increase in expected classification accuracy (when positive).
No live trading or multi-hand safety is implied.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from .bayes import ActionBiasPolicy
from .cfr import TabularPolicy,best_response_mixture,profile_value
from .game import Action
from .qt_safe_exact import exact_safe_weight
from .qt_v6_observability import run as observe
from .qt_v6_changepoint_eval import percentile_bootstrap


def run(*,seed=20261024,replicates=24,hands_per_type=48,
        max_worst_loss=.05,strategy="strategy.json"):
    base=TabularPolicy.load(strategy)
    probe=ActionBiasPolicy(base,{
        Action.CHECK:.7,Action.CALL:.7,
        Action.BET:2.5,Action.RAISE:2.5})
    cert={str(seat):exact_safe_weight(
        base,probe,seat,max_worst_loss,bisection_steps=10)
        for seat in (0,1)}
    w=min(x.weight for x in cert.values())
    actual_worst={}
    for seat in (0,1):
        worst=-best_response_mixture(
            (base,probe),(1-w,w),responder=1-seat)[0]
        if worst<cert[str(seat)].allowed_floor-1e-8:
            raise AssertionError("Unsafe information lottery")
        actual_worst[str(seat)]=worst
    fresh=observe(seed=seed,replicates=replicates,
                  hands_per_type=hands_per_type,strategy=strategy)
    fixed=fresh["probe_metrics"]["cfr"]
    active=fresh["probe_metrics"]["aggressive_probe"]
    delta_rep=[w*(a-b) for a,b in zip(
        fresh["replicate_metrics"]["aggressive_probe"]["accuracy"],
        fresh["replicate_metrics"]["cfr"]["accuracy"])]
    delta=statistics.fmean(delta_rep)
    means={
        metric:(1-w)*fixed[metric]+w*active[metric]
        for metric in ("accuracy","sensitivity","false_positive","jdiv")}
    exp_ev={}
    for seat in ("0","1"):
        e=fresh["exact_expected_chips_per_hand_by_seat"][seat]
        exp_ev[seat]=(1-w)*e["cfr"]+w*e["aggressive_probe"]
    return {
        "protocol":{"seed":seed,"replicates":replicates,
                    "hands_per_type_per_seat_per_replicate":hands_per_type,
                    "worst_case_chip_budget":max_worst_loss,
                    "policy_lottery":"one CFR or aggressive probe for entire hand",
                    "classification_lottery_observed":"yes"},
        "safe_probability_of_probe":w,
        "seatwise_unconstrained_safe_weights":{
            seat:certificate.weight for seat,certificate in cert.items()},
        "worst_expected_payoff_by_seat":actual_worst,
        "required_floor_by_seat":{
            seat:certificate.allowed_floor for seat,certificate in cert.items()},
        "mixed_probe_information":means,
        "base_information":fixed,
        "active_information":active,
        "accuracy_gain":{"mean":delta,
                         "ci95":percentile_bootstrap(delta_rep,seed+17)},
        "exact_expected_chips_by_seat":exp_ev,
        "base_chips_by_seat":{
            seat:values["cfr"]
            for seat,values in
            fresh["exact_expected_chips_per_hand_by_seat"].items()}
    }


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--seed",type=int,default=20261024)
    ap.add_argument("--replicates",type=int,default=24)
    ap.add_argument("--hands",type=int,default=48)
    ap.add_argument("--output",default=None)
    a=ap.parse_args()
    obj=run(seed=a.seed,replicates=a.replicates,
            hands_per_type=a.hands)
    if a.output:
        Path(a.output).write_text(json.dumps(obj,indent=2)+"\n")
    print(json.dumps(obj,indent=2))


if __name__=="__main__":
    main()
