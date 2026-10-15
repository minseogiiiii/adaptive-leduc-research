"""QT V6 research B: exact one-hand risk/reward frontier in frozen Leduc.

A whole-hand hidden lottery chooses CFR vs a posterior-mixture best response.
The opponent observes actions, not the lottery draw. Exact worst-case payoff
is computed by best responding to the *mixture* (not each branch separately).

This is risk of worst-case EXPECTED chip payoff, not bankroll VaR, CVaR,
drawdown, or real-market alpha. Tightness of bound and opportunity cost are
reported separately for each seat, with no post-hoc choice of risk budgets.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path

from .bayes import example_models
from .cfr import (TabularPolicy,best_response,best_response_mixture,
                  profile_value)
from .experiment import HELD_OUT_POLICIES
from .qt_v4 import safe_weight
from .qt_safe_exact import exact_safe_weight

BUDGETS=(0,0.02,0.05,0.1,0.2,0.4,0.8,2.0)


def _payoff(policy,opponent,seat):
    return (profile_value(policy,opponent) if seat==0
            else -profile_value(opponent,policy))


def _mixture_payoff(base,exploit,opp,seat,w):
    return (1-w)*_payoff(base,opp,seat)+w*_payoff(exploit,opp,seat)


def single_frontier(*,seat,belief_case="uniform",strategy="strategy.json",
                    budgets=BUDGETS,search_steps=10):
    if seat not in (0,1) or search_steps<0:
        raise ValueError("Invalid seat/search steps")
    baseline=TabularPolicy.load(strategy)
    models=example_models(baseline)
    names=tuple(models)
    prior=tuple(
        (0.85 if name=="calling" else 0.15/(len(names)-1))
        if belief_case=="calling_concentrated"
        else (1/len(names) for _ in names))
    if belief_case not in ("uniform","calling_concentrated"):
        raise ValueError("Unknown belief case")
    exploit_ev,exploit=best_response_mixture(tuple(models.values()),prior,seat)
    expected_baseline=math.fsum(
        p*_payoff(baseline,policy,seat)
        for p,policy in zip(prior,models.values()))
    baseline_worst=-best_response(baseline,1-seat)[0]
    opponent_worst_unrestricted=-best_response(exploit,1-seat)[0]
    heldout=HELD_OUT_POLICIES["rank_selective"](baseline)
    baseline_heldout=_payoff(baseline,heldout,seat)
    exploit_heldout=_payoff(exploit,heldout,seat)
    start=time.monotonic()
    rows=[]
    for budget in budgets:
        if not math.isfinite(budget) or budget<0:
            raise ValueError("Invalid budget")
        c=safe_weight(baseline,exploit,seat,budget,
                      baseline_worst=baseline_worst)
        e=exact_safe_weight(baseline,exploit,seat,budget,
                            baseline_worst=baseline_worst,
                            bisection_steps=search_steps)
        exact_worst=-best_response_mixture(
            (baseline,exploit),(1-e.weight,e.weight),1-seat)[0]
        cons_worst=-best_response_mixture(
            (baseline,exploit),(1-c.weight,c.weight),1-seat)[0]
        floor=baseline_worst-budget
        if exact_worst<floor-1e-8 or cons_worst<floor-1e-8:
            raise AssertionError("Unsafe ex-ante mixture")
        if e.weight+1e-10 < c.weight:
            raise AssertionError("Exact optimizer must weakly dominate conservative")
        posterior_ev=(1-e.weight)*expected_baseline+e.weight*exploit_ev
        posterior_ev_conservative=(1-c.weight)*expected_baseline+c.weight*exploit_ev
        heldout_ev=(1-e.weight)*baseline_heldout+e.weight*exploit_heldout
        rows.append({
            "budget":budget,"risk_floor":floor,
            "base_worst":baseline_worst,
            "worst_exact":exact_worst,
            "worst_conservative":cons_worst,
            "realized_worst_case_risk":baseline_worst-exact_worst,
            "weight_exact":e.weight,
            "weight_conservative":c.weight,
            "posterior_ev_exact":posterior_ev,
            "posterior_ev_conservative":posterior_ev_conservative,
            "gain_exact_over_conservative":posterior_ev-posterior_ev_conservative,
            "posterior_ev_gain_over_cfr":posterior_ev-expected_baseline,
            "opportunity_cost_vs_full_BR":exploit_ev-posterior_ev,
            "heldout_rank_ev":heldout_ev,
            "heldout_rank_delta_vs_cfr":heldout_ev-baseline_heldout,
        })
    for a,b in zip(rows,rows[1:]):
        if b["weight_exact"]+1e-10<a["weight_exact"]:
            raise AssertionError("Risk frontier weight not monotonic")
        if b["posterior_ev_exact"]+1e-8<a["posterior_ev_exact"]:
            raise AssertionError("Expected value not monotonic")
        if b["worst_exact"]>a["worst_exact"]+1e-8:
            raise AssertionError("Worst-case payoff must weakly decrease with exploitation")
    return {
        "protocol":{"seat":seat,"belief_case":belief_case,
                    "prior_types":dict(zip(names,prior)),"budgets":list(budgets),
                    "search_steps":search_steps,
                    "risk_type":"one-hand exact worst-case EXPECTED payoff",
                    "algorithm":"hidden whole-hand CFR/BR lottery"},
        "base_ev":expected_baseline,"full_BR_ev":exploit_ev,
        "base_worst":baseline_worst,"full_BR_worst":opponent_worst_unrestricted,
        "base_heldout_ev":baseline_heldout,"full_BR_heldout_ev":exploit_heldout,
        "rows":rows,"runtime_seconds":round(time.monotonic()-start,3)
    }


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--seat",type=int,choices=(0,1),required=True)
    ap.add_argument("--belief-case",
                    choices=("uniform","calling_concentrated"),
                    default="calling_concentrated")
    ap.add_argument("--steps",type=int,default=10)
    ap.add_argument("--output",default=None)
    a=ap.parse_args()
    r=single_frontier(seat=a.seat,belief_case=a.belief_case,
                      search_steps=a.steps)
    if a.output:
        Path(a.output).write_text(json.dumps(r,indent=2)+"\n")
    print(json.dumps(r,indent=2))


if __name__=="__main__":
    main()
