"""B2: richer exact safety frontier using whole-hand lotteries over 3 policies.

For each desired mixture direction, maximize posterior EV on a feasible
1D ray under exact worst-case single-hand EXPECTED payoff constraint.
The search grid over mixture directions is explicitly finite, not a
global solution to the full extensive-form constrained game.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from .bayes import example_models
from .cfr import TabularPolicy,best_response,best_response_mixture
from .experiment import HELD_OUT_POLICIES
from .qt_v6_frontier import _payoff

PHIS=(0,.2,.4,.6,.8,1)
EPS=(.1,.4)


def run(*,seat,steps=10,budgets=EPS,phis=PHIS,
        strategy="strategy.json"):
    if seat not in (0,1) or not isinstance(steps,int) or steps<0:
        raise ValueError("Invalid seat/steps")
    base=TabularPolicy.load(strategy)
    models=example_models(base)
    prior=tuple(.85 if name=="calling" else .05 for name in models)
    target_value,target=best_response_mixture(tuple(models.values()),prior,seat)
    _,aggressive=best_response(models["aggressive"],responder=seat)
    baseline_worst=-best_response(base,1-seat)[0]
    heldout=HELD_OUT_POLICIES["rank_selective"](base)
    policies=(base,target,aggressive)
    posterior_evs=[
        math.fsum(p*_payoff(pol,opp,seat)
                  for p,opp in zip(prior,models.values()))
        for pol in policies]
    heldout_evs=[_payoff(pol,heldout,seat) for pol in policies]

    def worst(w,phi):
        weights=(1-w,w*phi,w*(1-phi))
        return -best_response_mixture(policies,weights,
                                      responder=1-seat)[0]

    output=[]
    for budget in budgets:
        if budget<0 or not math.isfinite(budget):
            raise ValueError("Invalid budget")
        floor=baseline_worst-budget
        candidates=[]
        for phi in phis:
            gain_per_w=(phi*posterior_evs[1]+(1-phi)*posterior_evs[2]
                        -posterior_evs[0])
            if gain_per_w<=0:
                w=0.
            elif worst(1,phi)>=floor-1e-10:
                w=1.
            else:
                lo,hi=0.,1.
                for _ in range(steps):
                    mid=(lo+hi)/2
                    if worst(mid,phi)>=floor-1e-10:
                        lo=mid
                    else:
                        hi=mid
                w=lo
            value=(1-w)*posterior_evs[0]+w*(phi*posterior_evs[1]
                                          +(1-phi)*posterior_evs[2])
            risk=worst(w,phi)
            if risk<floor-1e-8:
                raise AssertionError("Exact adversarial safety floor violated")
            heldout_value=((1-w)*heldout_evs[0]+w*(phi*heldout_evs[1]
                                                +(1-phi)*heldout_evs[2]))
            candidates.append({"phi_target":phi,"weight_non_cfr":w,
                               "posterior_ev":value,"worst_ev":risk,
                               "heldout_ev":heldout_value})
        best=max(candidates,key=lambda x:x["posterior_ev"])
        one_d=next(x for x in candidates if abs(x["phi_target"]-1)<1e-10)
        if best["posterior_ev"]<one_d["posterior_ev"]-1e-9:
            raise AssertionError("2D grid must contain 1D target")
        output.append({
            "budget":budget,"risk_floor":floor,"best":best,
            "one_dimensional":one_d,
            "gain_over_one_dimensional":best["posterior_ev"]-
                                         one_d["posterior_ev"],
            "heldout_delta":best["heldout_ev"]-one_d["heldout_ev"],
            "candidate_rays":candidates
        })
    return {
        "protocol":{"seat":seat,"risk_type":"single-hand worst expected chips",
                    "target_prior":dict(zip(models,prior)),
                    "budget_grid":list(budgets),"phi_grid":list(phis),
                    "steps":steps,
                    "note":"Exact 3-policy lottery grid, not full game LP"},
        "baseline_ev":posterior_evs[0],
        "target_BR_ev":target_value,
        "aggressive_BR_ev_under_target_prior":posterior_evs[2],
        "baseline_worst":baseline_worst,
        "rows":output
    }


def main():
    a=argparse.ArgumentParser()
    a.add_argument("--seat",type=int,choices=(0,1),required=True)
    a.add_argument("--output",default=None)
    opt=a.parse_args()
    obj=run(seat=opt.seat)
    if opt.output:
        Path(opt.output).write_text(json.dumps(obj,indent=2)+"\n")
    print(json.dumps(obj,indent=2))


if __name__=="__main__":
    main()
