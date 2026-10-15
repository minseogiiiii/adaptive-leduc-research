"""Common-path prequential validation of Leduc online change point models.

Same legally-observable hands (CFR probe policy) for all filters.
Independent replicate = paired sequence of both seats, never individual hand.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from pathlib import Path

from .bayes import BayesianOpponentModel, _hand_likelihood, example_models
from .cfr import PolicyAgent, TabularPolicy
from .experiment import HELD_OUT_POLICIES
from .qt_v4 import SwitchingBeliefs
from .qt_v6_changepoint import RunLengthBeliefs
from .simulate import run_hand

SCENARIOS=("stationary_calling","stationary_aggressive",
           "switch_calling_aggressive","switch_aggressive_calling",
           "switch_early","heldout_rank_selective","heldout_pressure_reactive")


def scenario_policy(scenario,hand,hands,types,base):
    if scenario=="stationary_calling":
        return types["calling"]
    if scenario=="stationary_aggressive":
        return types["aggressive"]
    if scenario=="switch_calling_aggressive":
        return types["calling"] if hand < hands//2 else types["aggressive"]
    if scenario=="switch_aggressive_calling":
        return types["aggressive"] if hand < hands//2 else types["calling"]
    if scenario=="switch_early":
        return types["calling"] if hand < hands//4 else types["aggressive"]
    if scenario=="heldout_rank_selective":
        return HELD_OUT_POLICIES["rank_selective"](base)
    if scenario=="heldout_pressure_reactive":
        return HELD_OUT_POLICIES["pressure_reactive"](base)
    raise ValueError(scenario)


def percentile_bootstrap(values,seed,coverage=.95,draws=2000):
    if len(values)<2:
        raise ValueError("Need >=2 independent replicate blocks")
    r=random.Random(seed)
    sims=sorted(statistics.fmean(r.choices(values,k=len(values)))
                for _ in range(draws))
    tail=(1-coverage)/2
    return [sims[min(draws-1,int(tail*draws))],
            sims[min(draws-1,int((1-tail)*draws))]]


def run(*,scenario,hands=24,replicates=12,seed=20261018,
        strategy="strategy.json"):
    if scenario not in SCENARIOS or hands<8 or replicates<2:
        raise ValueError("Invalid experiment size/scenario")
    base=TabularPolicy.load(strategy)
    types=example_models(base)
    names=("stationary","switching_v4","bocpd_constant","bocpd_duration")
    scores={name:[] for name in names}
    after_scores={name:[] for name in names}
    detected=[];false_alarm=[]
    rng=random.Random(seed)
    switch_at=(hands//4 if scenario=="switch_early" else hands//2
               if scenario.startswith("switch_") else None)
    by_hand_delta=[[] for _ in range(hands)]
    for rep in range(replicates):
        block={name:[] for name in names}
        post={name:[] for name in names}
        hit=[];alarms=[]
        for seat in (0,1):
            fs={
                "stationary":BayesianOpponentModel(types),
                "switching_v4":SwitchingBeliefs(types,hazard=.1),
                "bocpd_constant":RunLengthBeliefs(types,hazard=.1),
                "bocpd_duration":RunLengthBeliefs(types,hazard_mode="duration",
                                                  short_hazard=.03,long_hazard=.2,
                                                  duration_knee=6)}
            for hand in range(hands):
                opp=scenario_policy(scenario,hand,hands,types,base)
                handseed=rng.randrange(1<<63)
                players=((PolicyAgent(base),PolicyAgent(opp)) if seat==0
                         else (PolicyAgent(opp),PolicyAgent(base)))
                terminal=run_hand(handseed,players)
                obs=terminal.observe(seat)
                likelihoods=[_hand_likelihood(obs,p) for p in fs["switching_v4"].policies]
                row={}
                for name,f in fs.items():
                    predictive=math.fsum(w*l for w,l in zip(f.posterior,likelihoods))
                    if predictive<=0:
                        raise ValueError("Zero likelihood")
                    row[name]=math.log(predictive)
                    block[name].append(row[name])
                    if hand>=hands//2:
                        post[name].append(row[name])
                by_hand_delta[hand].append(row["bocpd_duration"]-row["switching_v4"])
                for f in fs.values():
                    f.update(obs)
                cp=fs["bocpd_duration"].last_change_posterior
                if hand==switch_at:
                    hit.append(cp)
                elif hand>0:
                    alarms.append(cp)
        for name in names:
            scores[name].append(statistics.fmean(block[name]))
            after_scores[name].append(statistics.fmean(post[name]))
        if hit:
            detected.append(statistics.fmean(hit))
        false_alarm.append(statistics.fmean(alarms))
    delta=[a-b for a,b in zip(scores["bocpd_duration"],scores["switching_v4"])]
    constant_gap=max(abs(a-b) for a,b in zip(
        scores["bocpd_constant"],scores["switching_v4"]))
    if constant_gap>1e-10:
        raise AssertionError("Constant hazard BOCPD must equal V4")
    return {
        "protocol":{"scenario":scenario,"hands_per_seat":hands,
                    "replicates":replicates,"seed":seed,
                    "unit":"replicate mean of both seat trajectories",
                    "hazard_v4":.1,"hazard_duration_short":.03,
                    "hazard_duration_long":.2,"duration_knee":6,
                    "selection_policy":"fixed CFR probe; no policy leakage",
                    "switch_after":switch_at},
        "means":{name:statistics.fmean(v) for name,v in scores.items()},
        "second_half_means":{name:statistics.fmean(v) for name,v in after_scores.items()},
        "primary_delta_duration_minus_v4":{
            "mean":statistics.fmean(delta),
            "ci95":percentile_bootstrap(delta,seed+40),
            "ci97_5":percentile_bootstrap(delta,seed+40,coverage=.975)},
        "constant_hazard_identity_max_abs_error":constant_gap,
        "change_point_posterior_at_actual_switch":(
            statistics.fmean(detected) if detected else None),
        "mean_change_point_posterior_on_non_switch_hands":
            statistics.fmean(false_alarm),
        "time_course_delta":[statistics.fmean(v) for v in by_hand_delta],
        "replicate_differences":delta
    }


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--scenario",choices=SCENARIOS,required=True)
    ap.add_argument("--hands",type=int,default=24)
    ap.add_argument("--replicates",type=int,default=12)
    ap.add_argument("--seed",type=int,default=20261018)
    ap.add_argument("--output",default=None)
    a=ap.parse_args()
    result=run(scenario=a.scenario,hands=a.hands,
               replicates=a.replicates,seed=a.seed)
    if a.output:
        Path(a.output).write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({k:v for k,v in result.items()
                      if k not in ("replicate_differences","time_course_delta")},indent=2))


if __name__=="__main__":
    main()
