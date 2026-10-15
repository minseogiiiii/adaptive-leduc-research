"""Active probing and single-hand change identifiability diagnostic.

Independent fresh-seed follow-up to failed V6 A1 changepoint prior. Uses only
legally observed terminal trajectories and an offline type-label evaluator.
The probe policy is FIXED before each hand; no true opponent type in agent.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from pathlib import Path

from .bayes import ActionBiasPolicy, _hand_likelihood, example_models
from .cfr import PolicyAgent, TabularPolicy, UniformPolicy, profile_value
from .game import Action
from .simulate import run_hand
from .qt_v6_changepoint_eval import percentile_bootstrap

PROBES=("cfr","uniform","aggressive_probe")


def _auc(high,low):
    n=len(high)*len(low)
    return sum((1.0 if a>b else 0.5 if a==b else 0.0)
               for a in high for b in low)/n


def _likelihood_score(obs,models):
    a=_hand_likelihood(obs,models["aggressive"])
    c=_hand_likelihood(obs,models["calling"])
    if a<=0 or c<=0:
        raise ValueError("No full-support likelihood")
    return math.log(a/c)


def _payoff(our,opp,seat):
    return (profile_value(our,opp) if seat==0
            else -profile_value(opp,our))


def run(*,seed=20261022,replicates=24,hands_per_type=48,
        strategy="strategy.json"):
    if replicates<2 or hands_per_type<2:
        raise ValueError("Insufficient experiments")
    base=TabularPolicy.load(strategy)
    models=example_models(base)
    policies={
        "cfr":base,
        "uniform":UniformPolicy(),
        "aggressive_probe":ActionBiasPolicy(base,{
            Action.CHECK:.7,Action.CALL:.7,
            Action.BET:2.5,Action.RAISE:2.5})
    }
    # Exact game payoff of each probe policy, separate from sampled
    # information metrics. Same game/strategy across all comparisons.
    evs={
        str(seat):{
            probe:.5*_payoff(policy,models["calling"],seat)
                 +.5*_payoff(policy,models["aggressive"],seat)
            for probe,policy in policies.items()}
        for seat in (0,1)}
    r=random.Random(seed)
    per={probe:{"auc":[],"accuracy":[],"sensitivity":[],"false_positive":[],
                "jdiv":[]}
         for probe in PROBES}
    lr_threshold=math.log(9)
    for block in range(replicates):
        score={probe:{"aggressive":[],"calling":[]} for probe in PROBES}
        for seat in (0,1):
            for type_name in ("aggressive","calling"):
                opponent=models[type_name]
                for _ in range(hands_per_type):
                    handseed=r.randrange(1<<63)
                    for probe,policy in policies.items():
                        pair=((PolicyAgent(policy),PolicyAgent(opponent)) if seat==0
                              else (PolicyAgent(opponent),PolicyAgent(policy)))
                        obs=run_hand(handseed,pair).observe(seat)
                        score[probe][type_name].append(
                            _likelihood_score(obs,models))
        for probe in PROBES:
            hi=score[probe]["aggressive"]
            lo=score[probe]["calling"]
            accuracy=.5*(sum(1 if s>0 else .5 if s==0 else 0 for s in hi)/len(hi)
                          +sum(1 if s<0 else .5 if s==0 else 0 for s in lo)/len(lo))
            per[probe]["auc"].append(_auc(hi,lo))
            per[probe]["accuracy"].append(accuracy)
            per[probe]["sensitivity"].append(
                sum(s>lr_threshold for s in hi)/len(hi))
            per[probe]["false_positive"].append(
                sum(s>lr_threshold for s in lo)/len(lo))
            per[probe]["jdiv"].append(
                .5*(statistics.fmean(hi)-statistics.fmean(lo)))
    summary={}
    for probe in PROBES:
        summary[probe]={k:statistics.fmean(v) for k,v in per[probe].items()}
    contrasts={}
    for probe in ("uniform","aggressive_probe"):
        contrasts[probe]={}
        for metric in ("auc","accuracy"):
            diffs=[a-b for a,b in zip(per[probe][metric],
                                      per["cfr"][metric])]
            contrasts[probe][metric]={
                "delta":statistics.fmean(diffs),
                "ci95":percentile_bootstrap(diffs,seed+31)}
    return {
        "protocol":{"seed":seed,"replicates":replicates,
                    "hands_per_type_per_seat_per_replicate":hands_per_type,
                    "total_trajectories":replicates*hands_per_type*2*2*3,
                    "unit":"replicate with pooled two seats",
                    "fixed_type_prior":[.5,.5],
                    "switch_prior":.1,
                    "switch_posterior_threshold_LLR":lr_threshold,
                    "note":"Optimistic detector KNOWS before-and-after candidate types."},
        "probe_metrics":summary,"contrasts":contrasts,
        "exact_expected_chips_per_hand_by_seat":evs,
        "replicate_metrics":per
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--seed",type=int,default=20261022)
    p.add_argument("--replicates",type=int,default=24)
    p.add_argument("--hands",type=int,default=48)
    p.add_argument("--output",default=None)
    args=p.parse_args()
    data=run(seed=args.seed,replicates=args.replicates,
             hands_per_type=args.hands)
    if args.output:
        Path(args.output).write_text(json.dumps(data,indent=2)+"\n")
    print(json.dumps({k:v for k,v in data.items() if k!="replicate_metrics"},indent=2))


if __name__=="__main__":
    main()
