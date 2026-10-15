"""On-policy long-horizon value of information in the frozen Leduc game.

Three arms use matched chance cards and matched *probe opportunities*, but
each has separate, legally-observed Bayesian learning trajectories.

Only terminal PlayerObservation enters the learner. The opponent's true
policy enters the OFFLINE exact expected-value evaluator, never an agent.
The exploration event is a secret prehand lottery, not adaptive cheating.
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from pathlib import Path

from .bayes import ActionBiasPolicy,example_models
from .cfr import PolicyAgent,TabularPolicy,best_response_mixture,profile_value
from .game import Action
from .experiment import HELD_OUT_POLICIES
from .qt_safe_exact import exact_safe_weight
from .qt_v4 import SwitchingBeliefs
from .qt_v6_changepoint_eval import percentile_bootstrap
from .simulate import run_hand

ARMS=("no_probe","matched_cfr","active_probe")
SCENARIOS=("stationary_calling","switch_calling_aggressive","heldout_rank_selective")


def _expected_earnings(policy,opponent,seat):
    return (profile_value(policy,opponent) if seat==0
            else -profile_value(opponent,policy))


def risk_constrained_probe_probability(base,probe,budget=.05):
    """One-hand CFR/probe-lottery only; NOT a guarantee for adaptive BR arms."""
    return min(exact_safe_weight(base,probe,seat,budget,bisection_steps=10).weight
               for seat in (0,1))


def run(*,scenario,hands=24,replicates=4,seed=20261026,
        strategy="strategy.json",keep_rows=False):
    if scenario not in SCENARIOS or hands<8 or replicates<2:
        raise ValueError("Unknown scenario or invalid experiment size")
    base=TabularPolicy.load(strategy)
    types=example_models(base)
    policies=tuple(types.values())
    probe=ActionBiasPolicy(base,{
        Action.CHECK:.7,Action.CALL:.7,
        Action.BET:2.5,Action.RAISE:2.5})
    prob=risk_constrained_probe_probability(base,probe,budget=.05)
    heldout=HELD_OUT_POLICIES["rank_selective"](base)
    total={arm:[] for arm in ARMS}
    sampled={arm:[] for arm in ARMS}
    phase={part:{arm:[] for arm in ARMS}
           for part in ("first_half","second_half","probe_events","non_probe_events")}
    events=[]
    rows=[]
    clock=time.monotonic()
    rng=random.Random(seed)

    for replicate in range(replicates):
        perf={arm:[] for arm in ARMS}
        payout={arm:[] for arm in ARMS}
        this_phase={part:{arm:[] for arm in ARMS} for part in phase}
        n_events=0
        for seat in (0,1):
            beliefs={arm:SwitchingBeliefs(types,hazard=.1) for arm in ARMS}
            for hand in range(hands):
                handseed=rng.randrange(1<<63)
                opportunity=random.Random(
                    f"v7:{seed}:{replicate}:{seat}:{hand}").random()<prob
                n_events+=int(opportunity)
                opponent=(heldout if scenario=="heldout_rank_selective"
                          else types["calling"] if scenario=="stationary_calling"
                          or hand<hands//2 else types["aggressive"])
                part="first_half" if hand<hands//2 else "second_half"
                plays={}
                for arm in ARMS:
                    prior=beliefs[arm].posterior
                    if opportunity and arm=="matched_cfr":
                        selected=base
                    elif opportunity and arm=="active_probe":
                        selected=probe
                    else:
                        selected=best_response_mixture(
                            policies,prior,responder=seat)[1]
                    ev=_expected_earnings(selected,opponent,seat)
                    players=((PolicyAgent(selected),PolicyAgent(opponent))
                             if seat==0 else (PolicyAgent(opponent),PolicyAgent(selected)))
                    state=run_hand(handseed,players)
                    reward=state.rewards[seat]
                    # No hidden opponent cards except legally visible showdown cards.
                    beliefs[arm].update(state.observe(seat))
                    perf[arm].append(ev)
                    payout[arm].append(reward)
                    this_phase[part][arm].append(ev)
                    this_phase["probe_events" if opportunity
                               else "non_probe_events"][arm].append(ev)
                    plays[arm]={"ev":ev,"reward":reward,
                                "belief_aggressive_after_hand":
                                    beliefs[arm].named_posterior()["aggressive"]}
                if keep_rows:
                    rows.append({"replicate":replicate,"seat":seat,
                                 "hand":hand,"opportunity":opportunity,
                                 "phase":part,"arms":plays})
        events.append(n_events)
        for arm in ARMS:
            total[arm].append(statistics.fmean(perf[arm]))
            sampled[arm].append(statistics.fmean(payout[arm]))
        for part in phase:
            for arm in ARMS:
                v=this_phase[part][arm]
                phase[part][arm].append(statistics.fmean(v) if v else None)

    contrasts={}
    for a,b in (("active_probe","matched_cfr"),
                ("active_probe","no_probe"),
                ("matched_cfr","no_probe")):
        diffs=[x-y for x,y in zip(total[a],total[b])]
        contrasts[f"{a}_minus_{b}"]={
            "mean":statistics.fmean(diffs),
            "ci95":percentile_bootstrap(diffs,seed+17)}
    attribution={}
    for part in phase:
        values={}
        for arm in ARMS:
            v=[x for x in phase[part][arm] if x is not None]
            values[arm]=statistics.fmean(v) if v else None
        diffs=[a-b for a,b in zip(phase[part]["active_probe"],
                                 phase[part]["matched_cfr"])
               if a is not None and b is not None]
        attribution[part]={
            "mean_EV":values,
            "active_minus_matched":statistics.fmean(diffs) if diffs else None,
            "n_nonempty_blocks":len(diffs)}
    return {
        "protocol":{"scenario":scenario,"hands_per_seat":hands,
                    "replicates":replicates,"seed":seed,
                    "arms":list(ARMS),"safe_probe_probability":prob,
                    "unit":"independent replicate averaging two seats",
                    "oracle":"evaluates true synthetic opponent offline only",
                    "warning":"CFR/probe safety NOT a safety guarantee for an adaptive-BR hybrid"},
        "mean_exact_conditional_EV":{
            arm:statistics.fmean(v) for arm,v in total.items()},
        "mean_sampled_chips":{
            arm:statistics.fmean(v) for arm,v in sampled.items()},
        "paired_contrasts":contrasts,
        "attribution_by_phase":attribution,
        "mean_probe_opportunities_per_two_seat_match":statistics.fmean(events),
        "probe_counts_by_replicate":events,
        "runtime_seconds":round(time.monotonic()-clock,3),
        "replicate_exact_EV":total,
        "replicate_sampled_chips":sampled,
        "rows":rows if keep_rows else None}


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--scenario",choices=SCENARIOS,required=True)
    p.add_argument("--hands",type=int,default=24)
    p.add_argument("--replicates",type=int,default=4)
    p.add_argument("--seed",type=int,default=20261026)
    p.add_argument("--rows",action="store_true")
    p.add_argument("--output",default=None)
    a=p.parse_args()
    out=run(scenario=a.scenario,hands=a.hands,replicates=a.replicates,
            seed=a.seed,keep_rows=a.rows)
    if a.output:
        Path(a.output).write_text(json.dumps(out,indent=2)+"\n")
    print(json.dumps({k:v for k,v in out.items()
                      if k not in ("replicate_exact_EV","replicate_sampled_chips","rows")},
                     indent=2))


if __name__=="__main__":
    main()
