"""QT V5 on-policy Leduc research benchmark, not a market backtest.

Five policies play real legal Leduc hands with paired chance seeds. A separate
exact oracle evaluates conditional EV only for reporting; agents do not see it.
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from pathlib import Path

from .bayes import example_models
from .cfr import TabularPolicy, PolicyAgent
from .qt_v4 import DecisionAgent
from .qt_v5 import EvidenceDecisionAgent
from .qt_v5_forecast import _scenario_opponent, _boot_ci, SCENARIOS
from .simulate import run_hand


def run_decision(*, scenario, hands=12, replicates=6, seed=20261016,
                 strategy="strategy.json"):
    if scenario not in SCENARIOS or hands < 4 or replicates < 2:
        raise ValueError("Invalid scenario or experiment size")
    base=TabularPolicy.load(strategy)
    models=example_models(base)
    arms=("adaptive","switching","safe_exact","evidence","evidence_safe")
    rng=random.Random(seed)
    rewards={name:[] for name in arms}
    oracles={name:[] for name in arms}
    phase_oracles={phase:{arm:[] for arm in arms} for phase in ("pre","post")}
    meanbudgets=[]
    safe_margins=[]
    runtimes={name:[] for name in arms}
    start=time.monotonic()

    for rep in range(replicates):
        rw={name:[] for name in arms}
        ev={name:[] for name in arms}
        phase_ev={phase:{arm:[] for arm in arms} for phase in phase_oracles}
        for seat in (0,1):
            agents={
                "adaptive":DecisionAgent("adaptive",seat,base,models),
                "switching":DecisionAgent("switching",seat,base,models,hazard=.1),
                "safe_exact":DecisionAgent("safe_exact",seat,base,models,hazard=.1,budget=.1),
                "evidence":EvidenceDecisionAgent(seat,base,models,safe=False),
                "evidence_safe":EvidenceDecisionAgent(
                    seat,base,models,safe=True)}
            for hand in range(hands):
                opponent=_scenario_opponent(scenario,hand,hands,models,base)
                handseed=rng.randrange(1<<63)
                phase="pre" if hand < hands//2 else "post"
                for name in arms:
                    agent=agents[name]
                    t0=time.monotonic()
                    agent.prepare()
                    oracle=agent.exact_expected_value(opponent)
                    runtimes[name].append(time.monotonic()-t0)
                    if agent.certificate is not None:
                        safe_margins.append(
                            agent.certificate.certified_lower_bound-
                            agent.certificate.allowed_floor)
                    if name=="evidence_safe":
                        meanbudgets.append(agent.budget)
                    players=((agent,PolicyAgent(opponent)) if seat==0
                             else (PolicyAgent(opponent),agent))
                    final=run_hand(handseed,players)
                    rw[name].append(final.rewards[seat])
                    ev[name].append(oracle)
                    phase_ev[phase][name].append(oracle)
                    agent.finish_hand(final.observe(seat))
        for name in arms:
            rewards[name].append(statistics.fmean(rw[name]))
            oracles[name].append(statistics.fmean(ev[name]))
            for phase in phase_oracles:
                phase_oracles[phase][name].append(statistics.fmean(phase_ev[phase][name]))

    contrasts={}
    for candidate, control in (
            ("evidence","switching"),("evidence","adaptive"),
            ("evidence_safe","safe_exact"),("evidence_safe","evidence")):
        diffs=[x-y for x,y in zip(oracles[candidate],oracles[control])]
        contrasts[f"{candidate}_minus_{control}"]={
            "mean":statistics.fmean(diffs),
            "ci95":_boot_ci(diffs,seed=seed+5),
            "ci97_5":_boot_ci(diffs,seed=seed+5,coverage=.975)}
    phase_contrast={}
    for phase in phase_oracles:
        a=phase_oracles[phase]["evidence"]
        b=phase_oracles[phase]["switching"]
        diffs=[x-y for x,y in zip(a,b)]
        phase_contrast[phase]={
            "mean":statistics.fmean(diffs),
            "ci95":_boot_ci(diffs,seed=seed+6)}
    return {
        "protocol":{"scenario":scenario,"replicates":replicates,
                    "hands_per_seat":hands,"seed":seed,
                    "arms":list(arms),"primary":"evidence_minus_switching",
                    "independent_unit":"replicate mean across both seats",
                    "risk_gate":"0.02..0.1",
                    "note":"Synthetic frozen Leduc; oracle never used in decisions"},
        "exact_conditional_ev":{k:statistics.fmean(v) for k,v in oracles.items()},
        "realized_chips_per_hand":{k:statistics.fmean(v) for k,v in rewards.items()},
        "paired_exact_ev":contrasts,
        "by_phase_evidence_minus_switching":phase_contrast,
        "mean_gated_budget":statistics.fmean(meanbudgets),
        "safe_min_margin":min(safe_margins),
        "mean_prepare_and_exact_eval_seconds":{
            k:statistics.fmean(v) for k,v in runtimes.items()},
        "runtime_seconds":round(time.monotonic()-start,3),
        "replicate_exact_values":oracles,
        "replicate_realized_values":rewards
    }


def main():
    ap=argparse.ArgumentParser(description="QT V5 on-policy Leduc benchmark")
    ap.add_argument("--scenario",required=True,choices=SCENARIOS)
    ap.add_argument("--hands",type=int,default=12)
    ap.add_argument("--replicates",type=int,default=6)
    ap.add_argument("--seed",type=int,default=20261018)
    ap.add_argument("--output",default=None)
    a=ap.parse_args()
    r=run_decision(scenario=a.scenario,hands=a.hands,
                   replicates=a.replicates,seed=a.seed)
    if a.output:
        Path(a.output).write_text(json.dumps(r,indent=2)+"\n")
    print(json.dumps({k:v for k,v in r.items()
                      if not k.startswith("replicate_")},indent=2))


if __name__=="__main__":
    main()
