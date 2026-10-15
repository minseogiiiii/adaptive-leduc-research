"""QT V5: predeclared common-evidence predictive validation.

This diagnostic tests the inference component BEFORE using it in trading.
Each evaluator agent sees the SAME legally available terminal trajectories,
generated under a fixed CFR probe policy. Only the independent replicate
block, not individual hands, is a unit for the confidence interval.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import time
from pathlib import Path

from .bayes import BayesianOpponentModel, _hand_likelihood, example_models
from .cfr import PolicyAgent, TabularPolicy
from .experiment import HELD_OUT_POLICIES
from .qt_v4 import SwitchingBeliefs
from .qt_v5 import EvidenceWeightedBeliefs, gated_risk_budget
from .simulate import run_hand

SCENARIOS = (
    "stationary_calling",
    "stationary_aggressive",
    "switch_calling_aggressive",
    "switch_aggressive_calling",
    "heldout_rank_selective",
    "heldout_pressure_reactive",
)
MODELS = ("fixed", "hazard_01", "evidence")


def _observed_log_probability(beliefs, observation):
    return math.log(math.fsum(p * _hand_likelihood(observation, model)
                              for p, model in zip(beliefs.posterior,
                                                  beliefs.policies)))


def _scenario_opponent(scenario, hand, hands, models, baseline):
    if scenario == "stationary_calling":
        return models["calling"]
    if scenario == "stationary_aggressive":
        return models["aggressive"]
    if scenario == "switch_calling_aggressive":
        return models["calling"] if hand < hands//2 else models["aggressive"]
    if scenario == "switch_aggressive_calling":
        return models["aggressive"] if hand < hands//2 else models["calling"]
    if scenario == "heldout_rank_selective":
        return HELD_OUT_POLICIES["rank_selective"](baseline)
    if scenario == "heldout_pressure_reactive":
        return HELD_OUT_POLICIES["pressure_reactive"](baseline)
    raise ValueError("Unknown scenario")


def _boot_ci(values, *, seed, draws=2000):
    if len(values) < 2:
        raise ValueError("Need at least 2 independent replicates")
    rng = random.Random(seed)
    sims = sorted(statistics.fmean(rng.choices(values,k=len(values)))
                  for _ in range(draws))
    return [sims[int(draws*.025)], sims[int(draws*.975)]]


def run_forecasting(*, scenario, hands=24, replicates=24,
                    seed=20261016, strategy="strategy.json",
                    store_rows=False):
    """Frozen hyperparameters, paired paths, independent match-level CI."""
    if scenario not in SCENARIOS or hands < 4 or replicates < 2:
        raise ValueError("Invalid protocol")
    baseline = TabularPolicy.load(strategy)
    models = example_models(baseline)
    rng = random.Random(seed)
    scores = {name: [] for name in MODELS}
    post_scores = {name: [] for name in MODELS}
    modelprob, budgets = [], []
    rows = []
    start = time.monotonic()

    for rep in range(replicates):
        per_model = {name: [] for name in MODELS}
        post_model = {name: [] for name in MODELS}
        regime_probs = []
        risk_budgets = []
        for seat in (0, 1):
            beliefs = {
                "fixed": BayesianOpponentModel(models),
                "hazard_01": SwitchingBeliefs(models,hazard=.1),
                "evidence": EvidenceWeightedBeliefs(
                    models,change_hazard=.2,change_prior=.25)}
            for hand in range(hands):
                opponent = _scenario_opponent(scenario,hand,hands,models,baseline)
                hand_seed = rng.randrange(1<<63)
                players = ((PolicyAgent(baseline),PolicyAgent(opponent))
                           if seat == 0 else
                           (PolicyAgent(opponent),PolicyAgent(baseline)))
                terminal = run_hand(hand_seed,players)
                observation = terminal.observe(seat)
                ll = {name: _observed_log_probability(b,observation)
                      for name,b in beliefs.items()}
                for name in MODELS:
                    per_model[name].append(ll[name])
                    if hand >= hands//2:
                        post_model[name].append(ll[name])
                gate = beliefs["evidence"]
                regime_probs.append(gate.switch_model_probability)
                risk_budgets.append(gated_risk_budget(gate))
                if store_rows:
                    rows.append({"replicate":rep,"seat":seat,"hand":hand,
                                 "seed":hand_seed,"log_scores":ll,
                                 "switch_prob":gate.switch_model_probability})
                for b in beliefs.values():
                    b.update(observation)
        for name in MODELS:
            scores[name].append(statistics.fmean(per_model[name]))
            post_scores[name].append(statistics.fmean(post_model[name]))
        modelprob.append(statistics.fmean(regime_probs))
        budgets.append(statistics.fmean(risk_budgets))

    contrast = {}
    for phase, data in (("overall",scores),("second_half",post_scores)):
        for control in ("fixed","hazard_01"):
            diff = [a-b for a,b in zip(data["evidence"],data[control])]
            contrast[f"{phase}_evidence_minus_{control}"] = {
                "mean": statistics.fmean(diff),
                "bootstrap_95_percent": _boot_ci(diff,seed=seed+20)}
    return {
        "protocol":{"scenario":scenario,"hands_per_seat":hands,
                    "replicates":replicates,"seed":seed,
                    "common_probe_policy":"existing CFR",
                    "independent_unit":"replicate with both seats",
                    "model_prior_switch":0.25,"candidate_change_hazard":0.2,
                    "comparison_fixed_hazard":0.1,
                    "gated_risk_floor":0.02,"gated_risk_ceiling":0.1,
                    "metric":"predictive log-likelihood per observed hand"},
        "means": {name:statistics.fmean(x) for name,x in scores.items()},
        "second_half_means": {
            name:statistics.fmean(x) for name,x in post_scores.items()},
        "contrasts":contrast,
        "mean_switch_process_probability":statistics.fmean(modelprob),
        "mean_suggested_risk_budget":statistics.fmean(budgets),
        "runtime_seconds":round(time.monotonic()-start,3),
        "rows":rows if store_rows else None}


def main():
    ap = argparse.ArgumentParser(description="V5 common-path inference audit")
    ap.add_argument("--scenario",choices=SCENARIOS,required=True)
    ap.add_argument("--hands",type=int,default=24)
    ap.add_argument("--replicates",type=int,default=24)
    ap.add_argument("--seed",type=int,default=20261016)
    ap.add_argument("--strategy",default="strategy.json")
    ap.add_argument("--rows",action="store_true")
    ap.add_argument("--output",default=None)
    a = ap.parse_args()
    result=run_forecasting(scenario=a.scenario,hands=a.hands,
                           replicates=a.replicates,seed=a.seed,
                           strategy=a.strategy,store_rows=a.rows)
    if a.output:
        Path(a.output).write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k!="rows"},indent=2))


if __name__=="__main__":
    main()
