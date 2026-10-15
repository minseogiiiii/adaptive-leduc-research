"""QT V5: evidence-weighted model averaging for changing opponent strategies.

Two models compete sequentially on legally observed hands:
H0: type fixed for all hands (hazard=0)
H1: type redrawn toward original prior at each hand boundary (hazard=0.2).

Bayesian evidence compares the *predictive likelihood* of each observation,
not the realized chips. No oracle opponent policy or hidden card is used.
This module is an independent extension of V4 and changes no frozen rules.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

from .bayes import _hand_likelihood
from .qt_v4 import DecisionAgent


def logsumexp(values):
    maximum = max(values)
    if maximum == -math.inf:
        return -math.inf
    return maximum + math.log(math.fsum(math.exp(v - maximum) for v in values))


class EvidenceWeightedBeliefs:
    """Posterior mixture of stationary and change-prone opponent processes.

    Each H model maintains P(z_t|history up to t-1) before each hand. At the
    hand end, compute evidence p(hand | H), update H posterior, update types,
    then propagate types toward initial prior at the *next* hand boundary.
    """
    def __init__(self, models: Mapping, *, change_hazard=0.2,
                 change_prior=0.25, prior=None):
        if not models or not math.isfinite(change_hazard) or not 0 <= change_hazard <= 1:
            raise ValueError("Invalid candidate models or hazard")
        if not math.isfinite(change_prior) or not 0 < change_prior < 1:
            raise ValueError("Model prior must lie strictly between 0 and 1")
        self.names = tuple(models)
        self.policies = tuple(models.values())
        n = len(self.policies)
        self.initial = tuple(float(x) for x in (
            prior if prior is not None else (1.0/n for _ in range(n))))
        if len(self.initial)!=n or any(not math.isfinite(x) or x<=0 for x in self.initial) or abs(math.fsum(self.initial)-1)>1e-10:
            raise ValueError("Invalid type prior")
        self.change_hazard = float(change_hazard)
        self._type_p0 = self.initial
        self._type_p1 = self.initial
        self._model_p1 = float(change_prior)
        self.hands_seen = 0
        self.last_log_likelihood = None
        self.history = []

    @property
    def switch_model_probability(self):
        return self._model_p1

    @property
    def posterior(self):
        w = self._model_p1
        return tuple((1-w)*a + w*b for a,b in zip(self._type_p0,self._type_p1))

    def named_posterior(self):
        return dict(zip(self.names, self.posterior))

    def confidence(self):
        """Type posterior concentration only; NOT a calibration guarantee."""
        p = self.posterior
        if len(p)==1:
            return 1.0
        entropy = -math.fsum(x * math.log(x) for x in p if x)
        return max(0.0,min(1.0,1 - entropy/math.log(len(p))))

    def predictive_log_score(self, terminal):
        """Compute log P(observed hand | current predictive prior)."""
        likelihoods = [_hand_likelihood(terminal,m) for m in self.policies]
        evidence = math.fsum(w*l for w,l in zip(self.posterior, likelihoods))
        if evidence<=0 or not math.isfinite(evidence):
            raise ValueError("Impossible observation under the model")
        return math.log(evidence)

    def update(self, terminal):
        ls = [_hand_likelihood(terminal,m) for m in self.policies]
        if any(not math.isfinite(x) or x < 0 for x in ls):
            raise ValueError("Invalid opponent likelihood")
        def bayesian_type_update(prior):
            products = [w*l for w,l in zip(prior,ls)]
            evidence = math.fsum(products)
            if evidence<=0:
                raise ValueError("Hand impossible under hypothesis")
            return evidence, tuple(x/evidence for x in products)
        ev0, filtered0 = bayesian_type_update(self._type_p0)
        ev1, filtered1 = bayesian_type_update(self._type_p1)
        w1 = self._model_p1
        evidence = (1-w1)*ev0+w1*ev1
        self.last_log_likelihood = math.log(evidence)
        self._model_p1 = w1*ev1/evidence
        self._type_p0 = filtered0
        h = self.change_hazard
        self._type_p1 = tuple((1-h)*p+h*base
                              for p,base in zip(filtered1,self.initial))
        self.hands_seen+=1
        self.history.append({
            "hand":self.hands_seen,
            "switch_probability":self._model_p1,
            "type_confidence":self.confidence(),
            "log_predictive_score":self.last_log_likelihood,
        })
        return self.posterior


def gated_risk_budget(beliefs, *, floor=0.02, ceiling=0.1):
    """Predeclared conservative trade-off; budgets are absolute chips/hand.

    Higher type confidence + stronger stationary evidence permit more
    exploitation. This is a *heuristic* on top of an exact worst-case
    single-hand payoff constraint, not a probabilistic bankroll guarantee.
    """
    if not 0<=floor<=ceiling or not math.isfinite(ceiling):
        raise ValueError("Invalid risk floor/ceiling")
    reliability = beliefs.confidence()*(1-beliefs.switch_model_probability)
    return floor+(ceiling-floor)*max(0.0,min(1.0,reliability))


class EvidenceDecisionAgent(DecisionAgent):
    """Same Leduc exact-response machinery; different sequential beliefs.

    safe=False: unrestricted mixture best-response.
    safe=True: exact-safe whole-hand mixture with dynamically gated budget.
    """
    def __init__(self, seat, baseline, models, *, safe=False,
                 change_hazard=0.2, change_prior=0.25,
                 risk_floor=0.02, risk_ceiling=0.1):
        super().__init__("safe_exact" if safe else "switching", seat,
                         baseline, models, budget=risk_ceiling)
        self.beliefs = EvidenceWeightedBeliefs(
            models,change_hazard=change_hazard,change_prior=change_prior)
        self.gated_safe = safe
        self.risk_floor = risk_floor
        self.risk_ceiling = risk_ceiling
        self.budget_history=[]

    def prepare(self):
        if self.gated_safe:
            self.budget=gated_risk_budget(
                self.beliefs,floor=self.risk_floor,ceiling=self.risk_ceiling)
            self.budget_history.append(self.budget)
        super().prepare()
