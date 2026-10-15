"""Exact finite-type BOCPD for opponent-policy change points in frozen Leduc.

State is joint P(run_age, opponent_type | legally observed prior hands).
At a hand boundary the opponent type can redraw from the declared prior.

IMPORTANT mathematical control: with CONSTANT hazard h, marginalizing run_age
exactly reproduces the V4 SwitchingBeliefs(h) type posterior. Exposing a
run-length posterior alone cannot improve prediction or profit under those
same assumptions. To change predictions the *generative assumption* must
change, such as a duration-dependent hazard.
"""
from __future__ import annotations

import math
from typing import Mapping

from .bayes import _hand_likelihood


class RunLengthBeliefs:
    """Online changepoint filter over (run age, policy type).

    Pre-hand joint mass includes any reset between the last observed hand
    and the current one. Hand completion updates evidence then transitions
    to the next pre-hand state. The opponent's type remains fixed WITHIN a
    hand. There is no hidden-card access.
    """
    def __init__(self, models: Mapping, *, hazard=0.1, type_prior=None,
                 hazard_mode="constant", short_hazard=0.03,
                 long_hazard=0.2, duration_knee=6):
        if not models:
            raise ValueError("Require models")
        self.names=tuple(models)
        self.policies=tuple(models.values())
        n=len(self.policies)
        self.initial_prior=tuple(float(x) for x in (
            type_prior if type_prior is not None else
            (1/n for _ in range(n))))
        if (len(self.initial_prior)!=n or
            any(not math.isfinite(p) or p<=0 for p in self.initial_prior)
            or abs(math.fsum(self.initial_prior)-1)>1e-10):
            raise ValueError("Invalid opponent prior")
        if hazard_mode not in ("constant","duration"):
            raise ValueError("Hazard mode must be constant or duration")
        if any(not math.isfinite(v) or not 0<=v<=1
               for v in (hazard,short_hazard,long_hazard)):
            raise ValueError("Invalid hazards")
        if not isinstance(duration_knee,int) or duration_knee<1:
            raise ValueError("Invalid knee")
        self.hazard=float(hazard)
        self.hazard_mode=hazard_mode
        self.short_hazard=float(short_hazard)
        self.long_hazard=float(long_hazard)
        self.duration_knee=duration_knee
        self._joint={(0,k):p for k,p in enumerate(self.initial_prior)}
        self.hands_seen=0
        self.last_log_score=None
        self.last_change_posterior=None
        self.change_history=[]

    def _hazard(self,age):
        if self.hazard_mode=="constant":
            return self.hazard
        return (self.short_hazard if age<self.duration_knee
                else self.long_hazard)

    @property
    def posterior(self):
        mass=[0.0]*len(self.policies)
        for (_,k),v in self._joint.items():
            mass[k]+=v
        return tuple(mass)

    @property
    def run_length_posterior(self):
        mass={}
        for (age,_),p in self._joint.items():
            mass[age]=mass.get(age,0.0)+p
        return mass

    @property
    def prehand_change_probability(self):
        """P(new segment at CURRENT hand boundary | history before hand)."""
        if not self.hands_seen:
            return 0.0  # initial hand is not classified as a switch
        return self.run_length_posterior.get(0,0.0)

    def update(self,observation):
        ls=tuple(_hand_likelihood(observation,p) for p in self.policies)
        if any(not math.isfinite(x) or x<0 for x in ls):
            raise ValueError("Invalid hand likelihood")
        products={(age,k):mass*ls[k]
                  for (age,k),mass in self._joint.items()}
        evidence=math.fsum(products.values())
        if evidence<=0:
            raise ValueError("Hand impossible under all modeled types")
        self.last_log_score=math.log(evidence)
        conditioned={state:p/evidence for state,p in products.items()}
        # This probability describes the changepoint hypothesis AT the
        # hand just observed; it must not be used for that hand's decision.
        self.last_change_posterior=(sum(
            p for (age,_),p in conditioned.items() if age==0)
            if self.hands_seen else 0.0)
        self.change_history.append(self.last_change_posterior)
        upcoming={}
        for (age,k),p in conditioned.items():
            h=self._hazard(age)
            key=(age+1,k)
            upcoming[key]=upcoming.get(key,0.0)+p*(1-h)
            if h:
                for j,prior in enumerate(self.initial_prior):
                    key=(0,j)
                    upcoming[key]=upcoming.get(key,0.0)+p*h*prior
        self._joint={key:value for key,value in upcoming.items() if value>0}
        self.hands_seen+=1
        if abs(math.fsum(self._joint.values())-1)>1e-9:
            raise AssertionError("BOCPD state lost probability mass")
        return self.posterior

    def named_posterior(self):
        return dict(zip(self.names,self.posterior))
