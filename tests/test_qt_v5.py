"""Mathematical unit and integration checks for the QT V5 inference gate."""
from __future__ import annotations

import math
import unittest

from pokerlab.bayes import BayesianOpponentModel, example_models
from pokerlab.cfr import PolicyAgent, TabularPolicy
from pokerlab.qt_v4 import SwitchingBeliefs
from pokerlab.qt_v5 import (EvidenceWeightedBeliefs, EvidenceDecisionAgent,
                            gated_risk_budget)
from pokerlab.qt_v5_forecast import run_forecasting
from pokerlab.simulate import run_hand


class QT5Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base = TabularPolicy.load("strategy.json")
        cls.models = example_models(cls.base)

    def obs(self, seed=31):
        state = run_hand(seed, (PolicyAgent(self.base),
                                 PolicyAgent(self.models["calling"])))
        return state.observe(0)

    def test_all_zero_hazard_equals_original_bayesian(self):
        fixed = BayesianOpponentModel(self.models)
        evidence = EvidenceWeightedBeliefs(self.models,change_hazard=0)
        for seed in (4,5,6,7):
            a,b=fixed.update(self.obs(seed)),evidence.update(self.obs(seed))
            for x,y in zip(a,b):
                self.assertAlmostEqual(x,y,places=10)
            self.assertAlmostEqual(evidence.switch_model_probability,.25,places=10)

    def test_first_hand_hypotheses_indistinguishable(self):
        belief=EvidenceWeightedBeliefs(self.models)
        belief.update(self.obs(41))
        self.assertAlmostEqual(belief.switch_model_probability,.25,places=10)

    def test_posterior_and_risk_budget_are_valid(self):
        belief=EvidenceWeightedBeliefs(self.models)
        for seed in range(55,65):
            belief.update(self.obs(seed))
            self.assertAlmostEqual(sum(belief.posterior),1.0)
            self.assertTrue(all(0<=x<=1 for x in belief.posterior))
            self.assertTrue(0<=belief.switch_model_probability<=1)
            self.assertTrue(.02<=gated_risk_budget(belief)<= .1+1e-10)

    def test_unseen_hands_cannot_be_used_to_update(self):
        belief=EvidenceWeightedBeliefs(self.models)
        self.assertEqual(belief.hands_seen,0)
        self.assertEqual(len(belief.history),0)

    def test_evidence_safe_agent_uses_only_prehand_risk_budget(self):
        agent=EvidenceDecisionAgent(0,self.base,self.models,safe=True)
        predicted=gated_risk_budget(agent.beliefs)
        agent.prepare()
        self.assertAlmostEqual(agent.budget,predicted)
        self.assertGreaterEqual(agent.certificate.certified_lower_bound,
                                agent.certificate.allowed_floor-1e-8)
        game=run_hand(41,(agent,PolicyAgent(self.models["aggressive"])))
        agent.finish_hand(game.observe(0))
        self.assertEqual(agent.beliefs.hands_seen,1)

    def test_reproducible_probe_forecast(self):
        args=dict(scenario="switch_calling_aggressive",
                  hands=4,replicates=2,seed=991)
        x,y=run_forecasting(**args),run_forecasting(**args)
        self.assertEqual(x["means"],y["means"])
        self.assertEqual(x["contrasts"],y["contrasts"])

    def test_rejects_malformed_hyperparameters(self):
        with self.assertRaises(ValueError):
            EvidenceWeightedBeliefs(self.models,change_hazard=-.1)
        with self.assertRaises(ValueError):
            EvidenceWeightedBeliefs(self.models,change_prior=1.0)
        with self.assertRaises(ValueError):
            gated_risk_budget(EvidenceWeightedBeliefs(self.models),
                              floor=.2,ceiling=.1)


if __name__=="__main__":
    unittest.main()
