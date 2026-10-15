"""Tests for exact finite-type changepoint inference and V4 identity."""
import math
import unittest

from pokerlab.bayes import example_models
from pokerlab.cfr import PolicyAgent,TabularPolicy
from pokerlab.qt_v4 import SwitchingBeliefs
from pokerlab.qt_v6_changepoint import RunLengthBeliefs
from pokerlab.qt_v6_changepoint_eval import run
from pokerlab.simulate import run_hand


class ChangepointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base=TabularPolicy.load("strategy.json")
        cls.types=example_models(cls.base)

    def test_exact_constant_hazard_marginal_equivalence(self):
        for h in (0,.03,.1,.5,1):
            old=SwitchingBeliefs(self.types,hazard=h)
            new=RunLengthBeliefs(self.types,hazard=h)
            for seed in range(2,12):
                terminal=run_hand(seed,(PolicyAgent(self.base),
                                        PolicyAgent(self.types["aggressive"])))
                obs=terminal.observe(0)
                self.assertAlmostEqual(
                    sum(new.run_length_posterior.values()),1,places=10)
                self.assertLess(
                    max(abs(a-b) for a,b in zip(old.posterior,new.posterior)),1e-10)
                old.update(obs)
                new.update(obs)
                self.assertLess(
                    max(abs(a-b) for a,b in zip(old.posterior,new.posterior)),1e-10)

    def test_change_probability_is_never_available_before_observation(self):
        p=RunLengthBeliefs(self.types,hazard_mode="duration")
        self.assertEqual(p.prehand_change_probability,0)
        self.assertEqual(len(p.change_history),0)
        obs=run_hand(5,(PolicyAgent(self.base),
                        PolicyAgent(self.types["calling"]))).observe(0)
        p.update(obs)
        self.assertEqual(p.last_change_posterior,0)
        self.assertEqual(p.hands_seen,1)
        self.assertTrue(0<=p.prehand_change_probability<=1)

    def test_duration_filter_invariants(self):
        b=RunLengthBeliefs(self.types,hazard_mode="duration",
                           short_hazard=.03,long_hazard=.2,duration_knee=6)
        for seed in range(10,42):
            opp=self.types["calling"] if seed<26 else self.types["aggressive"]
            obs=run_hand(seed,(PolicyAgent(self.base),
                               PolicyAgent(opp))).observe(0)
            b.update(obs)
            self.assertAlmostEqual(sum(b.posterior),1,places=10)
            self.assertAlmostEqual(sum(b.run_length_posterior.values()),1,places=10)
            self.assertTrue(0<=b.last_change_posterior<=1)

    def test_small_reproducible_matched_paths(self):
        args=dict(scenario="switch_calling_aggressive",hands=8,
                  replicates=2,seed=1741)
        a,b=run(**args),run(**args)
        self.assertEqual(a["primary_delta_duration_minus_v4"],
                         b["primary_delta_duration_minus_v4"])
        self.assertLess(a["constant_hazard_identity_max_abs_error"],1e-10)


if __name__=="__main__":
    unittest.main()
