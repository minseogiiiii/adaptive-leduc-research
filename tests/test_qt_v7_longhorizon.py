"""On-policy long-horizon exploration safety and reproducibility checks."""
import unittest
from unittest.mock import patch

from pokerlab.qt_v7_longhorizon import run,risk_constrained_probe_probability
from pokerlab.bayes import ActionBiasPolicy
from pokerlab.cfr import TabularPolicy
from pokerlab.game import Action


class V7Tests(unittest.TestCase):
    def test_no_probe_zero_event_identity(self):
        with patch("pokerlab.qt_v7_longhorizon.risk_constrained_probe_probability",
                   return_value=0.0):
            r=run(scenario="stationary_calling",hands=8,replicates=2,seed=119)
        e=r["mean_exact_conditional_EV"]
        self.assertAlmostEqual(e["no_probe"],e["matched_cfr"],places=12)
        self.assertAlmostEqual(e["no_probe"],e["active_probe"],places=12)
        self.assertEqual(sum(r["probe_counts_by_replicate"]),0)
        self.assertAlmostEqual(
            r["paired_contrasts"]["active_probe_minus_matched_cfr"]["mean"],
            0.0,places=12)

    def test_matched_randomness_replay(self):
        with patch("pokerlab.qt_v7_longhorizon.risk_constrained_probe_probability",
                   return_value=0.5):
            a=run(scenario="switch_calling_aggressive",hands=8,
                  replicates=2,seed=190)
            b=run(scenario="switch_calling_aggressive",hands=8,
                  replicates=2,seed=190)
        self.assertEqual(a["paired_contrasts"],b["paired_contrasts"])
        self.assertEqual(a["probe_counts_by_replicate"],
                         b["probe_counts_by_replicate"])
        self.assertEqual(a["mean_sampled_chips"],b["mean_sampled_chips"])

    def test_safe_probe_lottery_bounds(self):
        base=TabularPolicy.load("strategy.json")
        probe=ActionBiasPolicy(base,{
            Action.CHECK:.7,Action.CALL:.7,
            Action.BET:2.5,Action.RAISE:2.5})
        weight=risk_constrained_probe_probability(base,probe)
        self.assertTrue(0<=weight<=1)


if __name__=="__main__":
    unittest.main()
