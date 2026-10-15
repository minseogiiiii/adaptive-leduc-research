"""Exact strategic risk frontier validations."""
import unittest

from pokerlab.qt_v6_frontier import single_frontier


class RiskFrontierTests(unittest.TestCase):
    def test_exact_monotone_and_certified(self):
        r=single_frontier(seat=0,belief_case="calling_concentrated",
                          budgets=(0.0,.1,.4),search_steps=3)
        x=r["rows"]
        self.assertEqual(len(x),3)
        self.assertAlmostEqual(x[0]["posterior_ev_exact"],r["base_ev"],places=7)
        for row in x:
            self.assertGreaterEqual(row["worst_exact"],
                                    row["risk_floor"]-1e-8)
            self.assertGreaterEqual(row["weight_exact"]+1e-10,
                                    row["weight_conservative"])
            self.assertGreaterEqual(row["gain_exact_over_conservative"],-1e-8)
        self.assertLessEqual(x[0]["weight_exact"],x[1]["weight_exact"]+1e-10)
        self.assertLessEqual(x[1]["weight_exact"],x[2]["weight_exact"]+1e-10)

    def test_negative_budget_rejected(self):
        with self.assertRaises(ValueError):
            single_frontier(seat=0,budgets=(-.1,))


if __name__=="__main__":
    unittest.main()
