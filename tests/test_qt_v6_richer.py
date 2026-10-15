"""Richer exact frontier safety and comparison tests."""
import unittest

from pokerlab.qt_v6_frontier_richer import run


class RichFrontierTests(unittest.TestCase):
    def test_richer_contains_original_ray_and_exact_floor(self):
        r=run(seat=0,steps=2,budgets=(.1,),phis=(0,1))
        x=r["rows"][0]
        self.assertGreaterEqual(x["gain_over_one_dimensional"],-1e-8)
        self.assertGreaterEqual(x["best"]["worst_ev"],x["risk_floor"]-1e-8)


if __name__=="__main__":
    unittest.main()
