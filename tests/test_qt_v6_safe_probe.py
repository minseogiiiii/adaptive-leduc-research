"""Safe one-hand active probe validation."""
import unittest

from pokerlab.qt_v6_safe_probe import run


class SafeProbeTests(unittest.TestCase):
    def test_safe_lottery_one_hand_bound(self):
        result=run(seed=619,replicates=2,hands_per_type=3)
        w=result["safe_probability_of_probe"]
        self.assertTrue(0<=w<=1)
        for seat in ("0","1"):
            self.assertGreaterEqual(
                result["worst_expected_payoff_by_seat"][seat],
                result["required_floor_by_seat"][seat]-1e-8)
        self.assertTrue(0<=result["mixed_probe_information"]["accuracy"]<=1)


if __name__=="__main__":
    unittest.main()
