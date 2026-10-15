"""Tests for Leduc value-of-information engine."""
import unittest

from pokerlab.bayes import example_models
from pokerlab.cfr import TabularPolicy
from pokerlab.qt_information import value_of_signal


class SignalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        models = example_models(TabularPolicy.load("strategy.json"))
        cls.opponents = (models["calling"], models["aggressive"])

    def test_uninformative_signal_is_worth_zero(self):
        r = value_of_signal(
            opponents=self.opponents, prior=(.5, .5), seat=0,
            signal_likelihoods={"a": (.5, .5), "b": (.5, .5)})
        self.assertAlmostEqual(r.evsi, 0, places=8)
        self.assertGreaterEqual(r.evpi, -1e-9)

    def test_perfect_signal_equals_evpi(self):
        r = value_of_signal(
            opponents=self.opponents, prior=(.5, .5), seat=1,
            signal_likelihoods={"a": (1, 0), "b": (0, 1)})
        self.assertAlmostEqual(r.evsi, r.evpi, places=8)

    def test_noisy_signal_between_zero_and_perfect_information(self):
        r = value_of_signal(
            opponents=self.opponents, prior=(.5, .5), seat=0,
            signal_likelihoods={"a": (.8, .2), "b": (.2, .8)})
        self.assertGreaterEqual(r.evsi, -1e-9)
        self.assertLessEqual(r.evsi, r.evpi + 1e-9)

    def test_rejects_invalid_signal(self):
        with self.assertRaises(ValueError):
            value_of_signal(
                opponents=self.opponents, prior=(.5, .5), seat=0,
                signal_likelihoods={"a": (.8, .8), "b": (.1, .1)})


if __name__ == "__main__":
    unittest.main()
