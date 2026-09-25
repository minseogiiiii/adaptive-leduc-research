import unittest

from pokerlab import Action
from pokerlab.bayes import ActionBiasPolicy
from pokerlab.cfr import UniformPolicy
from pokerlab.validate import validate_profile


class BetFirst:
    def probabilities(self, observation):
        actions = observation.legal_actions
        chosen = (Action.BET if Action.BET in actions else
                  Action.RAISE if Action.RAISE in actions else
                  Action.CALL if Action.CALL in actions else Action.CHECK)
        return tuple(float(action == chosen) for action in actions)


class FoldToBet:
    def probabilities(self, observation):
        actions = observation.legal_actions
        chosen = Action.FOLD if Action.FOLD in actions else Action.CHECK
        return tuple(float(action == chosen) for action in actions)


class ValidationTest(unittest.TestCase):
    def test_known_one_chip_fold_payoff_has_zero_sampling_error(self):
        result = validate_profile(BetFirst(), FoldToBet(), hands=20, seed=13)
        self.assertEqual(result["exact_value_p0"], 1)
        self.assertEqual(result["sample_mean_p0"], 1)
        self.assertEqual(result["sample_standard_error"], 0)
        self.assertTrue(result["within_four_standard_errors"])

    def test_asymmetric_stochastic_profile_matches_simulated_hands(self):
        uniform = UniformPolicy()
        folding = ActionBiasPolicy(uniform, {Action.FOLD: 3.0})
        result = validate_profile(uniform, folding, hands=4000, seed=37)
        self.assertTrue(result["within_four_standard_errors"], result)
        self.assertGreater(result["sample_standard_error"], 0)

    def test_requires_multiple_hands_for_sampling_error(self):
        with self.assertRaisesRegex(ValueError, "at least two"):
            validate_profile(UniformPolicy(), UniformPolicy(), hands=1, seed=0)


if __name__ == "__main__":
    unittest.main()
