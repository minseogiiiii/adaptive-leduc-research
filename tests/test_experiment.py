import unittest

from pokerlab.bayes import ActionBiasPolicy
from pokerlab.cfr import UniformPolicy
from pokerlab.experiment import run_experiment
from pokerlab.game import Action


class ExperimentTest(unittest.TestCase):
    def test_single_model_is_identical_to_frozen_control_on_shared_deals(self):
        baseline = UniformPolicy()
        opponent = ActionBiasPolicy(baseline, {Action.FOLD: 3.0})
        result = run_experiment(baseline, opponent, {"uniform": baseline},
                                hands=2, replicates=2, seed=19)
        self.assertEqual(len(result["hands"]), 8)
        self.assertEqual({row["seat"] for row in result["hands"]}, {0, 1})
        self.assertEqual({row["replicate"] for row in result["hands"]}, {0, 1})
        self.assertEqual(len({row["seed"] for row in result["hands"]}), 8)
        for row in result["hands"]:
            # Exact same policy and action/chance seed: any difference would
            # expose a mismatch in the paired match runner or prior control.
            self.assertEqual(row["rewards"]["learning"],
                             row["rewards"]["frozen_prior"])
            self.assertEqual(row["posterior"], {"uniform": 1.0})
        self.assertEqual(result["paired_contrasts"]["learning_minus_frozen_prior"],
                         {"mean": 0.0, "bootstrap_95_percent": [0.0, 0.0]})

    def test_rejects_insufficient_independent_replicates(self):
        baseline = UniformPolicy()
        with self.assertRaisesRegex(ValueError, "replicates at least two"):
            run_experiment(baseline, baseline, {"uniform": baseline},
                           hands=5, replicates=1, seed=0)
        with self.assertRaisesRegex(ValueError, "hands must be positive"):
            run_experiment(baseline, baseline, {"uniform": baseline},
                           hands=0, replicates=2, seed=0)

    def test_multiple_models_share_initial_decision_and_update_after_hand(self):
        baseline = UniformPolicy()
        opponent = ActionBiasPolicy(baseline, {Action.FOLD: 3.0})
        models = {"uniform": baseline, "folding": opponent}
        result = run_experiment(baseline, opponent, models,
                                hands=2, replicates=2, seed=19)
        for row in result["hands"]:
            if row["hand"] == 0:
                self.assertEqual(row["rewards"]["learning"],
                                 row["rewards"]["frozen_prior"])
        self.assertTrue(any(abs(row["posterior"]["folding"] - .5) > 1e-8
                            for row in result["hands"] if row["hand"] == 0))


if __name__ == "__main__":
    unittest.main()
