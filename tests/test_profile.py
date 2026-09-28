import unittest

from pokerlab.bayes import ActionBiasPolicy
from pokerlab.cfr import UniformPolicy
from pokerlab.experiment import run_experiment
from pokerlab.game import Action
from pokerlab.profile import profile_experiment, result_digest


class ProfileTest(unittest.TestCase):
    def test_profile_preserves_switch_experiment_result(self):
        baseline = UniformPolicy()
        before = ActionBiasPolicy(baseline, {Action.FOLD: 3.0})
        after = ActionBiasPolicy(baseline, {Action.BET: 3.0})
        models = {"uniform": baseline, "folding": before}
        inputs = {"hands": 2, "replicates": 2, "seed": 23,
                  "switch_to": after, "switch_after": 1}
        expected = run_experiment(baseline, before, models, **inputs)
        profiled = profile_experiment(baseline, before, models, top=8, **inputs)
        self.assertEqual(profiled["result_sha256"], result_digest(expected))
        self.assertEqual(profiled["mean_reward"], expected["mean_reward"])
        self.assertEqual(profiled["paired_contrasts"],
                         expected["paired_contrasts"])
        self.assertEqual(profiled["workload"], expected["design"])
        self.assertEqual(len(profiled["profile"]["top_by_cumulative_seconds"]), 8)
        self.assertGreater(profiled["profile"]["total_function_calls"], 0)
        self.assertGreater(profiled["profile"]["elapsed_seconds"], 0)
        rows = profiled["profile"]["top_by_cumulative_seconds"]
        self.assertEqual(rows, sorted(rows, key=lambda row: (
            -row["cumulative_seconds"], -row["self_seconds"],
            row["file"], row["line"], row["function"])))

    def test_rejects_nonpositive_profile_size(self):
        baseline = UniformPolicy()
        with self.assertRaisesRegex(ValueError, "top must be positive"):
            profile_experiment(baseline, baseline, {"uniform": baseline},
                               hands=1, replicates=2, seed=0, top=0)


if __name__ == "__main__":
    unittest.main()
