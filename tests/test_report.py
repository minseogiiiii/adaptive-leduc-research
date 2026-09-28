import copy
import unittest

from pokerlab.bayes import ActionBiasPolicy
from pokerlab.cfr import UniformPolicy
from pokerlab.experiment import run_experiment
from pokerlab.game import Action
from pokerlab.report import validate_result


class ReportValidationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        baseline = UniformPolicy()
        before = ActionBiasPolicy(baseline, {Action.FOLD: 3.0})
        after = ActionBiasPolicy(baseline, {Action.BET: 3.0})
        cls.result = run_experiment(baseline, before,
                                    {"uniform": baseline, "folding": before},
                                    hands=2, replicates=2, seed=31,
                                    switch_to=after, switch_after=1)
        cls.result["design"].update({"opponent": "calling",
                                     "switch_to": "aggressive",
                                     "strategy_sha256": "strategy-test",
                                     "rules_sha256": "rules-test"})

    def validate(self, result):
        validate_result(result, strategy_sha256="strategy-test",
                        rules_sha256="rules-test", opponent="calling",
                        switch_to="aggressive", switch_after=1)

    def test_valid_switch_and_independent_hand_grid(self):
        self.validate(self.result)

    def test_rejects_missing_row_and_misassigned_phase(self):
        missing = copy.deepcopy(self.result)
        missing["hands"].pop()
        with self.assertRaisesRegex(ValueError, "Missing or duplicate"):
            self.validate(missing)
        phase = copy.deepcopy(self.result)
        phase["hands"][0]["phase"] = "post_switch"
        with self.assertRaisesRegex(ValueError, "phase disagrees"):
            self.validate(phase)

    def test_rejects_corrupt_reward_summary_and_interval(self):
        corrupted = copy.deepcopy(self.result)
        corrupted["hands"][0]["rewards"]["learning"] += 1
        with self.assertRaisesRegex(ValueError, "Inconsistent"):
            self.validate(corrupted)
        interval = copy.deepcopy(self.result)
        interval["paired_contrasts"]["learning_minus_cfr"][
            "bootstrap_95_percent"][0] -= 1
        with self.assertRaisesRegex(ValueError, "interval lower"):
            self.validate(interval)

    def test_rejects_wrong_source_hash_and_repeated_seed(self):
        wrong_hash = copy.deepcopy(self.result)
        wrong_hash["design"]["rules_sha256"] = "other"
        with self.assertRaisesRegex(ValueError, "design rules_sha256"):
            self.validate(wrong_hash)
        duplicate_seed = copy.deepcopy(self.result)
        duplicate_seed["hands"][0]["seed"] = duplicate_seed["hands"][1]["seed"]
        with self.assertRaisesRegex(ValueError, "Hand seeds repeat"):
            self.validate(duplicate_seed)

    def test_rejects_posterior_reset_at_switch(self):
        reset = copy.deepcopy(self.result)
        row = next(row for row in reset["hands"]
                   if row["replicate"] == 0 and row["seat"] == 0
                   and row["hand"] == 1)
        self.assertNotEqual(row["posterior_before"],
                            {"uniform": 0.5, "folding": 0.5})
        row["posterior_before"] = {"uniform": 0.5, "folding": 0.5}
        with self.assertRaisesRegex(ValueError, "posterior continuity"):
            self.validate(reset)


if __name__ == "__main__":
    unittest.main()
