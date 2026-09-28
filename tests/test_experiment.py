import unittest
from statistics import fmean
from unittest.mock import patch

from pokerlab.bayes import ActionBiasPolicy
from pokerlab.cfr import UniformPolicy
from pokerlab.experiment import (PressureReactivePolicy, RankSelectivePolicy,
                                 RoundPolarizedPolicy, run_experiment)
from pokerlab.game import Action, Card, new_hand
from pokerlab.simulate import run_hand


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

    def test_held_out_opponent_reacts_to_private_rank(self):
        model = RankSelectivePolicy(UniformPolicy())
        strong = new_hand(Card("K", 0), Card("J", 0))
        weak = new_hand(Card("J", 0), Card("K", 0))
        first = model.probabilities(strong.observe(0))
        second = model.probabilities(weak.observe(0))
        self.assertGreater(first[strong.legal_actions().index(Action.BET)],
                           second[weak.legal_actions().index(Action.BET)])
        strong = strong.apply(Action.BET)
        weak = weak.apply(Action.BET)
        first = model.probabilities(strong.observe(1))
        second = model.probabilities(weak.observe(1))
        self.assertGreater(first[strong.legal_actions().index(Action.FOLD)],
                           second[weak.legal_actions().index(Action.FOLD)])
        self.assertTrue(all(p > 0 for p in first + second))

    def test_round_polarized_opponent_changes_with_betting_round(self):
        model = RoundPolarizedPolicy(UniformPolicy())
        state = new_hand(Card("J", 0), Card("Q", 0))
        pre = model.probabilities(state.observe(0))
        state = state.apply(Action.CHECK).apply(Action.CHECK)
        state = state.deal_public(Card("K", 0))
        post = model.probabilities(state.observe(0))
        self.assertGreater(post[state.legal_actions().index(Action.BET)],
                           pre[new_hand(Card("J", 0), Card("Q", 0)).legal_actions().index(Action.BET)])
        self.assertAlmostEqual(sum(pre), 1)
        self.assertAlmostEqual(sum(post), 1)
        self.assertTrue(all(p > 0 for p in pre + post))

    def test_pressure_reactive_opponent_uses_current_public_history(self):
        model = PressureReactivePolicy(UniformPolicy())
        state = new_hand(Card("J", 0), Card("Q", 0))
        unpressured = model.probabilities(state.observe(0))
        state = state.apply(Action.BET)
        pressured = model.probabilities(state.observe(1))
        self.assertGreater(pressured[state.legal_actions().index(Action.FOLD)],
                           pressured[state.legal_actions().index(Action.RAISE)])
        self.assertGreater(unpressured[1], unpressured[0])
        self.assertAlmostEqual(sum(pressured), 1)
        self.assertTrue(all(p > 0 for p in pressured))

    def test_switch_uses_correct_policy_without_resetting_beliefs(self):
        baseline = UniformPolicy()
        before = ActionBiasPolicy(baseline, {Action.FOLD: 3.0})
        after = ActionBiasPolicy(baseline, {Action.BET: 3.0, Action.RAISE: 3.0})
        models = {"uniform": baseline, "folding": before}
        used_policies = []

        # Each row contains three arms. Capture their opponent at each seat
        # without changing the game runner or its chance/action seeds.
        def record_hand(seed, players):
            opponent_seat = 1 if (len(used_policies) // (4 * 3)) % 2 == 0 else 0
            used_policies.append(players[opponent_seat].policy)
            return run_hand(seed, players)

        with patch("pokerlab.experiment.run_hand", side_effect=record_hand):
            result = run_experiment(baseline, before, models, hands=4,
                                    replicates=2, seed=29, switch_to=after,
                                    switch_after=2)
        self.assertEqual(len(result["hands"]), 16)
        self.assertEqual(len(used_policies), 48)
        for index, row in enumerate(result["hands"]):
            expected = before if row["hand"] < 2 else after
            self.assertTrue(all(policy is expected
                                for policy in used_policies[index * 3:index * 3 + 3]))
            self.assertEqual(row["phase"], "pre_switch" if row["hand"] < 2
                             else "post_switch")
            if row["hand"] == 0:
                self.assertEqual(row["posterior_before"],
                                 {"uniform": .5, "folding": .5})
                self.assertEqual(row["rewards"]["learning"],
                                 row["rewards"]["frozen_prior"])
            else:
                self.assertEqual(row["posterior_before"],
                                 result["hands"][index - 1]["posterior"])
        self.assertTrue(any(row["posterior_before"]["folding"] != .5
                            for row in result["hands"] if row["hand"] == 2))
        for phase in ("pre_switch", "post_switch"):
            summary = result["phases"][phase]
            self.assertEqual(summary["hands_per_seat"], 2)
            for arm in ("learning", "frozen_prior", "cfr"):
                observed = [row["rewards"][arm] for row in result["hands"]
                            if row["phase"] == phase]
                self.assertAlmostEqual(summary["mean_reward"][arm], fmean(observed))
        for control in ("frozen_prior", "cfr"):
            key = f"learning_minus_{control}"
            pre = result["phases"]["pre_switch"]["paired_contrasts"][key]["mean"]
            post = result["phases"]["post_switch"]["paired_contrasts"][key]["mean"]
            self.assertAlmostEqual(result["change_in_paired_contrasts"][key]["mean"],
                                   post - pre)

    def test_switch_requires_both_arguments_and_two_nonempty_phases(self):
        baseline = UniformPolicy()
        params = dict(hands=3, replicates=2, seed=0)
        for extras in ({"switch_after": 1}, {"switch_to": baseline}):
            with self.subTest(extras=extras):
                with self.assertRaisesRegex(ValueError, "specified together"):
                    run_experiment(baseline, baseline, {"uniform": baseline},
                                   **params, **extras)
        for boundary in (0, 3, -1):
            with self.subTest(boundary=boundary):
                with self.assertRaisesRegex(ValueError, "between 1 and hands - 1"):
                    run_experiment(baseline, baseline, {"uniform": baseline},
                                   **params, switch_to=baseline,
                                   switch_after=boundary)


if __name__ == "__main__":
    unittest.main()
