"""Integration and mathematical safety tests for QT V4 (real Leduc)."""
from __future__ import annotations

import math
import random
import unittest
from pathlib import Path
from unittest.mock import patch

from pokerlab.bayes import BayesianOpponentModel, example_models
from pokerlab.cfr import (PolicyAgent, TabularPolicy, best_response,
                          best_response_mixture, profile_value)
from pokerlab.qt_v4 import (DecisionAgent, SwitchingBeliefs, run_benchmark,
                            safe_weight)
from pokerlab.simulate import run_hand


class QTResearchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.baseline = TabularPolicy.load("strategy.json")
        cls.models = example_models(cls.baseline)

    def test_zero_hazard_agrees_with_existing_bayesian_model(self):
        original = BayesianOpponentModel(self.models)
        switching = SwitchingBeliefs(self.models, hazard=0)
        for seed in (12, 17, 19, 23):
            terminal = run_hand(seed, (PolicyAgent(self.baseline),
                                       PolicyAgent(self.models["calling"])))
            obs = terminal.observe(0)
            a = original.update(obs)
            b = switching.update(obs)
            for x, y in zip(a, b):
                self.assertAlmostEqual(x, y, places=10)

    def test_switching_forgets_some_evidence(self):
        stationary = SwitchingBeliefs(self.models, hazard=0)
        changing = SwitchingBeliefs(self.models, hazard=0.2)
        for seed in range(30, 37):
            terminal = run_hand(seed, (PolicyAgent(self.baseline),
                                       PolicyAgent(self.models["calling"])))
            obs = terminal.observe(0)
            stationary.update(obs)
            changing.update(obs)
        self.assertTrue(all(x >= .05 - 1e-12 for x in changing.posterior))
        self.assertAlmostEqual(sum(changing.posterior), 1)
        self.assertTrue(math.isfinite(changing.last_log_score))

    def test_zero_budget_is_safe_on_exact_game(self):
        for seat in (0, 1):
            exploit = best_response_mixture(
                tuple(self.models.values()), (0.25,) * 4, seat)[1]
            cert = safe_weight(self.baseline, exploit, seat, 0.0)
            self.assertGreaterEqual(cert.certified_lower_bound,
                                    cert.allowed_floor - 1e-9)
            self.assertTrue(0 <= cert.weight <= 1)

    def test_exact_worst_case_of_whole_hand_mixture_respects_budget(self):
        # Exact opponent best response to a hidden ex-ante policy draw.
        # This is stronger than checking against the four candidate types.
        seat = 0
        exploit = best_response_mixture(
            tuple(self.models.values()), (0.25,) * 4, seat)[1]
        cert = safe_weight(self.baseline, exploit, seat, 0.1)
        opponent_gain, _ = best_response_mixture(
            (self.baseline, exploit), (1 - cert.weight, cert.weight),
            responder=1 - seat)
        exact_worst = -opponent_gain
        self.assertGreaterEqual(exact_worst, cert.allowed_floor - 1e-8)
        self.assertGreaterEqual(exact_worst, cert.certified_lower_bound - 1e-8)

    def test_agent_never_updates_before_hand_finishes(self):
        agent = DecisionAgent("switching", 0, self.baseline, self.models,
                              hazard=0.1)
        prior = agent.beliefs.posterior
        agent.prepare()
        self.assertEqual(prior, agent.beliefs.posterior)
        terminal = run_hand(5, (agent, PolicyAgent(self.models["calling"])))
        self.assertEqual(prior, agent.beliefs.posterior)
        agent.finish_hand(terminal.observe(0))
        self.assertEqual(agent.beliefs.hands_seen, 1)

    def test_reproducible_small_real_leduc_benchmark(self):
        kwargs = dict(scenario="switch", hands=2, replicates=2,
                      seed=741, hazard=0.1, budget=0.1,
                      include_oracle=False)
        a = run_benchmark(**kwargs)
        b = run_benchmark(**kwargs)
        self.assertEqual(a["mean_realized_chips_per_hand"],
                         b["mean_realized_chips_per_hand"])
        self.assertEqual(a["paired_realized"], b["paired_realized"])
        self.assertGreaterEqual(a["safe_certificate"]["min_margin"], -1e-9)
        self.assertEqual(len(a["rows"]), 2 * 2 * 2 * 5)


if __name__ == "__main__":
    unittest.main()
