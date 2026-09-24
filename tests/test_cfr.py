import tempfile
import unittest
from pathlib import Path

from pokerlab import Action, Card, new_hand
from pokerlab.cfr import (CFRSolver, PolicyAgent, TabularPolicy, UniformPolicy,
                          best_response, deals, diagnostics, info_key, profile_value)
from pokerlab.simulate import run_hand


class CFRTest(unittest.TestCase):
    def test_deals_are_uniform_distinct_physical_cards(self):
        self.assertEqual(len(deals()), 120)
        self.assertEqual(len(set(deals())), 120)
        self.assertTrue(all(len({first, second, public}) == 3
                            for first, second, public in deals()))

    def test_infosets_exclude_unobserved_card_and_copy(self):
        a = new_hand(Card("J", 0), Card("Q", 0))
        b = new_hand(Card("J", 1), Card("K", 0))
        self.assertEqual(info_key(a.observe(0)), info_key(b.observe(0)))
        a = a.apply(Action.CHECK).apply(Action.CHECK).deal_public(Card("K", 1))
        b = b.apply(Action.CHECK).apply(Action.CHECK).deal_public(Card("K", 1))
        self.assertEqual(info_key(a.observe(0)), info_key(b.observe(0)))

    def test_exact_value_and_best_response_respect_information(self):
        uniform = UniformPolicy()
        class Passive:
            def probabilities(self, observation):
                return tuple(float(action in (Action.CHECK, Action.CALL))
                             for action in observation.legal_actions)

        self.assertAlmostEqual(profile_value(Passive(), Passive()), 0, places=12)
        br0, policy0 = best_response(uniform, 0)
        br1, policy1 = best_response(uniform, 1)
        self.assertAlmostEqual(profile_value(policy0, uniform), br0, places=11)
        self.assertAlmostEqual(-profile_value(uniform, policy1), br1, places=11)
        self.assertGreater(br0, 0)
        self.assertGreater(br1, 0)
        self.assertAlmostEqual(diagnostics(uniform, uniform).nash_conv,
                               br0 + br1, places=12)

    def test_average_strategy_converges_and_can_run_without_game_state(self):
        solver = CFRSolver()
        solver.train(5)
        early = diagnostics(solver.average_policy(), solver.average_policy())
        solver.train(95)
        strategy = solver.average_policy()
        late = diagnostics(strategy, strategy)
        self.assertEqual(solver.iterations, 100)
        self.assertLess(late.exploitability, early.exploitability)
        self.assertLess(late.exploitability, 0.3)
        self.assertAlmostEqual(late.exploitability, late.nash_conv / 2)

        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "strategy.json"
            strategy.save(path)
            loaded = TabularPolicy.load(path)
            self.assertEqual(loaded.table, strategy.table)
            self.assertAlmostEqual(profile_value(loaded, loaded), late.value_p0)
            hand = run_hand(7, (PolicyAgent(loaded), PolicyAgent(loaded)))
            self.assertAlmostEqual(sum(hand.rewards), 0)


if __name__ == "__main__":
    unittest.main()
