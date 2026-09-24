import unittest

from pokerlab import Action, Card, Phase, RuleError, all_cards, new_hand
from pokerlab.simulate import format_trace, run_hand


class LeducRulesTest(unittest.TestCase):
    def test_two_documented_hands(self):
        state = new_hand(Card("K", 1), Card("Q", 0))
        state = state.apply(Action.CHECK).apply(Action.CHECK)
        self.assertEqual(state.phase, Phase.DEAL_PUBLIC)
        self.assertIsNone(state.observe(0).public_card)
        state = state.deal_public(Card("K", 0))
        self.assertEqual(state.actor, 0)
        state = state.apply(Action.CHECK).apply(Action.CHECK)
        self.assertEqual(state.rewards, (1.0, -1.0))
        self.assertEqual(state.observe(0).opponent_card_at_showdown, Card("Q", 0))

        state = new_hand(Card("J", 0), Card("K", 0)).apply(Action.BET)
        self.assertEqual(state.contributions, (3, 1))
        state = state.apply(Action.FOLD)
        self.assertEqual(state.rewards, (1.0, -1.0))
        self.assertIsNone(state.public_card)
        self.assertIsNone(state.observe(0).opponent_card_at_showdown)

    def test_raise_cap_and_round_transition(self):
        state = new_hand(Card("Q", 0), Card("J", 0)).apply(Action.BET)
        self.assertEqual(state.legal_actions(), (Action.FOLD, Action.CALL, Action.RAISE))
        state = state.apply(Action.RAISE)
        self.assertEqual(state.contributions, (3, 5))
        self.assertEqual(state.legal_actions(), (Action.FOLD, Action.CALL))
        with self.assertRaises(RuleError):
            state.apply(Action.RAISE)
        state = state.apply(Action.CALL)
        self.assertEqual(state.contributions, (5, 5))
        self.assertEqual(state.phase, Phase.DEAL_PUBLIC)
        with self.assertRaises(RuleError):
            state.apply(Action.CHECK)

    def test_card_invariants_and_hidden_observation(self):
        hand = new_hand(Card("J", 0), Card("Q", 0))
        other = new_hand(Card("J", 0), Card("K", 0))
        self.assertEqual(hand.observe(0), other.observe(0))
        with self.assertRaises(RuleError):
            new_hand(Card("J", 0), Card("J", 0))
        with self.assertRaises(RuleError):
            hand.deal_public(Card("K", 0))
        hand = hand.apply(Action.CHECK).apply(Action.CHECK)
        with self.assertRaises(RuleError):
            hand.deal_public(Card("J", 0))
        hand = hand.deal_public(Card("K", 1))
        self.assertEqual(hand.observe(0).public_card, Card("K", 1))
        self.assertIsNone(hand.observe(0).opponent_card_at_showdown)
        self.assertEqual(hand.observe(1).legal_actions, ())

    def test_private_card_remains_hidden_after_public_actions(self):
        left = new_hand(Card("J", 0), Card("Q", 0))
        right = new_hand(Card("J", 0), Card("K", 0))
        for action in (Action.CHECK, Action.CHECK):
            left, right = left.apply(action), right.apply(action)
        left, right = left.deal_public(Card("J", 1)), right.deal_public(Card("J", 1))
        left, right = left.apply(Action.BET), right.apply(Action.BET)
        self.assertEqual(left.observe(0), right.observe(0))
        left, right = left.apply(Action.FOLD), right.apply(Action.FOLD)
        self.assertEqual(left.observe(0), right.observe(0))
        self.assertIsNone(left.observe(0).opponent_card_at_showdown)

    def test_actions_do_not_change_the_card_draw(self):
        class Passive:
            def choose(self, observation, rng):
                return (Action.CHECK if Action.CHECK in observation.legal_actions
                        else Action.CALL)

        class Betting:
            def choose(self, observation, rng):
                return (Action.BET if Action.BET in observation.legal_actions
                        else Action.CALL if Action.CALL in observation.legal_actions
                        else Action.CHECK)

        passive = run_hand(17, (Passive(), Passive()))
        aggressive = run_hand(17, (Betting(), Betting()))
        self.assertEqual(passive.private_cards, aggressive.private_cards)
        self.assertEqual(passive.public_card, aggressive.public_card)

    def test_ties_and_reproducibility(self):
        hand = new_hand(Card("Q", 0), Card("Q", 1))
        hand = hand.apply(Action.CHECK).apply(Action.CHECK)
        hand = hand.deal_public(Card("K", 0))
        hand = hand.apply(Action.CHECK).apply(Action.CHECK)
        self.assertEqual(hand.rewards, (0.0, 0.0))
        self.assertEqual(format_trace(run_hand(42)), format_trace(run_hand(42)))

    def test_all_reachable_terminals_and_observation_boundary(self):
        terminals = 0

        def walk(state):
            nonlocal terminals
            if state.phase == Phase.TERMINAL:
                terminals += 1
                self.assertAlmostEqual(sum(state.rewards), 0)
                self.assertGreaterEqual(sum(state.contributions), 2)
                self.assertEqual(
                    state.observe(0).opponent_card_at_showdown is not None,
                    state.showdown,
                )
                return
            if state.phase == Phase.DEAL_PUBLIC:
                self.assertEqual(state.legal_actions(), ())
                self.assertIsNone(state.observe(0).public_card)
                for card in all_cards():
                    if card not in state.private_cards:
                        walk(state.deal_public(card))
            else:
                self.assertTrue(state.legal_actions())
                self.assertEqual(state.observe(state.actor).legal_actions,
                                 state.legal_actions())
                for action in state.legal_actions():
                    walk(state.apply(action))

        for first in all_cards():
            for second in all_cards():
                if first != second:
                    walk(new_hand(first, second))
        self.assertGreater(terminals, 1_000)


if __name__ == "__main__":
    unittest.main()
