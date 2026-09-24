import math
import random
import unittest

from pokerlab import Action, Card, new_hand
from pokerlab.bayes import (ActionBiasPolicy, AdaptiveAgent,
                            BayesianOpponentModel, _hand_likelihood,
                            learning_match)
from pokerlab.cfr import (UniformPolicy, best_response, best_response_mixture,
                          profile_value)
from pokerlab.simulate import run_hand


class FoldByRank:
    def __init__(self, favored_rank=None):
        self.favored_rank = favored_rank

    def probabilities(self, observation):
        actions = observation.legal_actions
        if Action.FOLD not in actions:
            return (1 / len(actions),) * len(actions)
        fold = (0.8 if observation.own_card.rank == self.favored_rank
                else 0.2)
        return tuple(fold if action == Action.FOLD
                     else (1 - fold) / (len(actions) - 1)
                     for action in actions)


class CheckByRank:
    def __init__(self, favored_rank):
        self.favored_rank = favored_rank

    def probabilities(self, observation):
        actions = observation.legal_actions
        if Action.CHECK in actions:
            check = 0.8 if observation.own_card.rank == self.favored_rank else 0.2
            return tuple(check if action == Action.CHECK else 1 - check
                         for action in actions)
        return (1 / len(actions),) * len(actions)


class BayesianTest(unittest.TestCase):
    def test_fold_hides_card_and_marginalizes_five_candidates(self):
        models = {"rank_j": FoldByRank("J"), "constant": FoldByRank()}
        a = new_hand(Card("K", 0), Card("J", 0)).apply(Action.BET).apply(Action.FOLD)
        b = new_hand(Card("K", 0), Card("Q", 0)).apply(Action.BET).apply(Action.FOLD)
        self.assertEqual(a.observe(0), b.observe(0))
        self.assertAlmostEqual(_hand_likelihood(a.observe(0), models["rank_j"]), .44)
        self.assertAlmostEqual(_hand_likelihood(a.observe(0), models["constant"]), .20)
        first = BayesianOpponentModel(models).update(a.observe(0))
        second = BayesianOpponentModel(models).update(b.observe(0))
        self.assertEqual(first, second)
        self.assertAlmostEqual(first[0], .44 / (.44 + .20))

    def test_public_board_eliminates_an_incompatible_hidden_card(self):
        state = new_hand(Card("K", 0), Card("J", 0))
        state = state.apply(Action.CHECK).apply(Action.CHECK)
        state = state.deal_public(Card("J", 1))
        state = state.apply(Action.BET).apply(Action.FOLD)
        # The opponent's round-1 check has probability 1/2 in both models.
        # Given our K0 and public J1, the possible opponent cards are
        # J0, Q0, Q1, K1; the first model's mean fold rate is 0.35.
        a = _hand_likelihood(state.observe(0), FoldByRank("J"))
        b = _hand_likelihood(state.observe(0), FoldByRank())
        # Four compatible private cards have joint mass 4 * (1/5) * (1/4)
        # = 1/5 for this particular board, conditional on our own card.
        self.assertAlmostEqual(a, .5 * .35 / 5)
        self.assertAlmostEqual(b, .5 * .20 / 5)
        self.assertAlmostEqual(a / (a + b), .35 / (.35 + .20))

    def test_showdown_uses_revealed_private_card(self):
        state = new_hand(Card("K", 0), Card("J", 0))
        state = state.apply(Action.CHECK).apply(Action.CHECK)
        state = state.deal_public(Card("Q", 0))
        state = state.apply(Action.CHECK).apply(Action.CHECK)
        beliefs = BayesianOpponentModel({"J": CheckByRank("J"),
                                         "K": CheckByRank("K")})
        posterior = beliefs.update(state.observe(0))
        self.assertAlmostEqual(posterior[0], .8**2 / (.8**2 + .2**2))

    def test_exact_mixture_response_and_single_type_equivalence(self):
        uniform = UniformPolicy()
        folding = ActionBiasPolicy(uniform, {Action.FOLD: 3})
        for seat in (0, 1):
            single_value, _ = best_response(uniform, seat)
            mixture_value, _ = best_response_mixture((uniform,), (1.0,), seat)
            self.assertAlmostEqual(single_value, mixture_value, places=12)
            value, policy = best_response_mixture((uniform, folding), (.4, .6), seat)
            evaluated = sum(weight * (profile_value(policy, opponent) if seat == 0
                                      else -profile_value(opponent, policy))
                            for weight, opponent in ((.4, uniform), (.6, folding)))
            self.assertAlmostEqual(value, evaluated, places=10)
            oracle = sum(weight * best_response(opponent, seat)[0]
                         for weight, opponent in ((.4, uniform), (.6, folding)))
            self.assertLessEqual(value, oracle + 1e-10)

    def test_repeat_observations_update_log_posterior_and_match_adapter(self):
        policies = {"biased": FoldByRank("J"), "other": FoldByRank()}
        observations = new_hand(Card("K", 0), Card("Q", 0))
        observations = observations.apply(Action.BET).apply(Action.FOLD).observe(0)
        beliefs = BayesianOpponentModel(policies)
        for _ in range(300):
            beliefs.update(observations)
        self.assertEqual(beliefs.hands_seen, 300)
        self.assertTrue(all(math.isfinite(prob) for prob in beliefs.posterior))
        self.assertGreater(beliefs.posterior[0], .999999)

        uniform = UniformPolicy()
        learner = AdaptiveAgent(0, BayesianOpponentModel({"uniform": uniform}))
        results = learning_match(learner, uniform, hands=2, seed=42)
        self.assertEqual(len(results), 2)
        self.assertEqual(learner.beliefs.hands_seen, 2)
        self.assertEqual(results[-1].posterior, {"uniform": 1.0})
        self.assertAlmostEqual(sum(run_hand(42).rewards), 0)
        with self.assertRaises(ValueError):
            learner.finish_hand(new_hand(Card("K", 0), Card("J", 0)).observe(0))


if __name__ == "__main__":
    unittest.main()
