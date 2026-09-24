"""Bayesian learning over fixed opponent types in two-player Leduc v0.1.

Every posterior update uses a terminal *PlayerObservation* and integrates out
an opponent card that was never revealed. Every decision uses a precomputed
information-set best response to the current mixture of complete hand policies.
"""

from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from .cfr import (Policy, PolicyAgent, TabularPolicy, _validate,
                  best_response_mixture)
from .game import (Action, Card, Phase, PlayerObservation, all_cards,
                   new_hand)
from .simulate import run_hand


@dataclass(frozen=True)
class ActionBiasPolicy:
    """Full-support perturbation of a fixed base strategy.

    Multipliers are deliberately simple, declared assumptions about opponent
    behavior. Smoothing ensures that every legal action remains possible.
    """

    base: Policy
    factors: Mapping[Action, float]
    smoothing: float = 0.05

    def __post_init__(self) -> None:
        if not 0 < self.smoothing <= 1 or not math.isfinite(self.smoothing):
            raise ValueError("smoothing must be finite and in (0, 1]")
        if any(not math.isfinite(value) or value <= 0
               for value in self.factors.values()):
            raise ValueError("Action factors must be finite and positive")

    def probabilities(self, observation: PlayerObservation) -> tuple[float, ...]:
        actions = observation.legal_actions
        original = self.base.probabilities(observation)
        _validate(original, actions)
        unnormalized = [((1 - self.smoothing) * prob
                         + self.smoothing / len(actions))
                        * self.factors.get(action, 1.0)
                        for action, prob in zip(actions, original)]
        total = sum(unnormalized)
        return tuple(weight / total for weight in unnormalized)


def example_models(baseline: Policy) -> dict[str, Policy]:
    """Prespecified candidate policies; none is inferred from evaluation data."""
    return {
        "baseline": ActionBiasPolicy(baseline, {}),
        "folding": ActionBiasPolicy(
            baseline, {Action.FOLD: 3.0, Action.CALL: 0.7,
                       Action.BET: 0.8, Action.RAISE: 0.7}),
        "calling": ActionBiasPolicy(
            baseline, {Action.FOLD: 0.5, Action.CALL: 2.5,
                       Action.BET: 0.8, Action.RAISE: 0.8}),
        "aggressive": ActionBiasPolicy(
            baseline, {Action.FOLD: 0.7, Action.CALL: 0.7,
                       Action.BET: 2.5, Action.RAISE: 3.0}),
    }


def _hand_likelihood(observation: PlayerObservation, model: Policy) -> float:
    """P(public actions, board, optional revealed card | our card, model).

    Our own action probabilities and the deal of our card cancel between
    opponent types. The candidate opponent cards have equal prior probability
    1/5; a revealed board has conditional probability 1/4 for each compatible
    card. At a fold the other private card remains unobserved and is summed.
    """
    if observation.phase != Phase.TERMINAL or observation.player not in (0, 1):
        raise ValueError("A terminal observation from the learner is required")
    own = observation.own_card
    board = observation.public_card
    revealed = observation.opponent_card_at_showdown
    if revealed is not None and (revealed == own or revealed == board):
        raise ValueError("A revealed card must be distinct from known cards")
    if revealed is not None and board is None:
        raise ValueError("Showdown requires a public card")
    candidates = [revealed] if revealed is not None else [
        card for card in all_cards() if card != own and card != board]
    likelihood = 0.0
    for opponent_card in candidates:
        if opponent_card == own or opponent_card == board:
            continue
        cards = ((own, opponent_card) if observation.player == 0
                 else (opponent_card, own))
        state = new_hand(*cards)
        path_likelihood = 1.0
        for event in observation.actions:
            if state.phase == Phase.DEAL_PUBLIC:
                if board is None:
                    raise ValueError("Second-round actions require a public card")
                state = state.deal_public(board)
            if state.phase != Phase.BETTING or state.actor != event.player:
                raise ValueError("Public action history is inconsistent with the game")
            if state.round_index != event.round_index or event.action not in state.legal_actions():
                raise ValueError("Invalid round or action in public history")
            if event.player != observation.player:
                opp_observation = state.observe(event.player)
                probs = model.probabilities(opp_observation)
                _validate(probs, opp_observation.legal_actions)
                path_likelihood *= probs[opp_observation.legal_actions.index(event.action)]
            state = state.apply(event.action)
        if (state.phase != Phase.TERMINAL or state.public_card != board
                or state.contributions != observation.contributions
                or state.showdown != (revealed is not None)):
            raise ValueError("Terminal observation is inconsistent with replay")
        likelihood += path_likelihood / (5 * (4 if board is not None else 1))
    return likelihood


class BayesianOpponentModel:
    """Posterior over opponent policies, updated after each completed hand."""

    def __init__(self, models: Mapping[str, Policy],
                 prior: tuple[float, ...] | None = None) -> None:
        if not models:
            raise ValueError("At least one opponent model is required")
        self.names = tuple(models)
        self.policies = tuple(models.values())
        prior = prior if prior is not None else (1 / len(models),) * len(models)
        if (len(prior) != len(models) or any(
                not math.isfinite(weight) or weight <= 0 for weight in prior)
                or abs(sum(prior) - 1) > 1e-10):
            raise ValueError("Prior must be positive, finite, and sum to one")
        self._log_weights = [math.log(weight) for weight in prior]
        self.hands_seen = 0

    @property
    def posterior(self) -> tuple[float, ...]:
        largest = max(self._log_weights)
        weights = [math.exp(log_weight - largest)
                   for log_weight in self._log_weights]
        total = sum(weights)
        return tuple(weight / total for weight in weights)

    def update(self, terminal: PlayerObservation) -> tuple[float, ...]:
        likelihoods = [_hand_likelihood(terminal, model)
                       for model in self.policies]
        updated = [weight + math.log(likelihood) if likelihood > 0 else -math.inf
                   for weight, likelihood in zip(self._log_weights, likelihoods)]
        if all(weight == -math.inf for weight in updated):
            raise ValueError("Observed hand is impossible under every candidate model")
        self._log_weights = updated
        self.hands_seen += 1
        return self.posterior

    def named_posterior(self) -> dict[str, float]:
        return dict(zip(self.names, self.posterior))


class AdaptiveAgent:
    """Exact per-hand mixture BR, followed by a terminal-observation update.

    Call finish_hand(state.observe(player)) after each run_hand. The state is
    held by the match runner, never passed into this agent. A new BR policy is
    computed lazily on the first action of the next hand.
    """

    def __init__(self, player: int, beliefs: BayesianOpponentModel) -> None:
        if player not in (0, 1):
            raise ValueError("player must be 0 or 1")
        self.player = player
        self.beliefs = beliefs
        self._policy: TabularPolicy | None = None

    def choose(self, observation: PlayerObservation, rng: random.Random) -> Action:
        if observation.player != self.player:
            raise ValueError("Agent received another player's observation")
        if self._policy is None:
            _, self._policy = best_response_mixture(
                self.beliefs.policies, self.beliefs.posterior, self.player)
        return PolicyAgent(self._policy).choose(observation, rng)

    def finish_hand(self, observation: PlayerObservation) -> tuple[float, ...]:
        if observation.player != self.player:
            raise ValueError("Agent received another player's observation")
        posterior = self.beliefs.update(observation)
        self._policy = None
        return posterior


@dataclass(frozen=True)
class HandResult:
    reward: float
    posterior: dict[str, float]


def learning_match(agent: AdaptiveAgent, opponent: Policy, hands: int,
                   seed: int = 0) -> list[HandResult]:
    """Seeded sequential hands against one fixed opponent policy."""
    if hands < 1:
        raise ValueError("hands must be positive")
    players = ((agent, PolicyAgent(opponent)) if agent.player == 0
               else (PolicyAgent(opponent), agent))
    results = []
    for hand_index in range(hands):
        terminal = run_hand(seed + hand_index, players)
        agent.finish_hand(terminal.observe(agent.player))
        assert terminal.rewards is not None
        results.append(HandResult(terminal.rewards[agent.player],
                                  agent.beliefs.named_posterior()))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Bayesian Leduc learning demo")
    parser.add_argument("--strategy", type=Path, default=Path("strategy.json"))
    parser.add_argument("--opponent", choices=("baseline", "folding", "calling",
                                                "aggressive"), default="calling")
    parser.add_argument("--hands", type=int, default=12)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    baseline = TabularPolicy.load(args.strategy)
    models = example_models(baseline)
    learner = AdaptiveAgent(0, BayesianOpponentModel(models))
    outcomes = learning_match(learner, models[args.opponent], args.hands, args.seed)
    for index, result in enumerate(outcomes, 1):
        print(f"hand={index:>3} reward={result.reward:+.1f} "
              + " ".join(f"{name}={prob:.3f}"
                         for name, prob in result.posterior.items()))
    print(f"sample_mean_reward={sum(result.reward for result in outcomes) / args.hands:.4f}")
    print("This sample mean is a trace, not an experimental performance claim.")


if __name__ == "__main__":
    main()
