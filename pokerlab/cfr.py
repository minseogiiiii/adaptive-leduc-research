"""Full-tree vanilla CFR and exact evaluation for the frozen Leduc variant.

Only *rank*, public actions and the public rank enter an information set.
Physical card copies and unrevealed cards are evaluator-only information.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .game import Action, Card, GameState, Phase, PlayerObservation, all_cards, new_hand

InfoKey = tuple[int, str, str | None, tuple[tuple[int, int, str], ...]]
World = tuple[GameState, Card, float]  # state, predetermined public card, reach mass
ModelWorld = tuple[GameState, Card, float, int]  # plus fixed opponent type


def info_key(observation: PlayerObservation) -> InfoKey:
    """Information available at a decision, never evaluator-only cards."""
    if observation.phase != Phase.BETTING or observation.actor != observation.player:
        raise ValueError("An information key requires the acting player's observation")
    return (observation.player, observation.own_card.rank,
            observation.public_card.rank if observation.public_card else None,
            tuple((event.round_index, event.player, event.action.value)
                  for event in observation.actions))


def deals() -> tuple[tuple[Card, Card, Card], ...]:
    """All 6*5*4 equiprobable ordered private/public physical-card draws."""
    cards = all_cards()
    return tuple((first, second, public)
                 for first in cards for second in cards for public in cards
                 if first != second and public not in (first, second))


class Policy(Protocol):
    def probabilities(self, observation: PlayerObservation) -> tuple[float, ...]: ...


def _validate(probs: tuple[float, ...], actions: tuple[Action, ...]) -> None:
    if len(probs) != len(actions) or any(p < 0 or p > 1 for p in probs):
        raise ValueError("Invalid action probabilities")
    if abs(sum(probs) - 1) > 1e-10:
        raise ValueError("Action probabilities must sum to one")


@dataclass(frozen=True)
class TabularPolicy:
    """Complete tabular behavioral policy, indexed solely by observations."""

    table: dict[InfoKey, dict[Action, float]]

    def probabilities(self, observation: PlayerObservation) -> tuple[float, ...]:
        row = self.table[info_key(observation)]  # fail loudly on missing information sets
        probs = tuple(row[action] for action in observation.legal_actions)
        _validate(probs, observation.legal_actions)
        return probs

    def save(self, path: str | Path) -> None:
        entries = [{"player": key[0], "own_rank": key[1],
                    "public_rank": key[2], "history": [list(e) for e in key[3]],
                    "actions": {action.value: prob for action, prob in row.items()}}
                   for key, row in sorted(self.table.items(), key=lambda pair: str(pair[0]))]
        Path(path).write_text(json.dumps({"format": "leduc-v0.1-policy",
                                          "entries": entries}, indent=2) + "\n")

    @classmethod
    def load(cls, path: str | Path) -> TabularPolicy:
        data = json.loads(Path(path).read_text())
        if data.get("format") != "leduc-v0.1-policy":
            raise ValueError("Policy was not saved for the frozen Leduc v0.1 rules")
        table: dict[InfoKey, dict[Action, float]] = {}
        for entry in data["entries"]:
            key: InfoKey = (entry["player"], entry["own_rank"], entry["public_rank"],
                            tuple((int(r), int(p), str(a)) for r, p, a in entry["history"]))
            if key in table:
                raise ValueError("Duplicate information set in policy")
            table[key] = {Action(action): float(prob)
                          for action, prob in entry["actions"].items()}
        return cls(table)


@dataclass(frozen=True)
class UniformPolicy:
    def probabilities(self, observation: PlayerObservation) -> tuple[float, ...]:
        return (1 / len(observation.legal_actions),) * len(observation.legal_actions)


@dataclass(frozen=True)
class PolicyAgent:
    """Adapter for run_hand; receives only PlayerObservation, never GameState."""

    policy: Policy

    def choose(self, observation: PlayerObservation, rng: random.Random) -> Action:
        actions = observation.legal_actions
        probabilities = self.policy.probabilities(observation)
        _validate(probabilities, actions)
        draw = rng.random()
        cumulative = 0.0
        for action, probability in zip(actions, probabilities):
            cumulative += probability
            if draw < cumulative:
                return action
        return actions[-1]  # floating-point rounding at the end of the interval


def _regret_matching(regrets: list[float]) -> tuple[float, ...]:
    positives = [max(0.0, regret) for regret in regrets]
    total = sum(positives)
    if total == 0:
        return (1 / len(regrets),) * len(regrets)
    return tuple(value / total for value in positives)


class CFRSolver:
    """Simultaneous vanilla CFR: freeze regrets during every full-tree iteration."""

    def __init__(self) -> None:
        self.regrets: dict[InfoKey, list[float]] = {}
        self.strategy_sums: dict[InfoKey, list[float]] = {}
        self.actions: dict[InfoKey, tuple[Action, ...]] = {}
        self.iterations = 0

    def train(self, iterations: int) -> None:
        if iterations < 1:
            raise ValueError("iterations must be positive")
        all_deals = deals()
        chance = 1 / len(all_deals)
        for _ in range(iterations):
            regret_delta: dict[InfoKey, list[float]] = {}
            strategy_delta: dict[InfoKey, list[float]] = {}

            def walk(state: GameState, public: Card, reach0: float,
                     reach1: float) -> float:
                if state.phase == Phase.TERMINAL:
                    assert state.rewards is not None
                    return state.rewards[0]
                if state.phase == Phase.DEAL_PUBLIC:
                    return walk(state.deal_public(public), public, reach0, reach1)
                actor = state.actor
                assert actor is not None
                observation = state.observe(actor)
                key = info_key(observation)
                legal = observation.legal_actions
                if key not in self.regrets:
                    self.regrets[key] = [0.0] * len(legal)
                    self.strategy_sums[key] = [0.0] * len(legal)
                    self.actions[key] = legal
                if self.actions[key] != legal:
                    raise AssertionError("One information set has inconsistent legal actions")
                strategy = _regret_matching(self.regrets[key])
                values = [walk(state.apply(action), public,
                               reach0 * probability if actor == 0 else reach0,
                               reach1 * probability if actor == 1 else reach1)
                          for action, probability in zip(legal, strategy)]
                expected = sum(p * v for p, v in zip(strategy, values))
                own = reach0 if actor == 0 else reach1
                other = reach1 if actor == 0 else reach0
                sign = 1 if actor == 0 else -1
                rd = regret_delta.setdefault(key, [0.0] * len(legal))
                sd = strategy_delta.setdefault(key, [0.0] * len(legal))
                for i, probability in enumerate(strategy):
                    rd[i] += chance * other * sign * (values[i] - expected)
                    sd[i] += chance * own * probability
                return expected

            for first, second, public in all_deals:
                walk(new_hand(first, second), public, 1.0, 1.0)
            for key, delta in regret_delta.items():
                for i, amount in enumerate(delta):
                    self.regrets[key][i] += amount
            for key, delta in strategy_delta.items():
                for i, amount in enumerate(delta):
                    self.strategy_sums[key][i] += amount
            self.iterations += 1

    def average_policy(self) -> TabularPolicy:
        if not self.iterations:
            raise ValueError("Train before requesting the average policy")
        table = {}
        for key, weights in self.strategy_sums.items():
            total = sum(weights)
            if total <= 0:
                raise AssertionError("An information set received no average reach")
            table[key] = {action: weight / total
                          for action, weight in zip(self.actions[key], weights)}
        return TabularPolicy(table)


def _initial_worlds() -> list[World]:
    all_deals = deals()
    return [(new_hand(first, second), public, 1 / len(all_deals))
            for first, second, public in all_deals]


def profile_value(player0: Policy, player1: Policy) -> float:
    """Exact expected net reward for player 0 in chips per hand."""
    def visit(state: GameState, public: Card) -> float:
        if state.phase == Phase.TERMINAL:
            assert state.rewards is not None
            return state.rewards[0]
        if state.phase == Phase.DEAL_PUBLIC:
            return visit(state.deal_public(public), public)
        assert state.actor is not None
        obs = state.observe(state.actor)
        probs = (player0 if state.actor == 0 else player1).probabilities(obs)
        _validate(probs, obs.legal_actions)
        return sum(prob * visit(state.apply(action), public)
                   for action, prob in zip(obs.legal_actions, probs) if prob)

    return sum(weight * visit(state, public)
               for state, public, weight in _initial_worlds())


def best_response(opponent: Policy, responder: int) -> tuple[float, TabularPolicy]:
    """Exact deterministic best response to one fixed opponent policy."""
    return best_response_mixture((opponent,), (1.0,), responder)


def best_response_mixture(opponents: tuple[Policy, ...], priors: tuple[float, ...],
                          responder: int) -> tuple[float, TabularPolicy]:
    """Information-set BR to a type drawn once at the start of each hand.

    A type remains fixed for the whole hand. Opponent actions weight each
    compatible hidden card *and* type; the responder chooses one action across
    those worlds at each information set. Thus observed opponent actions also
    change the responder's implicit within-hand posterior. Chance's future
    public card is hidden until dealt. A new prior can be supplied each hand.
    """
    if responder not in (0, 1):
        raise ValueError("responder must be 0 or 1")
    if not opponents or len(opponents) != len(priors):
        raise ValueError("Provide one probability for every opponent type")
    if any(not math.isfinite(weight) or weight < 0 for weight in priors):
        raise ValueError("Type probabilities must be nonnegative and finite")
    if abs(sum(priors) - 1.0) > 1e-10:
        raise ValueError("Type probabilities must sum to one")
    selected: dict[InfoKey, dict[Action, float]] = {}

    def solve(worlds: list[ModelWorld]) -> float:
        if not worlds:
            return 0.0
        state = worlds[0][0]
        if state.phase == Phase.TERMINAL:
            return sum(weight * world.rewards[responder]
                       for world, _, weight, _ in worlds)  # type: ignore[index]
        if state.phase == Phase.DEAL_PUBLIC:
            by_rank: dict[str, list[ModelWorld]] = defaultdict(list)
            for world, public, weight, model_index in worlds:
                by_rank[public.rank].append((world.deal_public(public), public,
                                             weight, model_index))
            return sum(solve(group) for group in by_rank.values())
        assert state.actor is not None
        legal = state.legal_actions()
        if state.actor == responder:
            groups: dict[InfoKey, list[ModelWorld]] = defaultdict(list)
            for world, public, weight, model_index in worlds:
                groups[info_key(world.observe(responder))].append(
                    (world, public, weight, model_index))
            total = 0.0
            for key, group in groups.items():
                scores = [solve([(world.apply(action), public, weight, model_index)
                                 for world, public, weight, model_index in group])
                          for action in legal]
                best = max(range(len(legal)), key=lambda i: scores[i])
                selected[key] = {action: float(i == best)
                                 for i, action in enumerate(legal)}
                total += scores[best]
            return total
        total = 0.0
        for action_index, action in enumerate(legal):
            successor = []
            for world, public, weight, model_index in worlds:
                obs = world.observe(state.actor)
                probs = opponents[model_index].probabilities(obs)
                _validate(probs, legal)
                if probs[action_index]:
                    successor.append((world.apply(action), public,
                                      weight * probs[action_index], model_index))
            total += solve(successor)
        return total

    all_deals = deals()
    worlds = [(new_hand(first, second), public,
               prior / len(all_deals), model_index)
              for first, second, public in all_deals
              for model_index, prior in enumerate(priors) if prior]
    value = solve(worlds)
    return value, TabularPolicy(selected)


@dataclass(frozen=True)
class Diagnostics:
    value_p0: float
    br_p0: float
    br_p1: float
    nash_conv: float
    exploitability: float


def diagnostics(player0: Policy, player1: Policy) -> Diagnostics:
    """NashConv = BR0 - V0 + BR1 - V1; exploitability = NashConv / 2."""
    value = profile_value(player0, player1)
    br0, _ = best_response(player1, responder=0)
    br1, _ = best_response(player0, responder=1)
    nash_conv = (br0 - value) + (br1 + value)
    return Diagnostics(value, br0, br1, nash_conv, nash_conv / 2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train exact-tree CFR on Leduc v0.1")
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--output", type=Path, default=Path("strategy.json"))
    args = parser.parse_args()
    solver = CFRSolver()
    solver.train(args.iterations)
    strategy = solver.average_policy()
    strategy.save(args.output)
    stats = diagnostics(strategy, strategy)
    print(f"iterations={solver.iterations} infosets={len(strategy.table)}")
    print(f"value_p0={stats.value_p0:.6f} br_p0={stats.br_p0:.6f} "
          f"br_p1={stats.br_p1:.6f} exploitability={stats.exploitability:.6f}")
    print(f"saved={args.output}")


if __name__ == "__main__":
    main()
