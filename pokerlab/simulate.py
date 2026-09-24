"""Seeded hand runner. Agents never receive the simulator's private state."""

from __future__ import annotations

import random
from typing import Protocol

from .game import Action, GameState, Phase, PlayerObservation, all_cards, new_hand


class Agent(Protocol):
    def choose(self, observation: PlayerObservation, rng: random.Random) -> Action: ...


class RandomAgent:
    def choose(self, observation: PlayerObservation, rng: random.Random) -> Action:
        return rng.choice(observation.legal_actions)


def run_hand(seed: int, agents: tuple[Agent, Agent] | None = None) -> GameState:
    agents = agents or (RandomAgent(), RandomAgent())
    deck_rng = random.Random(seed)
    # Action draws do not consume or alter the card generator's random stream.
    action_rngs = (random.Random(f"{seed}:action:0"),
                   random.Random(f"{seed}:action:1"))
    cards = deck_rng.sample(all_cards(), 6)
    state = new_hand(cards[0], cards[1])
    while state.phase != Phase.TERMINAL:
        if state.phase == Phase.DEAL_PUBLIC:
            state = state.deal_public(cards[2])
        else:
            assert state.actor is not None
            actor = state.actor
            choice = agents[actor].choose(state.observe(actor), action_rngs[actor])
            state = state.apply(choice)
    return state


def format_trace(state: GameState) -> str:
    """Post-hand evaluator trace; never supply this to an agent."""
    lines = [f"Private cards: P0={state.private_cards[0]}, P1={state.private_cards[1]}",
             "Both players ante 1 (pot=2)."]
    for event in state.actions:
        if event.round_index == 1 and not any("Public card:" in x for x in lines):
            lines.append(f"Public card: {state.public_card}")
        lines.append(f"Round {event.round_index + 1}: P{event.player} {event.action.value}")
    if state.public_card is not None and not any("Public card:" in x for x in lines):
        lines.append(f"Public card: {state.public_card}")
    lines.append(f"Contributions: {state.contributions}; pot={sum(state.contributions)}")
    lines.append(f"Net rewards: P0={state.rewards[0]:+g}, P1={state.rewards[1]:+g}")
    return "\n".join(lines)
