"""Fully specified, immutable two-player fixed-limit Leduc hand state."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum


class RuleError(ValueError):
    """An action or chance outcome is forbidden by the frozen rules."""


@dataclass(frozen=True, order=True)
class Card:
    rank: str
    copy: int

    def __post_init__(self) -> None:
        if self.rank not in ("J", "Q", "K") or self.copy not in (0, 1):
            raise RuleError(f"Invalid card: {self.rank}{self.copy}")

    def __str__(self) -> str:
        return f"{self.rank}{self.copy}"


def all_cards() -> tuple[Card, ...]:
    return tuple(Card(rank, copy) for rank in ("J", "Q", "K") for copy in (0, 1))


class Action(str, Enum):
    CHECK = "check"
    BET = "bet"
    FOLD = "fold"
    CALL = "call"
    RAISE = "raise"


class Phase(str, Enum):
    BETTING = "betting"
    DEAL_PUBLIC = "deal_public"
    TERMINAL = "terminal"


@dataclass(frozen=True)
class PublicAction:
    round_index: int
    player: int
    action: Action


@dataclass(frozen=True)
class PlayerObservation:
    player: int
    own_card: Card
    public_card: Card | None
    opponent_card_at_showdown: Card | None
    actions: tuple[PublicAction, ...]
    contributions: tuple[int, int]
    round_index: int
    phase: Phase
    actor: int | None
    legal_actions: tuple[Action, ...]


@dataclass(frozen=True)
class GameState:
    """Privileged simulator state. Policies must receive observe(player) only."""

    private_cards: tuple[Card, Card]
    contributions: tuple[int, int] = (1, 1)
    public_card: Card | None = None
    round_index: int = 0
    phase: Phase = Phase.BETTING
    actor: int | None = 0
    round_bets: int = 0
    round_actions: tuple[Action, ...] = ()
    actions: tuple[PublicAction, ...] = ()
    winner: int | None = None  # Fold/showdown winner; a tied showdown uses None.
    showdown: bool = False
    rewards: tuple[float, float] | None = None

    def legal_actions(self) -> tuple[Action, ...]:
        if self.phase != Phase.BETTING or self.actor is None:
            return ()
        me = self.actor
        outstanding = max(self.contributions) - self.contributions[me]
        if outstanding:
            return ((Action.FOLD, Action.CALL, Action.RAISE)
                    if self.round_bets < 2 else (Action.FOLD, Action.CALL))
        return ((Action.CHECK, Action.BET) if self.round_bets == 0
                else (Action.CHECK,))

    def observe(self, player: int) -> PlayerObservation:
        if player not in (0, 1):
            raise RuleError("Player must be 0 or 1")
        return PlayerObservation(
            player=player,
            own_card=self.private_cards[player],
            public_card=self.public_card,
            opponent_card_at_showdown=(
                self.private_cards[1 - player]
                if self.phase == Phase.TERMINAL and self.showdown
                else None
            ),
            actions=self.actions,
            contributions=self.contributions,
            round_index=self.round_index,
            phase=self.phase,
            actor=self.actor,
            legal_actions=self.legal_actions() if self.actor == player else (),
        )

    def apply(self, action: Action) -> GameState:
        if action not in self.legal_actions():
            raise RuleError(f"{action} is illegal; choose from {self.legal_actions()}")
        assert self.actor is not None
        me = self.actor
        them = 1 - me
        history = self.actions + (PublicAction(self.round_index, me, action),)
        local = self.round_actions + (action,)
        if action == Action.FOLD:
            return self._settle(
                winner=them,
                actions=history,
                round_actions=local,
            )
        if action == Action.CHECK:
            state = replace(self, actor=them, actions=history, round_actions=local)
            return state._close_round() if self.round_actions == (Action.CHECK,) else state
        if action == Action.CALL:
            amount = max(self.contributions) - self.contributions[me]
            chips = list(self.contributions)
            chips[me] += amount
            return replace(
                self, contributions=tuple(chips), actions=history, round_actions=local,
            )._close_round()
        if action in (Action.BET, Action.RAISE):
            amount = (2, 4)[self.round_index]
            if action == Action.RAISE:
                amount += max(self.contributions) - self.contributions[me]
            chips = list(self.contributions)
            chips[me] += amount
            return replace(
                self, contributions=tuple(chips), actor=them,
                actions=history, round_actions=local,
                round_bets=self.round_bets + 1,
            )
        raise AssertionError("Every legal action must have a transition")

    def _close_round(self) -> GameState:
        if self.round_index == 0:
            return replace(self, phase=Phase.DEAL_PUBLIC, actor=None,
                           round_bets=0, round_actions=())
        if self.public_card is None:
            raise RuleError("Cannot settle a showdown without a public card")
        c0, c1 = self.private_cards
        score0 = (int(c0.rank == self.public_card.rank), "JQK".index(c0.rank))
        score1 = (int(c1.rank == self.public_card.rank), "JQK".index(c1.rank))
        winner = 0 if score0 > score1 else 1 if score1 > score0 else None
        return self._settle(winner=winner, showdown=True, round_actions=())

    def deal_public(self, card: Card) -> GameState:
        if self.phase != Phase.DEAL_PUBLIC:
            raise RuleError("Public card may only be dealt between rounds")
        if card not in all_cards() or card in self.private_cards:
            raise RuleError("Public card must be an unused physical card")
        return replace(self, public_card=card, round_index=1,
                       phase=Phase.BETTING, actor=0)

    def _settle(self, winner: int | None, **changes: object) -> GameState:
        pot = sum(self.contributions)
        if winner is None:
            income = (pot / 2, pot / 2)
        else:
            income = (float(pot), 0.0) if winner == 0 else (0.0, float(pot))
        rewards = (income[0] - self.contributions[0],
                   income[1] - self.contributions[1])
        return replace(self, phase=Phase.TERMINAL, actor=None,
                       winner=winner, rewards=rewards, **changes)


def new_hand(first: Card, second: Card) -> GameState:
    if first == second or first not in all_cards() or second not in all_cards():
        raise RuleError("The two private cards must be distinct deck cards")
    return GameState(private_cards=(first, second))
