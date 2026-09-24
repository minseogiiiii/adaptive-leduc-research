# Two-player Leduc: frozen project rules (version 0.1)

This document is the contract for the Week 1 engine. Some published variants
use different antes or bet caps. Any outside comparison requires matching
these choices first.

## Cards and ranks

The six distinct physical cards are J0, J1, Q0, Q1, K0, K1. The numeric suffix
distinguishes copies; it does not create a flush or affect hand strength. Each
player receives one private card, sampled without replacement. After the
first betting round, a remaining card becomes public. No card is dealt if
someone folds before that point.

At showdown, a private card matching the public rank makes a pair and beats
every non-pair. Otherwise, the higher private rank wins (J < Q < K). Equal
strength splits the pot. The public card cannot make both players a pair
because there are only two cards of each rank.

## Stakes, actors and legal actions

Both players ante 1 chip. Player 0 acts first in **each** of two rounds. The
first round's fixed bet/raise increment is 2 chips; the second round's is 4.
At most two bet/raise actions total are allowed per round: one opening `BET`
and, if offered, one `RAISE`. No check-raise is possible after the bet cap is
reached. No-limit stack sizes and all-in actions do not exist in this variant.

When no bet is outstanding, the actor may `CHECK`. If the round has no bet,
the actor may also `BET`. Two checks in a row end the round. If a bet is
outstanding, the actor may `FOLD` or `CALL`; `RAISE` is also legal while fewer
than two bet/raise actions have occurred. `CALL` matches the outstanding
contribution and ends the round. A `FOLD` ends the entire hand immediately.

Round 1 ends with either two checks or a call; the environment then deals
the public card. Round 2 follows the same action rules and ends in showdown.
Every action and the actor taking it is public. A winner receives the entire
pot; a tie divides it equally. Each player's *net* reward equals chips
received minus chips contributed, and the two net rewards sum to zero.

## Information boundary

`GameState` contains both private cards for the evaluator, while a policy
receives only `PlayerObservation`. The observation includes the requesting
player's own card; the public card if dealt; public actions with actor and
round; both players' contributions; whose turn it is; and currently legal
actions if it is this player's turn. It includes the opponent card **only at
showdown**. A folded opponent card remains hidden. Agents must never inspect
`GameState` or the trace's evaluator-only truth. The public card must not
appear in an observation while chance has not dealt it.

## Two complete hand examples

1. Deal player 0 `K1` and player 1 `Q0`. Both ante, creating a pot of 2.
   Round 1: player 0 `CHECK`, player 1 `CHECK`. Reveal `K0`. Round 2:
   player 0 `CHECK`, player 1 `CHECK`. Player 0 has a pair of kings,
   collects the pot of 2, and earns **+1** net; player 1 earns **-1**.
   Player 1 learns the opponent card only at showdown.
2. Deal player 0 `J0` and player 1 `K0`. Both ante. Round 1: player 0
   `BET`s 2 (contributions 3 and 1, pot 4). Player 1 `FOLD`s. Player 0
   receives 4 and earns **+1** net; player 1 earns **-1**. There is no
   public card and neither player learns the folded opponent's private card.

For a capped-raise example, after an opening 2-chip bet player 1 may call 2
and add a 2-chip raise, making their total contribution 5. Player 0 may call
the outstanding 2 chips or fold, but may not raise again in that round.

