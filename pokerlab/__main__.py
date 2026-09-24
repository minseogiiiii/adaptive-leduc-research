"""Demonstrate one reproducible complete hand."""

import argparse

from .simulate import format_trace, run_hand


def main() -> None:
    parser = argparse.ArgumentParser(description="Trace a seeded Leduc hand")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    print(format_trace(run_hand(args.seed)))


if __name__ == "__main__":
    main()

