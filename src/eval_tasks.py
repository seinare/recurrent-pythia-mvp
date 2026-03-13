from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluation task entrypoints for recurrent Pythia MVP")
    parser.add_argument(
        "--task",
        choices=["length-bpc", "passkey"],
        required=True,
        help="Planned evaluation task to run.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    raise NotImplementedError(f"Task '{args.task}' is not implemented yet.")


if __name__ == "__main__":
    main()
