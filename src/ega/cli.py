"""CLI entrypoint for EGA tooling."""

import argparse


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser for the `ega` command."""
    parser = argparse.ArgumentParser(
        prog="ega",
        description="Evidence-Gated Answering (enforcement/decision layer)",
    )
    parser.add_argument(
        "--version",
        action="store_true",
        help="Print package version and exit.",
    )
    return parser


def main() -> int:
    """Run the CLI.

    TODO: Add commands for policy validation and decision simulation.
    """
    parser = build_parser()
    args = parser.parse_args()
    if args.version:
        from ega import __version__

        print(__version__)
        return 0
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
