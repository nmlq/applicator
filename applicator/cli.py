import argparse
import logging
from pathlib import Path
from typing import Optional, Sequence


def readable_file(value: str) -> Path:
    """Convert a CLI path argument to a readable file path.

    :param value: Path supplied on the command line.
    :return: A validated file path.
    :raises argparse.ArgumentTypeError: If the path is missing or unreadable.
    """
    # argparse uses this exception to report invalid values as CLI errors.
    path = Path(value)
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"file does not exist: {value}")
    if not path.stat().st_mode & 0o444:
        raise argparse.ArgumentTypeError(f"file is not readable: {value}")
    return path


def build_parser() -> argparse.ArgumentParser:
    """Build and return the command-line argument parser.

    :return: Parser configured for the ``applicator`` command.
    """
    parser = argparse.ArgumentParser(
        prog="applicator",
        description="Apply an LLM prompt to one CSV column, one row at a time.",
    )
    # A subcommand keeps room for additional CLI operations in the future.
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser(
        "run",
        help="apply a prompt to a CSV column",
        description="Apply an LLM prompt to one CSV column, one row at a time.",
    )
    run_parser.add_argument(
        "input_csv",
        type=readable_file,
        help="input CSV file",
    )
    run_parser.add_argument(
        "prompt_json",
        type=readable_file,
        help="JSON file containing the prompt template",
    )
    run_parser.add_argument(
        "--column",
        "-c",
        required=True,
        help="CSV column to process",
    )
    run_parser.add_argument(
        "--output",
        "-o",
        required=True,
        dest="output_csv",
        type=Path,
        help="output CSV file",
    )
    run_parser.add_argument(
        "--model",
        "-m",
        default="gpt-4.1-mini",
        help="model to use (default: gpt-4.1-mini)",
    )
    run_parser.add_argument(
        "--base-url",
        help="OpenAI-compatible API base URL",
    )
    run_parser.add_argument(
        "--api-key",
        help="API key for the model provider",
    )
    run_parser.add_argument(
        "--output-column",
        default="applicator_output",
        help="name of the generated CSV column (default: applicator_output)",
    )
    run_parser.set_defaults(handler=run)

    return parser


def run(args: argparse.Namespace) -> None:
    """Execute the ``run`` command with parsed arguments.

    :param args: Namespace produced by :func:`build_parser`.
    """
    # Import lazily so help and parser usage do not require the LLM dependency.
    from .core import apply_prompt_to_csv

    apply_prompt_to_csv(
        input_csv=args.input_csv,
        prompt_json=args.prompt_json,
        column=args.column,
        output_csv=args.output_csv,
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        output_column=args.output_column,
    )


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Parse CLI arguments and dispatch the selected command.

    :param argv: Optional argument sequence; defaults to ``sys.argv``.
    """
    # Passing argv explicitly keeps this entry point straightforward to test.
    args = build_parser().parse_args(argv)
    # Logs go to stderr so stdout stays clean for any piped output.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    args.handler(args)


if __name__ == "__main__":
    main()
