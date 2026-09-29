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


def positive_int(value: str) -> int:
    """Convert a CLI argument to a positive integer.

    :param value: Number supplied on the command line.
    :return: The parsed integer.
    :raises argparse.ArgumentTypeError: If the value is not a positive integer.
    """
    try:
        number = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not an integer: {value}") from None
    if number < 1:
        raise argparse.ArgumentTypeError(f"must be at least 1: {value}")
    return number


# Options that only apply to a local model, mapped to LocalModelConfig fields.
LOCAL_OPTIONS = {
    "device": "--device",
    "dtype": "--dtype",
    "max_new_tokens": "--max-new-tokens",
    "temperature": "--temperature",
    "top_p": "--top-p",
    "top_k": "--top-k",
    "repetition_penalty": "--repetition-penalty",
    "trust_remote_code": "--trust-remote-code",
}


def validate_run_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Reject option combinations that mix the API and local model paths.

    :param parser: Parser used to report errors.
    :param args: Parsed ``run`` arguments.
    """
    # Exits through parser.error so mistakes read like any other usage error.
    if args.local_model:
        for dest, flag in (("base_url", "--base-url"), ("api_key", "--api-key")):
            if getattr(args, dest) is not None:
                parser.error(f"{flag} cannot be used with --local-model")
    else:
        for dest, flag in LOCAL_OPTIONS.items():
            if getattr(args, dest) is not None:
                parser.error(f"{flag} requires --local-model")


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

    # Local model options load a Hugging Face model in-process instead of calling an API.
    local = run_parser.add_argument_group(
        "local model",
        "Run a Hugging Face model on this machine instead of calling an API. "
        "Requires: pip install 'applicator[local]'",
    )
    local.add_argument(
        "--local-model",
        metavar="MODEL",
        help="Hugging Face model id or local model directory to run in-process",
    )
    local.add_argument(
        "--device",
        help="device map: auto, cuda, cuda:N, mps or cpu (default: auto)",
    )
    local.add_argument(
        "--dtype",
        choices=["auto", "float16", "bfloat16", "float32"],
        help="weight dtype (default: auto)",
    )
    local.add_argument(
        "--max-new-tokens",
        type=positive_int,
        help="maximum tokens generated per row (default: 512)",
    )
    local.add_argument(
        "--temperature",
        type=float,
        help="sampling temperature; omit or 0 for greedy decoding",
    )
    local.add_argument(
        "--top-p",
        type=float,
        help="nucleus sampling probability (used when --temperature > 0)",
    )
    local.add_argument(
        "--top-k",
        type=positive_int,
        help="top-k sampling cutoff (used when --temperature > 0)",
    )
    local.add_argument(
        "--repetition-penalty",
        type=float,
        help="penalty for repeated tokens, e.g. 1.1",
    )
    local.add_argument(
        "--trust-remote-code",
        action="store_true",
        default=None,
        help="allow the model repository to run custom code",
    )
    run_parser.set_defaults(handler=run)

    return parser


def run(args: argparse.Namespace) -> None:
    """Execute the ``run`` command with parsed arguments.

    :param args: Namespace produced by :func:`build_parser`.
    """
    # Import lazily so help and parser usage do not require the LLM dependency.
    from .core import LocalModelConfig, apply_prompt_to_csv

    local_model = None
    if args.local_model:
        # Unset options fall back to the LocalModelConfig defaults.
        options = {
            dest: getattr(args, dest)
            for dest in LOCAL_OPTIONS
            if getattr(args, dest) is not None
        }
        local_model = LocalModelConfig(model=args.local_model, **options)

    apply_prompt_to_csv(
        input_csv=args.input_csv,
        prompt_json=args.prompt_json,
        column=args.column,
        output_csv=args.output_csv,
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        output_column=args.output_column,
        local_model=local_model,
    )


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Parse CLI arguments and dispatch the selected command.

    :param argv: Optional argument sequence; defaults to ``sys.argv``.
    """
    # Passing argv explicitly keeps this entry point straightforward to test.
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "command", None) == "run":
        validate_run_args(parser, args)
    # Logs go to stderr so stdout stays clean for any piped output.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    # The HTTP clients log every API request at INFO; only surface warnings and errors.
    for name in ("httpx", "httpx2"):
        logging.getLogger(name).setLevel(logging.WARNING)
    # transformers repeats generation-config warnings on every row; keep errors only.
    logging.getLogger("transformers").setLevel(logging.ERROR)
    args.handler(args)


if __name__ == "__main__":
    main()
