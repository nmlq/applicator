"""Measure local-model throughput at several batch sizes on the same rows.

The model is loaded once, then the input CSV is processed at each batch size.
For each size the script reports rows per second, the speedup over the first
batch size, peak GPU memory (CUDA only), and how many outputs match the first
batch size. By default it uses the synthetic MRI reports in tests/fixtures.

Example:

    python scripts/benchmark_batch_size.py \\
        --local-model Qwen/Qwen3-4B --device cuda --dtype float32 --no-think \\
        --batch-sizes 1 4 8 16
"""
import argparse
import logging
import os
import tempfile
import time
from pathlib import Path

import pandas

from applicator.core import LocalModelConfig, build_local_llm, make_transform
from applicator.readers import read_csv, read_json

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "tests" / "fixtures"
OUTPUT_COLUMN = "applicator_output"


def build_parser() -> argparse.ArgumentParser:
    """Build the benchmark argument parser.

    :return: Parser for the benchmark options.
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--input", type=Path, default=FIXTURES_DIR / "mri_reports.example.csv")
    parser.add_argument("--prompt", type=Path, default=FIXTURES_DIR / "mri_prompt.example.json")
    parser.add_argument("--column", default="report")
    parser.add_argument("--rows", type=int, help="use only the first N rows")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    parser.add_argument("--output-dir", type=Path, help="keep each run's CSV here")
    parser.add_argument("--local-model", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--no-think", action="store_true")
    parser.add_argument("--load-in-4bit", action="store_true")
    return parser


def main() -> None:
    """Run the batch size sweep and print a results table."""
    args = build_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    # Same as the CLI: transformers repeats generation warnings on every batch.
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

    import torch

    output_dir = args.output_dir or Path(tempfile.mkdtemp(prefix="applicator-benchmark-"))
    output_dir.mkdir(parents=True, exist_ok=True)

    input_csv = args.input
    if args.rows:
        input_csv = output_dir / "input.csv"
        pandas.read_csv(args.input, dtype=str, keep_default_na=False).head(args.rows).to_csv(input_csv, index=False)
    values = pandas.read_csv(input_csv, dtype=str, keep_default_na=False)[args.column].tolist()

    config = LocalModelConfig(
        model=args.local_model,
        device=args.device,
        dtype=args.dtype,
        max_new_tokens=args.max_new_tokens,
        no_think=args.no_think,
        load_in_4bit=args.load_in_4bit,
    )
    # The pipeline is built for the largest batch; each run then caps how many
    # prompts LangChain hands it at once, so one loaded model serves every size.
    llm = build_local_llm(config, batch_size=max(args.batch_sizes))
    transform = make_transform(llm, read_json(args.prompt))
    cuda = torch.cuda.is_available()

    # The first generation pays one-off costs such as CUDA initialization.
    logging.info("Warming up")
    transform(values[:1])

    results = []
    baseline = None
    for batch_size in args.batch_sizes:
        llm.pipeline.batch_size = batch_size
        output_csv = output_dir / f"batch_{batch_size}.csv"
        if cuda:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        logging.info("Batch size %d", batch_size)
        start = time.perf_counter()
        try:
            read_csv(input_csv, output_csv, args.column, OUTPUT_COLUMN, transform, batch_size=batch_size)
        except torch.OutOfMemoryError:
            logging.warning("Out of GPU memory at batch size %d; skipping larger sizes", batch_size)
            results.append((batch_size, None, None, None))
            break
        seconds = time.perf_counter() - start
        outputs = pandas.read_csv(output_csv, dtype=str, keep_default_na=False)[OUTPUT_COLUMN].tolist()
        if baseline is None:
            baseline = outputs
        matches = sum(a == b for a, b in zip(outputs, baseline))
        peak_gb = torch.cuda.max_memory_allocated() / 1e9 if cuda else None
        results.append((batch_size, seconds, peak_gb, matches))

    rows = len(values)
    first_rate = rows / results[0][1] if results[0][1] else None
    print(f"\n{rows} rows, model {args.local_model}, outputs in {output_dir}\n")
    print(f"{'batch':>5}  {'seconds':>8}  {'rows/s':>7}  {'speedup':>7}  {'peak GPU':>8}  {'match first':>11}")
    for batch_size, seconds, peak_gb, matches in results:
        if seconds is None:
            print(f"{batch_size:>5}  {'out of GPU memory':>45}")
            continue
        rate = rows / seconds
        peak = f"{peak_gb:.1f} GB" if peak_gb is not None else "n/a"
        speedup = f"{rate / first_rate:.2f}x" if first_rate else "n/a"
        print(f"{batch_size:>5}  {seconds:>8.1f}  {rate:>7.2f}  {speedup:>7}  {peak:>8}  {f'{matches}/{rows}':>11}")


if __name__ == "__main__":
    main()
