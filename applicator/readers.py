import json
import logging
from pathlib import Path
from typing import Callable, List

import pandas
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

logger = logging.getLogger(__name__)


def read_json(path: Path) -> str:
    """Read and validate a prompt template from a JSON file.

    :param path: JSON file containing a ``prompt`` string.
    :return: The validated prompt template.
    :raises ValueError: If the prompt is missing, empty, or lacks ``{input}``.
    """
    # Keep JSON parsing here so callers receive one consistent prompt shape.
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    prompt = data.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Prompt JSON must contain a non-empty 'prompt' string.")

    if "{input}" not in prompt:
        raise ValueError("Prompt must contain the {input} placeholder.")

    return prompt


def read_csv(
    input_csv: Path,
    output_csv: Path,
    input_column: str,
    output_column: str,
    transform: Callable[[List[str]], List[str]],
    batch_size: int = 1,
) -> None:
    """Transform one CSV column and write the results to another CSV file.

    :param input_csv: Source CSV file.
    :param output_csv: Destination CSV file.
    :param input_column: Column whose values are transformed.
    :param output_column: Column receiving transformed values.
    :param transform: Function mapping a batch of input values to one output
        value each, in the same order.
    :param batch_size: Number of rows passed to each ``transform`` call.
    :raises ValueError: If the input has no header or the column is missing.
    """
    # Read only the header first so validation happens before any output exists.
    # dtype=str and keep_default_na=False keep values exactly as written.
    read_options = dict(encoding="utf-8-sig", dtype=str, keep_default_na=False)
    try:
        header = pandas.read_csv(input_csv, nrows=0, **read_options)
    except pandas.errors.EmptyDataError:
        raise ValueError("Input CSV has no header.") from None

    fieldnames = list(header.columns)
    if input_column not in fieldnames:
        raise ValueError(
            f"Column '{input_column}' not found. "
            f"Available columns: {', '.join(fieldnames)}"
        )

    if output_column not in fieldnames:
        fieldnames.append(output_column)

    # Open the destination only after validating the source header.
    with output_csv.open("w", encoding="utf-8", newline="") as dst:
        pandas.DataFrame(columns=fieldnames).to_csv(dst, index=False)

        logger.info("Writing results to %s", output_csv)
        # Route log records through tqdm so they don't garble the progress bar.
        with logging_redirect_tqdm():
            # chunksize buffers one batch of rows at a time instead of the whole file.
            # No total: counting rows would mean reading the file twice.
            chunks = pandas.read_csv(input_csv, chunksize=batch_size, **read_options)
            with tqdm(desc="Processing rows", unit="row") as progress:
                for chunk_df in chunks:
                    chunk_df[output_column] = transform(chunk_df[input_column].tolist())
                    chunk_df.to_csv(dst, columns=fieldnames, header=False, index=False)
                    # Flush each batch so completed work remains available immediately.
                    dst.flush()
                    progress.update(len(chunk_df))
