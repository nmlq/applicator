import csv as csv_module
import json
from pathlib import Path
from typing import Callable


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
    transform: Callable[[str], str],
) -> None:
    """Transform one CSV column and write the results to another CSV file.

    :param input_csv: Source CSV file.
    :param output_csv: Destination CSV file.
    :param input_column: Column whose values are transformed.
    :param output_column: Column receiving transformed values.
    :param transform: Function applied to each input value.
    :raises ValueError: If the input has no header or the column is missing.
    """
    # DictReader preserves the input schema while allowing named-column access.
    with input_csv.open("r", encoding="utf-8-sig", newline="") as src:
        reader = csv_module.DictReader(src)

        if not reader.fieldnames:
            raise ValueError("Input CSV has no header.")

        if input_column not in reader.fieldnames:
            raise ValueError(
                f"Column '{input_column}' not found. "
                f"Available columns: {', '.join(reader.fieldnames)}"
            )

        fieldnames = list(reader.fieldnames)
        if output_column not in fieldnames:
            fieldnames.append(output_column)

        # Open the destination only after validating the source header.
        with output_csv.open("w", encoding="utf-8", newline="") as dst:
            writer = csv_module.DictWriter(dst, fieldnames=fieldnames)
            writer.writeheader()

            for row_number, row in enumerate(reader, start=1):
                value = row.get(input_column, "")
                # Flush each row so completed work remains available immediately.
                row[output_column] = transform(value or "")
                writer.writerow(row)
                dst.flush()
                print(f"Processed row {row_number}", flush=True)
