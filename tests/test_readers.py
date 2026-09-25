import csv
import json

import pytest

from applicator.readers import read_csv, read_json


def test_read_json_returns_prompt_template(tmp_path):
    """Verify that a valid JSON prompt is returned unchanged.

    :param tmp_path: Pytest-provided temporary directory.
    """
    prompt_path = tmp_path / "prompt.json"
    prompt_path.write_text(
        json.dumps({"prompt": "Summarize {input}"}),
        encoding="utf-8",
    )

    assert read_json(prompt_path) == "Summarize {input}"


@pytest.mark.parametrize(
    "payload, expected_message",
    [
        ({}, "non-empty 'prompt' string"),
        ({"prompt": ""}, "non-empty 'prompt' string"),
        ({"prompt": "No placeholder"}, "contain the {input} placeholder"),
    ],
)
def test_read_json_rejects_invalid_prompt(payload, expected_message, tmp_path):
    """Verify that malformed prompt JSON raises ``ValueError``.

    :param payload: JSON object written to the temporary prompt file.
    :param expected_message: Expected validation message fragment.
    :param tmp_path: Pytest-provided temporary directory.
    """
    prompt_path = tmp_path / "prompt.json"
    prompt_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=expected_message):
        read_json(prompt_path)


def test_read_csv_transforms_rows_and_preserves_columns(tmp_path, capsys):
    """Verify row transformation, column preservation, and progress output.

    :param tmp_path: Pytest-provided temporary directory.
    :param capsys: Pytest output-capture fixture.
    """
    input_path = tmp_path / "input.csv"
    output_path = tmp_path / "output.csv"
    input_path.write_text("name,age\nAda,36\nGrace,28\n", encoding="utf-8")

    read_csv(
        input_csv=input_path,
        output_csv=output_path,
        input_column="name",
        output_column="greeting",
        transform=lambda value: f"Hello {value}",
    )

    with output_path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    assert rows == [
        {"name": "Ada", "age": "36", "greeting": "Hello Ada"},
        {"name": "Grace", "age": "28", "greeting": "Hello Grace"},
    ]
    # Progress goes to stderr via tqdm, leaving stdout clean.
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Processing rows" in captured.err


def test_read_csv_rejects_input_without_header(tmp_path):
    """Verify that a headerless CSV is rejected.

    :param tmp_path: Pytest-provided temporary directory.
    """
    input_path = tmp_path / "input.csv"
    output_path = tmp_path / "output.csv"
    input_path.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="Input CSV has no header"):
        read_csv(input_path, output_path, "name", "result", str.upper)


def test_read_csv_rejects_unknown_input_column(tmp_path):
    """Verify that an unknown input column is rejected.

    :param tmp_path: Pytest-provided temporary directory.
    """
    input_path = tmp_path / "input.csv"
    output_path = tmp_path / "output.csv"
    input_path.write_text("name,age\nAda,36\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Column 'missing' not found"):
        read_csv(input_path, output_path, "missing", "result", str.upper)


def test_read_example_files(example_input_csv, example_prompt_json, tmp_path):
    """Verify processing using the shared example input files.

    :param example_input_csv: Shared example CSV fixture.
    :param example_prompt_json: Shared example prompt fixture.
    :param tmp_path: Pytest-provided temporary directory.
    """
    prompt = read_json(example_prompt_json)
    output_path = tmp_path / "output.csv"

    read_csv(
        input_csv=example_input_csv,
        output_csv=output_path,
        input_column="text",
        output_column="result",
        transform=lambda value: prompt.format(input=value),
    )

    with output_path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    assert rows == [
        {
            "id": "1",
            "text": "The first example row.",
            "result": (
                "Analyze the following input and return a concise result:\n\n"
                "The first example row."
            ),
        },
        {
            "id": "2",
            "text": "The second example row.",
            "result": (
                "Analyze the following input and return a concise result:\n\n"
                "The second example row."
            ),
        },
    ]
