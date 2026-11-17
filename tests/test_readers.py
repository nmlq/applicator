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


def test_read_csv_transforms_rows_and_drops_input_column(tmp_path, capsys):
    """Verify row transformation, input-column removal, and progress output.

    :param tmp_path: Pytest-provided temporary directory.
    :param capsys: Pytest output-capture fixture.
    """
    input_path = tmp_path / "input.csv"
    output_path = tmp_path / "output.csv"
    input_path.write_text("name,age\nAda,36\nGrace,28\n", encoding="utf-8")

    read_csv(
        input_csv=input_path,
        output_csv=output_path,
        input_columns="name",
        output_column="greeting",
        transform=lambda values: [f"Hello {value}" for value in values],
    )

    with output_path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    assert rows == [
        {"age": "36", "greeting": "Hello Ada"},
        {"age": "28", "greeting": "Hello Grace"},
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


def test_read_csv_joins_multiple_columns_with_newline(tmp_path):
    """Verify several input columns are joined per row, in order, by a newline.

    :param tmp_path: Pytest-provided temporary directory.
    """
    input_path = tmp_path / "input.csv"
    output_path = tmp_path / "output.csv"
    input_path.write_text(
        "id,ImpressionText,ReportText\n"
        "1,No acute findings.,Brain MRI is normal.\n"
        '2,,"Line one\nline two"\n',
        encoding="utf-8",
    )
    batches = []

    def transform(values):
        """Record each batch and echo its values."""
        batches.append(values)
        return values

    read_csv(
        input_csv=input_path,
        output_csv=output_path,
        input_columns=["ReportText", "ImpressionText"],
        output_column="result",
        transform=transform,
        batch_size=2,
    )

    expected = [
        "Brain MRI is normal.\nNo acute findings.",
        # An empty value still contributes its separator.
        "Line one\nline two\n",
    ]
    assert batches == [expected]

    with output_path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    # Input columns are dropped; the rest keep their original order.
    assert list(rows[0]) == ["id", "result"]
    assert [row["result"] for row in rows] == expected


def test_read_csv_lists_every_missing_input_column(tmp_path):
    """Verify all missing input columns are named in one error.

    :param tmp_path: Pytest-provided temporary directory.
    """
    input_path = tmp_path / "input.csv"
    output_path = tmp_path / "output.csv"
    input_path.write_text("name,age\nAda,36\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Columns 'first', 'last' not found"):
        read_csv(input_path, output_path, ["name", "first", "last"], "result", str.upper)

    assert not output_path.exists()


def test_read_csv_rejects_empty_input_columns(tmp_path):
    """Verify an empty column list is rejected.

    :param tmp_path: Pytest-provided temporary directory.
    """
    input_path = tmp_path / "input.csv"
    output_path = tmp_path / "output.csv"
    input_path.write_text("name\nAda\n", encoding="utf-8")

    with pytest.raises(ValueError, match="At least one input column"):
        read_csv(input_path, output_path, [], "result", str.upper)


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
        input_columns="text",
        output_column="result",
        transform=lambda values: [prompt.format(input=value) for value in values],
    )

    with output_path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    assert rows == [
        {
            "id": "1",
            "result": (
                "Analyze the following input and return a concise result:\n\n"
                "The first example row."
            ),
        },
        {
            "id": "2",
            "result": (
                "Analyze the following input and return a concise result:\n\n"
                "The second example row."
            ),
        },
    ]


def test_read_csv_passes_rows_in_batches(tmp_path):
    """Verify rows are transformed in batches and written in input order.

    :param tmp_path: Pytest-provided temporary directory.
    """
    input_path = tmp_path / "input.csv"
    output_path = tmp_path / "output.csv"
    input_path.write_text("name\nAda\nGrace\nLinus\n", encoding="utf-8")
    batches = []

    def transform(values):
        """Record each batch and upper-case its values."""
        batches.append(values)
        return [value.upper() for value in values]

    read_csv(input_path, output_path, "name", "result", transform, batch_size=2)

    with output_path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    assert batches == [["Ada", "Grace"], ["Linus"]]
    assert [row["result"] for row in rows] == ["ADA", "GRACE", "LINUS"]


@pytest.mark.parametrize(
    "limit, expected_batches",
    [
        (1, [["Ada"]]),
        (3, [["Ada", "Grace"], ["Linus"]]),
        (10, [["Ada", "Grace"], ["Linus", "Barbara"]]),
        (None, [["Ada", "Grace"], ["Linus", "Barbara"]]),
    ],
)
def test_read_csv_limits_rows(tmp_path, limit, expected_batches):
    """Verify ``limit`` caps the rows read, across batch boundaries.

    :param tmp_path: Pytest-provided temporary directory.
    :param limit: Maximum number of rows to read.
    :param expected_batches: Batches the transform should receive.
    """
    input_path = tmp_path / "input.csv"
    output_path = tmp_path / "output.csv"
    input_path.write_text("name\nAda\nGrace\nLinus\nBarbara\n", encoding="utf-8")
    batches = []

    def transform(values):
        """Record each batch and upper-case its values."""
        batches.append(values)
        return [value.upper() for value in values]

    read_csv(
        input_path, output_path, "name", "result", transform, batch_size=2, limit=limit
    )

    with output_path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    assert batches == expected_batches
    expected = [value.upper() for batch in expected_batches for value in batch]
    assert [row["result"] for row in rows] == expected


def test_read_csv_keeps_non_input_columns_in_order(tmp_path):
    """Verify only the input columns are removed and the rest keep their order.

    :param tmp_path: Pytest-provided temporary directory.
    """
    input_path = tmp_path / "input.csv"
    output_path = tmp_path / "output.csv"
    input_path.write_text(
        "ReportText,id,ImpressionText,site\nA,1,B,north\n",
        encoding="utf-8",
    )

    read_csv(
        input_path,
        output_path,
        ["ReportText", "ImpressionText"],
        "result",
        lambda values: [value.upper() for value in values],
    )

    with output_path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    assert rows == [{"id": "1", "site": "north", "result": "A\nB"}]
