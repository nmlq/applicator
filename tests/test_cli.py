import argparse

import pytest

import applicator.cli as cli


def test_readable_file_returns_path_for_readable_file(tmp_path):
    """Verify readable file paths are converted to ``Path`` objects.

    :param tmp_path: Pytest-provided temporary directory.
    """
    path = tmp_path / "input.csv"
    path.write_text("text\nhello\n", encoding="utf-8")

    assert cli.readable_file(str(path)) == path


def test_readable_file_rejects_missing_file():
    """Verify missing files produce an argparse type error."""
    with pytest.raises(argparse.ArgumentTypeError, match="file does not exist"):
        cli.readable_file("missing.csv")


def test_build_parser_parses_run_options(tmp_path):
    """Verify all ``run`` command options are parsed correctly.

    :param tmp_path: Pytest-provided temporary directory.
    """
    input_path = tmp_path / "input.csv"
    prompt_path = tmp_path / "prompt.json"
    input_path.write_text("text\nhello\n", encoding="utf-8")
    prompt_path.write_text('{"prompt": "Echo {input}"}', encoding="utf-8")

    args = cli.build_parser().parse_args(
        [
            "run",
            str(input_path),
            str(prompt_path),
            "--column",
            "text",
            "--output",
            "output.csv",
            "--model",
            "test-model",
            "--base-url",
            "https://example.test/v1",
            "--api-key",
            "test-key",
            "--output-column",
            "result",
        ]
    )

    assert args.command == "run"
    assert args.input_csv == input_path
    assert args.prompt_json == prompt_path
    assert args.column == "text"
    assert args.output_csv.name == "output.csv"
    assert args.model == "test-model"
    assert args.base_url == "https://example.test/v1"
    assert args.api_key == "test-key"
    assert args.output_column == "result"


def test_run_passes_parsed_arguments_to_core(monkeypatch, tmp_path):
    """Verify parsed arguments reach the core application function.

    :param monkeypatch: Pytest monkeypatch fixture.
    :param tmp_path: Pytest-provided temporary directory.
    """
    calls = {}

    import applicator.core as core

    def fake_apply_prompt_to_csv(**kwargs):
        """Capture arguments passed from the CLI command handler."""
        calls.update(kwargs)

    monkeypatch.setattr(core, "apply_prompt_to_csv", fake_apply_prompt_to_csv)

    args = argparse.Namespace(
        input_csv=tmp_path / "input.csv",
        prompt_json=tmp_path / "prompt.json",
        column="text",
        output_csv=tmp_path / "output.csv",
        model="test-model",
        base_url=None,
        api_key=None,
        output_column="result",
    )
    cli.run(args)

    assert calls == {
        "input_csv": args.input_csv,
        "prompt_json": args.prompt_json,
        "column": "text",
        "output_csv": args.output_csv,
        "model": "test-model",
        "base_url": None,
        "api_key": None,
        "output_column": "result",
    }


def test_main_dispatches_to_command_handler(monkeypatch):
    """Verify ``main`` dispatches parsed arguments to their handler.

    :param monkeypatch: Pytest monkeypatch fixture.
    """
    calls = []

    class FakeParser:
        def parse_args(self, argv):
            """Return a namespace containing a test command handler."""
            assert argv == ["run"]
            return argparse.Namespace(handler=lambda args: calls.append(args))

    monkeypatch.setattr(cli, "build_parser", lambda: FakeParser())

    cli.main(["run"])

    assert len(calls) == 1
