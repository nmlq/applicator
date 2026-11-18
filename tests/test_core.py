from types import SimpleNamespace

import applicator.core as core


def test_apply_prompt_to_csv_builds_llm_and_transforms_rows(monkeypatch, tmp_path):
    """Verify model configuration and per-row prompt transformation.

    :param monkeypatch: Pytest monkeypatch fixture.
    :param tmp_path: Pytest-provided temporary directory.
    """
    calls = []
    llm_kwargs = {}

    class FakeLLM:
        def __init__(self, **kwargs):
            """Record model construction arguments."""
            llm_kwargs.update(kwargs)

        def batch(self, prompts):
            """Return predictable content for a batch of model requests."""
            calls.append(prompts)
            return [
                SimpleNamespace(content="Summary")
                if prompt.endswith("Ada")
                else SimpleNamespace(content=["non-string", "content"])
                for prompt in prompts
            ]

    captured = {}

    def fake_read_csv(**kwargs):
        """Capture the CSV reader callback and exercise one batch of two values."""
        captured.update(kwargs)
        captured["results"] = kwargs["transform"](["Ada", "Grace"])

    monkeypatch.setattr(core, "ChatOpenAI", FakeLLM)
    monkeypatch.setattr(core, "read_json", lambda path: "Summarize {input}")
    monkeypatch.setattr(core, "read_csv", fake_read_csv)

    core.apply_prompt_to_csv(
        input_csv=tmp_path / "input.csv",
        prompt_json=tmp_path / "prompt.json",
        columns=["name"],
        output_csv=tmp_path / "output.csv",
        model="test-model",
        base_url="https://example.test/v1",
        api_key="test-key",
        output_column="summary",
        batch_size=2,
        limit=3,
        drop_columns=["site"],
    )

    assert llm_kwargs == {
        "model": "test-model",
        "base_url": "https://example.test/v1",
        "api_key": "test-key",
    }
    assert calls == [["Summarize Ada", "Summarize Grace"]]
    assert captured["input_columns"] == ["name"]
    assert captured["output_column"] == "summary"
    assert captured["batch_size"] == 2
    assert captured["limit"] == 3
    assert captured["drop_columns"] == ["site"]
    assert captured["results"] == ["Summary", "['non-string', 'content']"]


def test_apply_prompt_to_csv_omits_optional_llm_arguments(monkeypatch, tmp_path):
    """Verify optional client arguments are omitted when unset.

    :param monkeypatch: Pytest monkeypatch fixture.
    :param tmp_path: Pytest-provided temporary directory.
    """
    llm_kwargs = {}

    class FakeLLM:
        def __init__(self, **kwargs):
            """Record model construction arguments."""
            llm_kwargs.update(kwargs)

    monkeypatch.setattr(core, "ChatOpenAI", FakeLLM)
    monkeypatch.setattr(core, "read_json", lambda path: "Echo {input}")
    monkeypatch.setattr(core, "read_csv", lambda **kwargs: None)

    core.apply_prompt_to_csv(
        input_csv=tmp_path / "input.csv",
        prompt_json=tmp_path / "prompt.json",
        columns=["text"],
        output_csv=tmp_path / "output.csv",
        model="test-model",
    )

    assert llm_kwargs == {"model": "test-model"}


def test_apply_prompt_to_csv_disables_api_reasoning(monkeypatch, tmp_path):
    """Verify ``no_think`` asks the API for ``reasoning_effort="none"``.

    :param monkeypatch: Pytest monkeypatch fixture.
    :param tmp_path: Pytest-provided temporary directory.
    """
    llm_kwargs = {}

    class FakeLLM:
        def __init__(self, **kwargs):
            """Record model construction arguments."""
            llm_kwargs.update(kwargs)

    monkeypatch.setattr(core, "ChatOpenAI", FakeLLM)
    monkeypatch.setattr(core, "read_json", lambda path: "Echo {input}")
    monkeypatch.setattr(core, "read_csv", lambda **kwargs: None)

    core.apply_prompt_to_csv(
        input_csv=tmp_path / "input.csv",
        prompt_json=tmp_path / "prompt.json",
        columns=["text"],
        output_csv=tmp_path / "output.csv",
        model="qwen3:14b",
        no_think=True,
    )

    assert llm_kwargs == {"model": "qwen3:14b", "reasoning_effort": "none"}


def test_apply_prompt_to_csv_passes_multiple_columns_in_order(monkeypatch, tmp_path):
    """Verify several input columns reach the CSV reader in the given order.

    :param monkeypatch: Pytest monkeypatch fixture.
    :param tmp_path: Pytest-provided temporary directory.
    """
    captured = {}

    monkeypatch.setattr(core, "ChatOpenAI", lambda **kwargs: None)
    monkeypatch.setattr(core, "read_json", lambda path: "Echo {input}")
    monkeypatch.setattr(core, "read_csv", lambda **kwargs: captured.update(kwargs))

    core.apply_prompt_to_csv(
        input_csv=tmp_path / "input.csv",
        prompt_json=tmp_path / "prompt.json",
        columns=["ReportText", "ImpressionText"],
        output_csv=tmp_path / "output.csv",
        model="test-model",
    )

    assert captured["input_columns"] == ["ReportText", "ImpressionText"]
