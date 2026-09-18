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

        def invoke(self, prompt):
            """Return predictable content for a model request."""
            calls.append(prompt)
            if prompt.endswith("Ada"):
                return SimpleNamespace(content="Summary")
            return SimpleNamespace(content=["non-string", "content"])

    captured = {}

    def fake_read_csv(**kwargs):
        """Capture the CSV reader callback and exercise two values."""
        captured.update(kwargs)
        captured["first_result"] = kwargs["transform"]("Ada")
        captured["second_result"] = kwargs["transform"]("Grace")

    monkeypatch.setattr(core, "ChatOpenAI", FakeLLM)
    monkeypatch.setattr(core, "read_json", lambda path: "Summarize {input}")
    monkeypatch.setattr(core, "read_csv", fake_read_csv)

    core.apply_prompt_to_csv(
        input_csv=tmp_path / "input.csv",
        prompt_json=tmp_path / "prompt.json",
        column="name",
        output_csv=tmp_path / "output.csv",
        model="test-model",
        base_url="https://example.test/v1",
        api_key="test-key",
        output_column="summary",
    )

    assert llm_kwargs == {
        "model": "test-model",
        "base_url": "https://example.test/v1",
        "api_key": "test-key",
    }
    assert calls == ["Summarize Ada", "Summarize Grace"]
    assert captured["input_column"] == "name"
    assert captured["output_column"] == "summary"
    assert captured["first_result"] == "Summary"
    assert captured["second_result"] == "['non-string', 'content']"


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
        column="text",
        output_csv=tmp_path / "output.csv",
        model="test-model",
    )

    assert llm_kwargs == {"model": "test-model"}
