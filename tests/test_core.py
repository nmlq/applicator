import sys
from types import ModuleType, SimpleNamespace

import pytest

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


def test_apply_prompt_to_csv_uses_local_model_when_configured(monkeypatch, tmp_path):
    """Verify a local model config bypasses the API client.

    :param monkeypatch: Pytest monkeypatch fixture.
    :param tmp_path: Pytest-provided temporary directory.
    """
    built = []

    class FakeLocalLLM:
        def invoke(self, prompt):
            """Return predictable content for a local generation."""
            return SimpleNamespace(content=f"local: {prompt}")

    def fail_openai(**kwargs):
        """Fail if the API client is created for a local run."""
        raise AssertionError("ChatOpenAI should not be created")

    captured = {}

    def fake_read_csv(**kwargs):
        """Exercise the transform with one value."""
        captured["result"] = kwargs["transform"]("Ada")

    def fake_build_local_llm(config):
        """Record the config and return a fake local model."""
        built.append(config)
        return FakeLocalLLM()

    monkeypatch.setattr(core, "ChatOpenAI", fail_openai)
    monkeypatch.setattr(core, "build_local_llm", fake_build_local_llm)
    monkeypatch.setattr(core, "read_json", lambda path: "Summarize {input}")
    monkeypatch.setattr(core, "read_csv", fake_read_csv)

    config = core.LocalModelConfig(model="org/model")
    core.apply_prompt_to_csv(
        input_csv=tmp_path / "input.csv",
        prompt_json=tmp_path / "prompt.json",
        column="name",
        output_csv=tmp_path / "output.csv",
        model="unused",
        local_model=config,
    )

    assert built == [config]
    assert captured["result"] == "local: Summarize Ada"


@pytest.fixture
def fake_huggingface(monkeypatch):
    """Install fake ``torch`` and ``langchain_huggingface`` modules.

    :param monkeypatch: Pytest monkeypatch fixture.
    :return: Dict recording pipeline and chat model construction arguments.
    """
    recorded = {}

    class FakeTokenizer:
        def apply_chat_template(self, messages, **kwargs):
            """Record template arguments and render a predictable prompt."""
            recorded["template"] = kwargs
            return f"<chat>{messages[0]['content']}"

    tokenizer = FakeTokenizer()

    class FakeHuggingFacePipeline:
        @classmethod
        def from_model_id(cls, **kwargs):
            """Record pipeline arguments and return a pipeline on CPU."""
            recorded["pipeline"] = kwargs
            instance = cls()
            model = SimpleNamespace(device=SimpleNamespace(type="cpu"))
            instance.pipeline = SimpleNamespace(model=model, tokenizer=tokenizer)
            return instance

        def invoke(self, text):
            """Return a completion echoing the rendered prompt."""
            return f"reply to {text}"

    langchain_huggingface = ModuleType("langchain_huggingface")
    langchain_huggingface.HuggingFacePipeline = FakeHuggingFacePipeline
    torch = ModuleType("torch")
    torch.cuda = SimpleNamespace(is_available=lambda: False)
    torch.backends = SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False))

    monkeypatch.setitem(sys.modules, "langchain_huggingface", langchain_huggingface)
    monkeypatch.setitem(sys.modules, "torch", torch)
    return recorded


def test_build_local_llm_defaults_to_greedy_decoding(fake_huggingface):
    """Verify default settings load the model for greedy generation.

    :param fake_huggingface: Recorded fake Hugging Face arguments.
    """
    core.build_local_llm(core.LocalModelConfig(model="org/model"))

    pipeline = fake_huggingface["pipeline"]
    assert pipeline["model_id"] == "org/model"
    assert pipeline["task"] == "text-generation"
    assert pipeline["device_map"] == "auto"
    assert pipeline["model_kwargs"] == {"dtype": "auto"}
    assert pipeline["pipeline_kwargs"] == {
        "max_new_tokens": 512,
        "do_sample": False,
        "return_full_text": False,
    }


def test_build_local_llm_applies_chat_template(fake_huggingface):
    """Verify prompts are wrapped in the chat template before generation.

    :param fake_huggingface: Recorded fake Hugging Face arguments.
    """
    llm = core.build_local_llm(core.LocalModelConfig(model="org/model"))

    response = llm.invoke("Summarize Ada")

    assert response.content == "reply to <chat>Summarize Ada"
    assert fake_huggingface["template"] == {"tokenize": False, "add_generation_prompt": True}


def test_build_local_llm_disables_thinking(fake_huggingface):
    """Verify ``no_think`` passes ``enable_thinking=False`` to the chat template.

    :param fake_huggingface: Recorded fake Hugging Face arguments.
    """
    llm = core.build_local_llm(core.LocalModelConfig(model="org/model", no_think=True))

    llm.invoke("Summarize Ada")

    assert fake_huggingface["template"]["enable_thinking"] is False


def test_build_local_llm_passes_sampling_settings(fake_huggingface):
    """Verify sampling settings are forwarded when temperature is positive.

    :param fake_huggingface: Recorded fake Hugging Face arguments.
    """
    core.build_local_llm(
        core.LocalModelConfig(
            model="org/model",
            device="cuda:1",
            dtype="bfloat16",
            max_new_tokens=32,
            temperature=0.7,
            top_p=0.9,
            top_k=40,
            repetition_penalty=1.1,
            trust_remote_code=True,
        )
    )

    pipeline = fake_huggingface["pipeline"]
    assert pipeline["device_map"] == "cuda:1"
    assert pipeline["model_kwargs"] == {"dtype": "bfloat16", "trust_remote_code": True}
    assert pipeline["pipeline_kwargs"] == {
        "max_new_tokens": 32,
        "do_sample": True,
        "return_full_text": False,
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 40,
        "repetition_penalty": 1.1,
    }


def test_build_local_llm_ignores_sampling_settings_when_greedy(fake_huggingface):
    """Verify top-p and top-k are dropped when decoding greedily.

    :param fake_huggingface: Recorded fake Hugging Face arguments.
    """
    core.build_local_llm(core.LocalModelConfig(model="org/model", top_p=0.9, top_k=40))

    pipeline_kwargs = fake_huggingface["pipeline"]["pipeline_kwargs"]
    assert "top_p" not in pipeline_kwargs
    assert "top_k" not in pipeline_kwargs


def test_build_local_llm_explains_missing_dependencies(monkeypatch):
    """Verify a missing optional dependency produces an install hint.

    :param monkeypatch: Pytest monkeypatch fixture.
    """
    # A None entry in sys.modules makes the import raise ImportError.
    monkeypatch.setitem(sys.modules, "langchain_huggingface", None)

    with pytest.raises(RuntimeError, match=r"applicator\[local\]"):
        core.build_local_llm(core.LocalModelConfig(model="org/model"))


def test_build_local_llm_quantizes_to_4bit(fake_huggingface, monkeypatch):
    """Verify ``load_in_4bit`` passes an NF4 quantization config to the model.

    :param fake_huggingface: Recorded fake Hugging Face arguments.
    :param monkeypatch: Pytest monkeypatch fixture.
    """
    transformers = ModuleType("transformers")
    transformers.BitsAndBytesConfig = lambda **kwargs: kwargs
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setitem(sys.modules, "bitsandbytes", ModuleType("bitsandbytes"))
    sys.modules["torch"].float32 = "torch.float32"

    core.build_local_llm(
        core.LocalModelConfig(model="org/model", dtype="float32", load_in_4bit=True)
    )

    model_kwargs = fake_huggingface["pipeline"]["model_kwargs"]
    assert model_kwargs["dtype"] == "float32"
    assert model_kwargs["quantization_config"] == {
        "load_in_4bit": True,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_use_double_quant": True,
        "bnb_4bit_compute_dtype": "torch.float32",
    }


def test_build_4bit_config_keeps_default_compute_dtype_for_auto(fake_huggingface, monkeypatch):
    """Verify ``dtype=auto`` leaves the compute dtype to ``bitsandbytes``.

    :param fake_huggingface: Recorded fake Hugging Face arguments.
    :param monkeypatch: Pytest monkeypatch fixture.
    """
    transformers = ModuleType("transformers")
    transformers.BitsAndBytesConfig = lambda **kwargs: kwargs
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setitem(sys.modules, "bitsandbytes", ModuleType("bitsandbytes"))

    assert core.build_4bit_config("auto")["bnb_4bit_compute_dtype"] is None


def test_build_4bit_config_explains_missing_bitsandbytes(monkeypatch):
    """Verify a missing ``bitsandbytes`` produces an install hint.

    :param monkeypatch: Pytest monkeypatch fixture.
    """
    monkeypatch.setitem(sys.modules, "bitsandbytes", None)

    with pytest.raises(RuntimeError, match="pip install bitsandbytes"):
        core.build_4bit_config("float32")
