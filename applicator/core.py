import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional

from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI

from .readers import read_csv, read_json

logger = logging.getLogger(__name__)


@dataclass
class LocalModelConfig:
    """Settings for running a Hugging Face model in-process.

    :param model: Hugging Face model id or local directory with model weights.
    :param device: ``device_map`` passed to ``transformers``, such as ``auto``,
        ``cuda``, ``cuda:1``, ``mps`` or ``cpu``.
    :param dtype: Weight dtype, such as ``auto``, ``float16`` or ``bfloat16``.
    :param max_new_tokens: Maximum number of tokens generated per row.
    :param temperature: Sampling temperature; ``None`` or ``0`` means greedy.
    :param top_p: Nucleus sampling probability mass.
    :param top_k: Number of highest-probability tokens considered when sampling.
    :param repetition_penalty: Penalty applied to repeated tokens.
    :param trust_remote_code: Allow model repositories to run custom code.
    :param no_think: Disable the thinking phase of reasoning models such as
        Qwen3; templates without a thinking switch ignore it.
    :param load_in_4bit: Quantize the weights to 4-bit with ``bitsandbytes``
        while loading; ``dtype`` then sets the compute dtype.
    """

    model: str
    device: str = "auto"
    dtype: str = "auto"
    max_new_tokens: int = 512
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    repetition_penalty: Optional[float] = None
    trust_remote_code: bool = False
    no_think: bool = False
    load_in_4bit: bool = False


class LocalChatModel:
    """Chat wrapper that applies the model's chat template before generating.

    ``ChatHuggingFace`` cannot forward chat template arguments such as
    ``enable_thinking``, so this applies the template itself.

    :param pipeline: ``HuggingFacePipeline`` configured for text generation.
    :param chat_template_kwargs: Extra arguments for ``apply_chat_template``.
    """

    def __init__(self, pipeline, chat_template_kwargs: Optional[dict] = None):
        self.pipeline = pipeline
        self.tokenizer = pipeline.pipeline.tokenizer
        self.chat_template_kwargs = chat_template_kwargs or {}

    def invoke(self, prompt: str) -> AIMessage:
        """Generate a reply to one user message.

        :param prompt: User message text.
        :return: Message containing only the generated reply.
        """
        return self.batch([prompt])[0]

    def batch(self, prompts: List[str]) -> List[AIMessage]:
        """Generate replies to several user messages in one pass on the device.

        :param prompts: User message texts.
        :return: One message per prompt, in the same order.
        """
        texts = [
            self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
                **self.chat_template_kwargs,
            )
            for prompt in prompts
        ]
        # HuggingFacePipeline.batch sends the prompts to the model as padded batches.
        return [AIMessage(content=text) for text in self.pipeline.batch(texts)]


def build_openai_llm(
    model: str,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    no_think: bool = False,
) -> ChatOpenAI:
    """Build a client for an OpenAI-compatible API.

    :param model: Model identifier passed to ``ChatOpenAI``.
    :param base_url: Optional OpenAI-compatible API base URL.
    :param api_key: Optional API key for the model provider.
    :param no_think: Ask the provider to skip reasoning with
        ``reasoning_effort="none"``.
    :return: Chat model whose ``invoke`` calls the remote API.
    """
    kwargs = {"model": model}
    if base_url:
        kwargs["base_url"] = base_url
    if api_key:
        kwargs["api_key"] = api_key
    if no_think:
        # Ollama's OpenAI endpoint ignores its native think=false here, and a
        # /no_think prompt prefix, but honors reasoning_effort="none".
        kwargs["reasoning_effort"] = "none"

    # Optional connection settings are omitted so the client can use its defaults.
    logger.info("Initializing model %s%s", model, f" at {base_url}" if base_url else "")
    return ChatOpenAI(**kwargs)


def build_4bit_config(dtype: str):
    """Build a ``bitsandbytes`` config that quantizes weights to 4-bit NF4.

    :param dtype: Compute dtype name; ``auto`` keeps the ``bitsandbytes`` default.
    :return: ``BitsAndBytesConfig`` for ``from_pretrained``.
    :raises RuntimeError: If ``bitsandbytes`` is not installed.
    """
    try:
        import bitsandbytes  # noqa: F401
    except ImportError as error:
        raise RuntimeError(
            "--load-in-4bit needs bitsandbytes. Install it with: pip install bitsandbytes"
        ) from error
    import torch
    from transformers import BitsAndBytesConfig

    # Weights are stored in 4-bit but dequantized to this dtype for each matmul.
    compute_dtype = None if dtype == "auto" else getattr(torch, dtype)
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )


def build_local_llm(config: LocalModelConfig, batch_size: int = 1):
    """Load a Hugging Face model into this process for local generation.

    :param config: Local model and generation settings.
    :param batch_size: Number of prompts generated together on the device.
    :return: Chat model whose ``invoke`` runs on the local device.
    :raises RuntimeError: If the optional local-model dependencies are missing.
    """
    # Imported lazily: torch and transformers are large, optional dependencies.
    try:
        import torch
        from langchain_huggingface import HuggingFacePipeline
    except ImportError as error:
        raise RuntimeError(
            "Local models need the optional dependencies. "
            "Install them with: pip install 'applicator[local]'"
        ) from error

    # Greedy decoding unless a positive temperature asks for sampling.
    generation = {
        "max_new_tokens": config.max_new_tokens,
        "do_sample": bool(config.temperature),
        # Return only the completion, not the prompt followed by the completion.
        "return_full_text": False,
    }
    if config.temperature:
        generation["temperature"] = config.temperature
        if config.top_p is not None:
            generation["top_p"] = config.top_p
        if config.top_k is not None:
            generation["top_k"] = config.top_k
    if config.repetition_penalty is not None:
        generation["repetition_penalty"] = config.repetition_penalty

    model_kwargs = {"dtype": config.dtype}
    if config.trust_remote_code:
        model_kwargs["trust_remote_code"] = True
    if config.load_in_4bit:
        model_kwargs["quantization_config"] = build_4bit_config(config.dtype)

    logger.info(
        "Loading local model %s (device=%s, dtype=%s%s)",
        config.model,
        config.device,
        config.dtype,
        ", 4-bit" if config.load_in_4bit else "",
    )
    # device_map rather than device: device only accepts CUDA indexes, not mps.
    pipeline = HuggingFacePipeline.from_model_id(
        model_id=config.model,
        task="text-generation",
        device_map=config.device,
        model_kwargs=model_kwargs,
        pipeline_kwargs=generation,
        batch_size=batch_size,
    )
    # Batched generation must pad on the left so every prompt ends where
    # generation starts; right padding corrupts the shorter prompts' output.
    pipeline.pipeline.tokenizer.padding_side = "left"
    model = pipeline.pipeline.model
    logger.info("Model loaded on %s", model.device)
    gpu_available = torch.cuda.is_available() or torch.backends.mps.is_available()
    if model.device.type == "cpu" and gpu_available:
        logger.warning("Model is on CPU although a GPU is available; pass --device to select it")

    # Qwen3-style templates pre-fill an empty think block when this is False.
    chat_template_kwargs = {"enable_thinking": False} if config.no_think else {}
    return LocalChatModel(pipeline, chat_template_kwargs)


def make_transform(llm, prompt_template: str) -> Callable[[List[str]], List[str]]:
    """Build the batch transform that sends CSV values to a language model.

    :param llm: Chat model with a ``batch`` method, local or API-backed.
    :param prompt_template: Prompt containing the ``{input}`` placeholder.
    :return: Function mapping CSV values to model replies, in the same order.
    """

    def apply(values: List[str]) -> List[str]:
        """Transform a batch of CSV values by invoking the configured language model.

        :param values: CSV values to insert into the prompt template.
        :return: Text content returned by the language model, one per value.
        """
        # Formatting happens per row so each request receives the current value.
        prompts = [prompt_template.format(input=value) for value in values]
        responses = llm.batch(prompts)
        # Normalize provider-specific response content into text for the CSV writer.
        return [
            response.content if isinstance(response.content, str) else str(response.content)
            for response in responses
        ]

    return apply


def apply_prompt_to_csv(
    input_csv: Path,
    prompt_json: Path,
    column: str,
    output_csv: Path,
    model: str,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    output_column: str = "applicator_output",
    local_model: Optional[LocalModelConfig] = None,
    batch_size: int = 1,
    no_think: bool = False,
) -> None:
    """Apply a prompt-driven LLM transformation to a CSV column.

    :param input_csv: Source CSV file.
    :param prompt_json: JSON file containing the prompt template.
    :param column: CSV column whose values are sent to the model.
    :param output_csv: Destination CSV file.
    :param model: Model identifier passed to ``ChatOpenAI``.
    :param base_url: Optional OpenAI-compatible API base URL.
    :param api_key: Optional API key for the model provider.
    :param output_column: Destination column for model responses.
    :param local_model: When set, run this Hugging Face model locally instead
        of calling an API; ``model``, ``base_url`` and ``api_key`` are unused.
    :param batch_size: Rows processed together: generated as one batch on a
        local model, or sent as concurrent requests to an API.
    :param no_think: Disable reasoning on the API path; a local model uses
        ``LocalModelConfig.no_think`` instead.
    """
    # Validate the prompt before creating the client or opening the output file.
    logger.info("Loading prompt from %s", prompt_json)
    prompt_template = read_json(prompt_json)

    # The model is created once and reused for every row.
    if local_model:
        llm = build_local_llm(local_model, batch_size=batch_size)
    else:
        llm = build_openai_llm(model, base_url=base_url, api_key=api_key, no_think=no_think)

    logger.info("Processing column '%s' from %s (batch size %d)", column, input_csv, batch_size)
    read_csv(
        input_csv=input_csv,
        output_csv=output_csv,
        input_column=column,
        output_column=output_column,
        transform=make_transform(llm, prompt_template),
        batch_size=batch_size,
    )
    logger.info("Done. Results written to %s", output_csv)
