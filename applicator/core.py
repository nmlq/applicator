import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

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


def build_openai_llm(
    model: str,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
) -> ChatOpenAI:
    """Build a client for an OpenAI-compatible API.

    :param model: Model identifier passed to ``ChatOpenAI``.
    :param base_url: Optional OpenAI-compatible API base URL.
    :param api_key: Optional API key for the model provider.
    :return: Chat model whose ``invoke`` calls the remote API.
    """
    kwargs = {"model": model}
    if base_url:
        kwargs["base_url"] = base_url
    if api_key:
        kwargs["api_key"] = api_key

    # Optional connection settings are omitted so the client can use its defaults.
    logger.info("Initializing model %s%s", model, f" at {base_url}" if base_url else "")
    return ChatOpenAI(**kwargs)


def build_local_llm(config: LocalModelConfig):
    """Load a Hugging Face model into this process for local generation.

    :param config: Local model and generation settings.
    :return: Chat model whose ``invoke`` runs on the local device.
    :raises RuntimeError: If the optional local-model dependencies are missing.
    """
    # Imported lazily: torch and transformers are large, optional dependencies.
    try:
        import torch
        from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
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

    logger.info(
        "Loading local model %s (device=%s, dtype=%s)", config.model, config.device, config.dtype
    )
    # device_map rather than device: device only accepts CUDA indexes, not mps.
    pipeline = HuggingFacePipeline.from_model_id(
        model_id=config.model,
        task="text-generation",
        device_map=config.device,
        model_kwargs=model_kwargs,
        pipeline_kwargs=generation,
    )
    model = pipeline.pipeline.model
    logger.info("Model loaded on %s", model.device)
    gpu_available = torch.cuda.is_available() or torch.backends.mps.is_available()
    if model.device.type == "cpu" and gpu_available:
        logger.warning("Model is on CPU although a GPU is available; pass --device to select it")

    # Reuse the pipeline's tokenizer so the chat template isn't loaded twice.
    return ChatHuggingFace(llm=pipeline, tokenizer=pipeline.pipeline.tokenizer)


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
    """
    # Validate the prompt before creating the client or opening the output file.
    logger.info("Loading prompt from %s", prompt_json)
    prompt_template = read_json(prompt_json)

    # The model is created once and reused for every row.
    if local_model:
        llm = build_local_llm(local_model)
    else:
        llm = build_openai_llm(model, base_url=base_url, api_key=api_key)

    def apply(value: str) -> str:
        """Transform one CSV value by invoking the configured language model.

        :param value: CSV value to insert into the prompt template.
        :return: Text content returned by the language model.
        """
        # Formatting happens per row so each request receives the current value.
        prompt = prompt_template.format(input=value)
        response = llm.invoke(prompt)
        # Normalize provider-specific response content into text for the CSV writer.
        return response.content if isinstance(response.content, str) else str(response.content)

    logger.info("Processing column '%s' from %s", column, input_csv)
    read_csv(
        input_csv=input_csv,
        output_csv=output_csv,
        input_column=column,
        output_column=output_column,
        transform=apply,
    )
    logger.info("Done. Results written to %s", output_csv)
