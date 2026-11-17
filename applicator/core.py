import logging
from pathlib import Path
from typing import Callable, List, Optional, Sequence

from langchain_openai import ChatOpenAI

from .readers import read_csv, read_json

logger = logging.getLogger(__name__)


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


def make_transform(llm, prompt_template: str) -> Callable[[List[str]], List[str]]:
    """Build the batch transform that sends CSV values to a language model.

    :param llm: Chat model with a ``batch`` method.
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
    columns: Sequence[str],
    output_csv: Path,
    model: str,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    output_column: str = "applicator_output",
    batch_size: int = 1,
    no_think: bool = False,
) -> None:
    """Apply a prompt-driven LLM transformation to one or more CSV columns.

    :param input_csv: Source CSV file.
    :param prompt_json: JSON file containing the prompt template.
    :param columns: CSV columns whose values are sent to the model; several
        columns are joined per row, in order, with a newline.
    :param output_csv: Destination CSV file.
    :param model: Model identifier passed to ``ChatOpenAI``.
    :param base_url: Optional OpenAI-compatible API base URL.
    :param api_key: Optional API key for the model provider.
    :param output_column: Destination column for model responses.
    :param batch_size: Rows sent to the API as concurrent requests.
    :param no_think: Ask the provider to skip reasoning with
        ``reasoning_effort="none"``.
    """
    # Validate the prompt before creating the client or opening the output file.
    logger.info("Loading prompt from %s", prompt_json)
    prompt_template = read_json(prompt_json)

    # The model is created once and reused for every row.
    llm = build_openai_llm(model, base_url=base_url, api_key=api_key, no_think=no_think)

    logger.info(
        "Processing column(s) %s from %s (batch size %d)",
        ", ".join(f"'{column}'" for column in columns),
        input_csv,
        batch_size,
    )
    read_csv(
        input_csv=input_csv,
        output_csv=output_csv,
        input_columns=columns,
        output_column=output_column,
        transform=make_transform(llm, prompt_template),
        batch_size=batch_size,
    )
    logger.info("Done. Results written to %s", output_csv)
