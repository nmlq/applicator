from pathlib import Path
from langchain_openai import ChatOpenAI

from .readers import read_csv, read_json


def apply_prompt_to_csv(
    input_csv: Path,
    prompt_json: Path,
    column: str,
    output_csv: Path,
    model: str,
    base_url: str | None = None,
    api_key: str | None = None,
    output_column: str = "applicator_output",
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
    """
    # Validate the prompt before creating the client or opening the output file.
    prompt_template = read_json(prompt_json)

    kwargs = {"model": model}
    if base_url:
        kwargs["base_url"] = base_url
    if api_key:
        kwargs["api_key"] = api_key

    # Optional connection settings are omitted so the client can use its defaults.
    llm = ChatOpenAI(**kwargs)

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

    read_csv(
        input_csv=input_csv,
        output_csv=output_csv,
        input_column=column,
        output_column=output_column,
        transform=apply,
    )
