import sys
import types
from pathlib import Path

import pytest


if "langchain_openai" not in sys.modules:
    langchain_openai = types.ModuleType("langchain_openai")
    langchain_openai.ChatOpenAI = object
    sys.modules["langchain_openai"] = langchain_openai


FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def example_input_csv() -> Path:
    """Return the path to the example input CSV fixture.

    :return: Path to the reusable CSV test fixture.
    """
    return FIXTURES_DIR / "input.example.csv"


@pytest.fixture
def example_prompt_json() -> Path:
    """Return the path to the example prompt JSON fixture.

    :return: Path to the reusable JSON test fixture.
    """
    return FIXTURES_DIR / "prompt.example.json"
