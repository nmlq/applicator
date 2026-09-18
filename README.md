# Applicator

Applicator is a small Python CLI that applies an LLM prompt to one CSV column,
one row at a time, and writes the result into a new CSV column.

## Requirements

- Python 3.9 or newer
- An OpenAI-compatible model provider
- An API key, unless the configured provider does not require one

Runtime dependencies are listed in `requirements.txt`. The legacy
`setup.py` reads that file when installing the package.

## Installation

Create and activate a virtual environment, then install Applicator:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

Install pytest separately when running the test suite:

```bash
pip install pytest
```

### Install a released shiv package

Each tag matching `vX.Y.Z` creates a GitHub release containing a standalone
executable. Download the asset for the release you want, install it on your
`PATH`, and make it executable:

```bash
VERSION="vX.Y.Z"
mkdir -p "$HOME/.local/bin"
curl --fail --location \
  --output "$HOME/.local/bin/applicator" \
  "https://github.com/nmlq/applicator/releases/download/${VERSION}/applicator-${VERSION}"
chmod +x "$HOME/.local/bin/applicator"
```

The shiv package bootstraps its Python dependencies on first run, so after
installing it you can invoke the CLI directly:

```bash
applicator --help
```

To publish a release, push a semantic version tag such as `v1.2.3`. Tags must
contain exactly three numeric components; prerelease and build suffixes are not
published by the release workflow.

For the default OpenAI client configuration, set the API key in the
environment:

```bash
export OPENAI_API_KEY="..."
```

## Prompt format

Prompt files are JSON objects containing a non-empty `prompt` string. The
prompt must include the `{input}` placeholder:

```json
{
  "prompt": "Summarize this text:\n\n{input}"
}
```

The placeholder is replaced with the value from the selected CSV column for
each row.

## Command-line usage

The CLI exposes a `run` subcommand:

```bash
applicator run INPUT_CSV PROMPT_JSON \
  --column COLUMN \
  --output OUTPUT_CSV
```

Example:

```bash
applicator run input.csv prompt.json \
  --column text \
  --output output.csv \
  --model gpt-4.1-mini
```

### Options

| Option | Short form | Required | Default | Description |
| --- | --- | --- | --- | --- |
| `INPUT_CSV` | — | yes | — | Readable source CSV file |
| `PROMPT_JSON` | — | yes | — | Readable prompt JSON file |
| `--column` | `-c` | yes | — | Input CSV column to process |
| `--output` | `-o` | yes | — | Destination CSV file |
| `--model` | `-m` | no | `gpt-4.1-mini` | Model identifier |
| `--base-url` | — | no | provider default | OpenAI-compatible API base URL |
| `--api-key` | — | no | provider environment | API key |
| `--output-column` | — | no | `applicator_output` | Name of the generated column |

For an OpenAI-compatible local endpoint:

```bash
applicator run input.csv prompt.json \
  --column text \
  --output output.csv \
  --model your-model-name \
  --base-url http://localhost:11434/v1 \
  --api-key dummy
```

### Ollama

Ollama exposes an OpenAI-compatible API when its local server is running.
Install Ollama separately, start the server, and pull a model:

```bash
ollama serve
ollama pull llama3.2
```

Configure the endpoint and model with environment variables:

```bash
export OLLAMA_BASE_URL="http://localhost:11434/v1"
export OLLAMA_MODEL="llama3.2"
export OLLAMA_API_KEY="ollama"
```

Then run Applicator against Ollama:

```bash
applicator run input.csv prompt.json \
  --column text \
  --output output.csv \
  --model "$OLLAMA_MODEL" \
  --base-url "$OLLAMA_BASE_URL" \
  --api-key "$OLLAMA_API_KEY"
```

The local Ollama API does not normally require authentication. Applicator
still accepts `--api-key` because `langchain-openai` expects an API key
parameter; a placeholder such as `ollama` is sufficient for a local server.
If Ollama is configured on another host, replace `localhost` with that host's
address and ensure the server accepts connections from the client.

The CLI validates that the input CSV and prompt JSON files exist and are
readable before processing.

## Environment variables

The underlying OpenAI-compatible client reads `OPENAI_API_KEY` when
`--api-key` is not provided. You can also keep the model and endpoint in shell
variables and pass them to the CLI:

```bash
export OPENAI_API_KEY="example-key-not-real"
export APPLICATOR_MODEL="example-chat-model"
export APPLICATOR_BASE_URL="https://api.example.invalid/v1"

applicator run input.csv prompt.json \
  --column text \
  --output output.csv \
  --model "$APPLICATOR_MODEL" \
  --base-url "$APPLICATOR_BASE_URL"
```

The values above use the reserved `.invalid` domain and are placeholders only;
they will not connect to a real service. Never commit real API keys to source
control. For a real provider, replace the key and endpoint with the values
documented by that provider.

## Example use case: support-ticket triage

Suppose `support_tickets.csv` contains customer tickets:

```csv
ticket_id,subject,body
1001,Unable to export report,"The export button returns an error after I select CSV."
1002,Question about billing,"Can I change the billing date for my subscription?"
```

Create `triage_prompt.json`:

```json
{
  "prompt": "Classify this support ticket with a short category and priority. Return only one line.\n\n{input}"
}
```

Set the connection details through environment variables. This example uses a
fictional endpoint and model:

```bash
export OPENAI_API_KEY="support-demo-key-not-real"
export SUPPORT_MODEL="support-classifier-demo"
export SUPPORT_BASE_URL="https://llm.support-demo.example.invalid/v1"
```

Run Applicator against the ticket body:

```bash
applicator run support_tickets.csv triage_prompt.json \
  --column body \
  --output triaged_tickets.csv \
  --output-column triage \
  --model "$SUPPORT_MODEL" \
  --base-url "$SUPPORT_BASE_URL"
```

The generated `triaged_tickets.csv` keeps the original columns and adds a
`triage` column containing one model response per ticket. Because the endpoint
uses `.invalid`, this command is a documentation example and is not expected
to produce live results.

## Processing behavior

- Reads CSV rows with Python's streaming `csv` module instead of loading the
  entire file into memory.
- Makes one LLM invocation per input row.
- Preserves all original CSV columns.
- Adds the generated response to `applicator_output` by default.
- Writes and flushes each completed row immediately.
- Prints progress for each processed row.
- Raises an error when the input CSV has no header or the requested column is
  missing.
- Does not implement batching, concurrency, agents, or MCP.

## Tests

Run the test suite from the project root:

```bash
python3 -m pytest -q
```

The tests mock the LLM client, so they do not make network requests or require
an API key. The suite covers:

- JSON prompt validation
- CSV transformation and validation
- LLM configuration and response conversion
- CLI parsing, validation, and dispatch
- End-to-end reader behavior using the example fixtures

## Test fixtures

Reusable example inputs are stored in `tests/fixtures/`:

- `tests/fixtures/input.example.csv` contains `id` and `text` columns with two
  sample rows.
- `tests/fixtures/prompt.example.json` contains a valid prompt using
  `{input}`.

The fixture paths are exposed as the `example_input_csv` and
`example_prompt_json` pytest fixtures in `tests/conftest.py`.

## Project layout

```text
applicator/
├── applicator/
│   ├── cli.py       # argparse command-line interface
│   ├── core.py      # LLM-backed CSV processing
│   └── readers.py   # JSON prompt and CSV readers
├── tests/
│   ├── fixtures/    # Reusable CSV and JSON test inputs
│   ├── conftest.py  # Shared pytest fixtures and test stubs
│   ├── test_cli.py
│   ├── test_core.py
│   └── test_readers.py
├── requirements.txt # Runtime dependencies
└── setup.py         # Package metadata and console entry point
```
