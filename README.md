# Applicator

Applicator is a small Python CLI that applies an LLM prompt to every row of
a CSV, using the text of one or more columns, and writes each result into a
new CSV column. It calls an
OpenAI-compatible API endpoint, such as OpenAI or a local Ollama server, and
can send rows in concurrent batches.

## Requirements

- Python 3.9 or newer
- An OpenAI-compatible API endpoint, plus an API key unless the provider does
  not require one

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

Each tag matching `vX.Y.Z` creates a GitHub release containing standalone
executables. Because they bundle compiled dependencies (such as
`pydantic_core`), there is one asset per platform and Python version:

```
applicator-vX.Y.Z-<platform>-py<version>
```

where `<platform>` is `linux-x86_64`, `linux-aarch64`, `macos-arm64`
(Apple Silicon), `macos-x86_64` (Intel) or `windows-x86_64`, and `<version>`
is `311`, `312` or `313`. Windows assets end in `.pyz`. Each asset runs with the matching
`pythonX.Y` found on your `PATH`, so that interpreter must be installed. Using
an asset built for another platform or Python version fails at import time
with errors like `cannot import name 'ArgsKwargs' from
'pydantic_core._pydantic_core' (unknown location)`.

Download the matching asset, install it on your `PATH`, and make it
executable:

```bash
VERSION="vX.Y.Z"
PLATFORM="macos-arm64"   # or macos-x86_64, linux-x86_64, linux-aarch64
PY="312"                 # 311, 312 or 313; requires python3.12 on PATH
mkdir -p "$HOME/.local/bin"
curl --fail --location \
  --output "$HOME/.local/bin/applicator" \
  "https://github.com/nmlq/applicator/releases/download/${VERSION}/applicator-${VERSION}-${PLATFORM}-py${PY}"
chmod +x "$HOME/.local/bin/applicator"
```

On Windows, download `applicator-vX.Y.Z-windows-x86_64-py312.pyz` (or the
`py311`/`py313` asset) and run it through the Python launcher, which reads
the pinned version from the file:

```powershell
py applicator-vX.Y.Z-windows-x86_64-py312.pyz --help
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
each row. When several columns are selected, it is replaced with their values
joined by a newline (see [Multiple input columns](#multiple-input-columns)).

## Command-line usage

The CLI exposes a `run` subcommand:

```bash
applicator run INPUT_CSV PROMPT_JSON \
  --column COLUMN [COLUMN ...] \
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
| `--column` | `-c` | yes | — | Input CSV column(s) to process; see [Multiple input columns](#multiple-input-columns) |
| `--output` | `-o` | yes | — | Destination CSV file |
| `--model` | `-m` | no | `gpt-4.1-mini` | Model identifier |
| `--base-url` | — | no | provider default | OpenAI-compatible API base URL |
| `--api-key` | — | no | provider environment | API key |
| `--output-column` | — | no | `applicator_output` | Name of the generated column |
| `--batch-size` | — | no | `1` | Rows processed together; see [Batching](#batching) |
| `--limit` | — | no | all rows | Process at most the first N data rows; see [Limiting rows](#limiting-rows) |
| `--no-think` | — | no | off | Disable thinking in reasoning models; see [Reasoning models](#reasoning-models) |

### Multiple input columns

Pass more than one column name to `--column` to send several columns of each
row in one prompt. Applicator joins the row's values in the order given, with
a newline between them, and substitutes the result for `{input}`. Both forms
below are equivalent:

```bash
applicator run input.csv prompt.json \
  --column ReportText ImpressionText \
  --output output.csv

applicator run input.csv prompt.json \
  -c ReportText -c ImpressionText \
  --output output.csv
```

For a row where `ReportText` is `Brain MRI is normal.` and `ImpressionText` is
`No acute findings.`, the prompt receives:

```text
Brain MRI is normal.
No acute findings.
```

This matches the pandas expression
`df[['ReportText', 'ImpressionText']].agg('\n'.join, axis=1)`. The joined text
contains no column names or other labels; add any wording the model needs to
the prompt itself. An empty value still contributes its newline, so a row with
an empty `ImpressionText` ends with a trailing newline. Every named column must
exist in the CSV header, and the error lists all missing names.

The input columns are not written to the output CSV. For example, running with
`--column ReportText ImpressionText` produces an output without `ReportText`
or `ImpressionText`; every other column is kept unchanged, in its original
order, followed by the output column.

`--column COLUMN [COLUMN ...]` takes every following word until the next
option, so put the input CSV and prompt JSON before it, or follow it with
another option such as `--output`.

### Limiting rows

`--limit N` processes only the first N data rows of the input CSV, which is
useful for debugging a prompt or testing a model on a large file without
paying for every row. The header row is not counted, and N must be a positive
integer.

```bash
applicator run input.csv prompt.json \
  --column text \
  --output sample.csv \
  --limit 10
```

The output CSV contains only the processed rows. Reading stops once the limit
is reached, so the rest of the file is never loaded. A limit larger than the
number of rows processes the whole file. `--limit` works with
`--batch-size`; the last batch is shortened if needed so that no more than N
rows are sent.

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

For reasoning models such as `qwen3`, add `--no-think`. Thinking is otherwise
on, and it is slow even though the thinking text does not appear in the
output column (see [Reasoning models](#reasoning-models)).

`--batch-size N` sends N requests at once. Ollama only processes them in
parallel when it is started with `OLLAMA_NUM_PARALLEL` of at least N; each
parallel slot reserves its own context memory, set with
`OLLAMA_CONTEXT_LENGTH`. When Ollama runs as a systemd service, set these with
`sudo systemctl edit ollama`:

```ini
[Service]
Environment="OLLAMA_NUM_PARALLEL=4"
Environment="OLLAMA_CONTEXT_LENGTH=4096"
```

Then run `sudo systemctl daemon-reload && sudo systemctl restart ollama`.
Check with `ollama ps` that the model shows `100% GPU`; if the parallel slots
do not fit in GPU memory, Ollama moves part of the model to the CPU.

The CLI validates that the input CSV and prompt JSON files exist and are
readable before processing.

### Run several prompts with `run-local.sh`

`run-local.sh` runs every prompt file in a directory against one or more
columns of a CSV, using a local Ollama server. For each `*.json` file in
`PROMPTS_DIR` and each column in `COLUMNS`, it calls `applicator run` with
`--no-think` and writes the result next to the prompt as
`<prompt>_<column>_output.csv`:

```bash
./run-local.sh
```

With the defaults, `prompts/summary.json` and the columns `ReportText` and
`ImpressionText` produce `prompts/summary_ReportText_output.csv` and
`prompts/summary_ImpressionText_output.csv`. The script stops at the first
failed run, and exits with an error if `PROMPTS_DIR` contains no `.json`
files.

Override the defaults with environment variables:

| Variable | Default | Description |
| --- | --- | --- |
| `VENV_DIR` | `venv` | Virtual environment to activate before running |
| `INPUT_CSV` | `input.csv` | Source CSV file |
| `PROMPTS_DIR` | `prompts` | Directory of prompt JSON files |
| `COLUMNS` | `ReportText ImpressionText` | Space-separated input columns |
| `OLLAMA_BASE_URL` | `http://localhost:11434/v1` | Ollama API base URL |
| `OLLAMA_MODEL` | `gemma3n:e2b` | Ollama model to use |
| `OLLAMA_API_KEY` | `ollama` | Placeholder API key |
| `BATCH_SIZE` | `4` | Value passed to `--batch-size` |

For example, to use the `.venv` created in [Installation](#installation) and
a different model and column:

```bash
VENV_DIR=.venv OLLAMA_MODEL=qwen3:14b COLUMNS=text ./run-local.sh
```

`input.csv` and `prompts/` are listed in `.gitignore`, so local data, prompts
and their outputs are not committed.

## Reasoning models

Reasoning models such as Qwen3 think before answering by default. Thinking
can generate hundreds of tokens per row and makes each row much slower.
`--no-think` turns it off by sending `reasoning_effort="none"` with each
request.

Ollama returns thinking in a separate field that Applicator does not save, so
without `--no-think` the output column looks clean while each row still pays
for the thinking. Ollama 0.34 ignores both a `/no_think` prefix in the prompt
and its native `think: false` option on this endpoint;
`reasoning_effort="none"` is the setting it honors. Hosted providers may
reject `"none"` for models that do not support it.

In testing with Qwen3 on Ollama, `--no-think` was 4.6 times faster for eight
reports on Apple Silicon and 6.5 times faster for a single report on a Tesla
P40.

## Batching

`--batch-size N` sends the requests for N rows to the API concurrently. The
next batch starts when every request in the current one has returned.

```bash
applicator run input.csv prompt.json \
  --column text \
  --output output.csv \
  --model gpt-4.1-mini \
  --batch-size 8
```

Larger batches only help if the endpoint processes requests in parallel; for
Ollama, see [Ollama](#ollama). Hosted providers may rate-limit large batches.
Start at 4 and double it while the rows per second shown in the progress bar
keep improving.

Output is written and flushed after each batch, so an interrupted run loses
at most one batch of rows.

## Performance baselines

These measurements were taken with Ollama on an NVIDIA Tesla P40 (24 GB,
driver 560.35.05, CUDA 12.6) and an Apple Silicon Mac (Ollama 0.34.4). They
show relative differences between configurations; absolute times depend on
the GPU, the length of the inputs and answers, and the prompt. Answers were
one-sentence summaries.

Inputs:

- **Synthetic MRI reports**: the 64 fictional reports in
  `tests/fixtures/mri_reports.example.csv`, 8 to 153 words each.
- **Real MRI reports**: a private set of reports, not included in the
  repository. A sampled report was about 440 tokens.

### Qwen3-14B on the Tesla P40, real MRI reports

| Settings | `--batch-size` | Seconds per report | Rows measured |
| --- | --- | --- | --- |
| thinking on | 4 | 7.72 | 112 |
| thinking on (`/no_think` in the prompt, ignored) | 4 | 9.30 | 8 |
| `--no-think` | 4 | **1.15** | 16 |

Ollama ran with `OLLAMA_NUM_PARALLEL=4` and `OLLAMA_CONTEXT_LENGTH=4096`; the
model used 11.3 GB of GPU memory and ran `100% GPU`. The 8-row and 16-row
figures are early readings from longer runs; the 8-row figure may include
Ollama loading the model after a restart.

### One report on the Tesla P40, Ollama `qwen3:14b`

Timed with Ollama's native API, one request at a time:

| Thinking | Prompt tokens | Prompt time | Answer tokens | Answer time | Total |
| --- | --- | --- | --- | --- | --- |
| on | 442 | 1.1 s (407 tokens/s) | 458 | 18.7 s (24.5 tokens/s) | 20.2 s |
| off | 445 | 1.1 s (421 tokens/s) | 45 | 1.8 s (25.1 tokens/s) | **3.1 s** |

The P40 reads a report at about 400 tokens per second and writes the answer
at about 25 tokens per second, so the number of generated tokens dominates
the time. Hidden thinking added about 400 tokens per report.

### Apple Silicon Mac

Ollama `qwen3:1.7b` through Applicator, 8 synthetic MRI reports with
`--batch-size 4`: 13.6 seconds with thinking on and 2.95 seconds with
`--no-think`, including start-up.

### GPU clocks during a long run

During the Ollama runs the P40 was at 84 °C and 100% utilization, drawing
about 240 W of its 250 W limit. Its clock stayed at the 1531 MHz maximum and
`nvidia-smi -q -d PERFORMANCE` showed no thermal slowdown, only
`SW Power Cap: Active`. To check a GPU for throttling during a run:

```bash
nvidia-smi --query-gpu=temperature.gpu,clocks.sm,clocks.max.sm,power.draw --format=csv -l 2
```

### Takeaways

- **Turn thinking off** with `--no-think` for reasoning models. It was the
  largest single speedup: 6.5 times faster for one report on the P40.
- **Batch rows** with `--batch-size`, and start Ollama with
  `OLLAMA_NUM_PARALLEL` of at least the batch size so it processes the
  requests in parallel.

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

- Reads the CSV with pandas one batch at a time instead of loading the entire
  file into memory.
- Sends rows to the API in batches of `--batch-size` rows (default 1); the
  requests in a batch are sent concurrently.
- Stops after the first `--limit` rows when a limit is given.
- Preserves all original CSV columns.
- Adds the generated response to `applicator_output` by default.
- Writes and flushes each completed batch immediately.
- Prints progress for each processed row.
- Joins the values of multiple `--column` columns per row, in order, with a
  newline.
- Raises an error when the input CSV has no header or a requested column is
  missing.
- Does not implement agents or MCP.

## Tests

Run the test suite from the project root:

```bash
python3 -m pytest -q
```

The tests mock the LLM client, so they do not make network requests or
require an API key. The suite covers:

- JSON prompt validation
- CSV transformation and validation, including batched processing,
  joining multiple input columns, and row limits
- LLM configuration and response conversion
- The `--no-think` switch
- CLI parsing, validation, and dispatch, including both multiple-column forms
  and `--limit` validation
- End-to-end reader behavior using the example fixtures

## Test fixtures

Reusable example inputs are stored in `tests/fixtures/`:

- `tests/fixtures/input.example.csv` contains `id` and `text` columns with two
  sample rows.
- `tests/fixtures/prompt.example.json` contains a valid prompt using
  `{input}`.
- `tests/fixtures/mri_reports.example.csv` contains `id` and `report` columns
  with 64 synthetic MRI reports of 8 to 153 words. The reports are generated
  from fictional findings and contain no patient data.
- `tests/fixtures/mri_prompt.example.json` asks for a one-sentence summary of
  a report.

The fixture paths are exposed as the `example_input_csv` and
`example_prompt_json` pytest fixtures in `tests/conftest.py`.

## Project layout

```text
applicator/
├── .claude/
│   ├── CLAUDE.md    # Rules for Claude Code in this repository
│   └── skills/      # Project skills for Claude Code
├── applicator/
│   ├── cli.py       # argparse command-line interface
│   ├── core.py      # API client setup, CSV processing
│   └── readers.py   # JSON prompt and CSV readers
├── tests/
│   ├── fixtures/    # Reusable CSV and JSON test inputs
│   ├── conftest.py  # Shared pytest fixtures and test stubs
│   ├── test_cli.py
│   ├── test_core.py
│   └── test_readers.py
├── requirements.txt # Runtime dependencies
├── run-local.sh     # Run every prompt in prompts/ against Ollama
└── setup.py         # Package metadata and console entry point
```