# Applicator

## A lightweight CSV-to-LLM automation tool

---

## 1. What is Applicator?

Applicator is a command-line tool that applies an LLM prompt to one column of
a CSV file, one row at a time.

It:

- Reads structured CSV input
- Inserts each selected value into a prompt template
- Sends one request per row to an OpenAI-compatible model endpoint
- Writes the response into a new CSV column
- Preserves the original CSV data

Applicator is intentionally small, predictable, and easy to integrate into
scripts and data-processing workflows.

---

## 2. Why was Applicator made?

Many useful automation tasks have the same basic shape:

1. Start with tabular data
2. Select one text field
3. Ask a model to transform or classify the text
4. Save the result for review or downstream processing

Without a focused tool, users often need to write custom code for:

- CSV parsing
- Prompt formatting
- Model configuration
- Output file handling
- Progress reporting
- Error validation

Applicator packages that repeated workflow into one small CLI command.

---

## 3. Design goals

### Simple

The primary workflow is one command with a CSV file, a prompt file, and a
column name.

### Streaming

Rows are processed one at a time instead of loading the entire CSV into
memory.

### Provider-flexible

Applicator works with OpenAI-compatible hosted services, local servers, and
tools such as Ollama.

### Resumable output

Each completed row is written and flushed immediately, reducing the amount of
completed work lost if processing stops.

### Testable

The LLM client is isolated behind the core processing function, allowing the
test suite to run without network requests or API credentials.

---

## 4. How Applicator works

```text
Input CSV
    |
    v
Select input column
    |
    v
Load JSON prompt template
    |
    v
Replace {input} for each row
    |
    v
Call OpenAI-compatible LLM endpoint
    |
    v
Write response to output column
    |
    v
Output CSV
```

Example transformation:

```text
CSV value:
The export button returns an error.

Prompt template:
Classify this support ticket:

{input}

Model request:
Classify this support ticket:

The export button returns an error.

CSV output:
Original columns + generated model response
```

---

## 5. Prompt format

Prompt files are JSON documents with a non-empty `prompt` field.

```json
{
  "prompt": "Summarize this text:\n\n{input}"
}
```

The `{input}` placeholder is required. Applicator replaces it with the value
from the selected CSV column.

This keeps prompts versionable, reviewable, and separate from shell commands.

---

## 6. Installation

Applicator requires Python 3.9 or newer.

Create a virtual environment and install the package:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

Runtime dependencies are listed in `requirements.txt`. The project’s
`setup.py` reads that file when installing the package.

---

## 7. Basic command-line usage

```bash
applicator run INPUT_CSV PROMPT_JSON \
  --column COLUMN \
  --output OUTPUT_CSV
```

Example:

```bash
applicator run reviews.csv review_prompt.json \
  --column review_text \
  --output reviewed.csv \
  --model gpt-4.1-mini
```

Important options:

| Option | Description |
| --- | --- |
| `--column`, `-c` | CSV column sent to the prompt |
| `--output`, `-o` | Output CSV path |
| `--model`, `-m` | Model identifier |
| `--base-url` | OpenAI-compatible API endpoint |
| `--api-key` | Provider API key |
| `--output-column` | Name of the generated output column |

---

## 8. Environment variables

The default OpenAI-compatible client can read the API key from
`OPENAI_API_KEY`:

```bash
export OPENAI_API_KEY="your-api-key"
```

Model and endpoint settings can also be stored in shell variables:

```bash
export APPLICATOR_MODEL="gpt-4.1-mini"
export APPLICATOR_BASE_URL="https://api.example.invalid/v1"

applicator run input.csv prompt.json \
  --column text \
  --output output.csv \
  --model "$APPLICATOR_MODEL" \
  --base-url "$APPLICATOR_BASE_URL"
```

The `.invalid` endpoint above is fictional and is included only as a safe
documentation example.

Never commit real API keys to source control.

---

## 9. Ollama support

Ollama provides a local OpenAI-compatible endpoint.

Start Ollama and download a model:

```bash
ollama serve
ollama pull llama3.2
```

Configure the local endpoint:

```bash
export OLLAMA_BASE_URL="http://localhost:11434/v1"
export OLLAMA_MODEL="llama3.2"
export OLLAMA_API_KEY="ollama"
```

Run Applicator:

```bash
applicator run input.csv prompt.json \
  --column text \
  --output output.csv \
  --model "$OLLAMA_MODEL" \
  --base-url "$OLLAMA_BASE_URL" \
  --api-key "$OLLAMA_API_KEY"
```

Ollama normally does not require authentication. The placeholder API key is
accepted because the OpenAI-compatible client expects an API key parameter.

---

## 10. General use cases

Applicator can support many row-by-row text-processing tasks.

### Support-ticket triage

- Classify issue type
- Assign priority
- Suggest a routing team

### Customer feedback analysis

- Detect sentiment
- Extract product names
- Identify recurring themes

### Content operations

- Generate summaries
- Create tags
- Rewrite descriptions
- Check tone or style

### Data preparation

- Normalize free-text values
- Extract structured fields
- Detect sensitive information
- Convert text into labels for later analysis

### Research workflows

- Summarize papers or notes
- Extract entities
- Create short abstracts
- Generate review categories

The same CLI workflow can be reused by changing only the input column and
prompt file.

---

## 11. Example: support-ticket triage

Input file:

```csv
ticket_id,subject,body
1001,Unable to export report,"The export button returns an error after I select CSV."
1002,Question about billing,"Can I change the billing date for my subscription?"
```

Prompt file, `triage_prompt.json`:

```json
{
  "prompt": "Classify this support ticket with a short category and priority. Return only one line.\n\n{input}"
}
```

Command:

```bash
applicator run support_tickets.csv triage_prompt.json \
  --column body \
  --output triaged_tickets.csv \
  --output-column triage \
  --model "$OLLAMA_MODEL" \
  --base-url "$OLLAMA_BASE_URL" \
  --api-key "$OLLAMA_API_KEY"
```

The resulting file keeps `ticket_id`, `subject`, and `body`, and adds the
generated `triage` column.

---

## 12. Processing behavior

Applicator:

- Validates that input files exist and are readable
- Validates the prompt structure and `{input}` placeholder
- Validates that the requested CSV column exists
- Preserves original CSV columns
- Adds the output column if it does not already exist
- Processes exactly one row per model invocation
- Flushes each completed output row
- Prints progress after each processed row

Applicator does not currently provide batching, concurrency, agents, or MCP
features. Its narrow scope keeps the core workflow easy to understand.

---

## 13. Testing

Run the tests from the project root:

```bash
python3 -m pytest -q
```

The tests do not call a real model. They use mocked model behavior to verify:

- JSON prompt validation
- CSV reading and writing
- Missing headers and columns
- Model configuration
- Prompt formatting
- CLI parsing and dispatch
- Example fixture processing

The reusable test inputs are located at:

```text
tests/fixtures/input.example.csv
tests/fixtures/prompt.example.json
```

---

## 14. Project structure

```text
applicator/
├── applicator/
│   ├── cli.py       # argparse command-line interface
│   ├── core.py      # LLM-backed CSV processing
│   └── readers.py   # JSON prompt and CSV readers
├── tests/
│   ├── fixtures/    # Reusable CSV and JSON inputs
│   ├── conftest.py  # Shared pytest fixtures and test stubs
│   ├── test_cli.py
│   ├── test_core.py
│   └── test_readers.py
├── requirements.txt # Runtime dependencies
├── setup.py         # Package metadata and CLI entry point
├── README.md        # Project documentation
└── presentation.md  # Presentation outline
```

---

## 15. Key takeaway

Applicator turns a common LLM data-processing pattern into a repeatable
command:

```text
CSV column + prompt template + model endpoint = enriched CSV
```

It is small enough for a one-off task, structured enough for a repeatable
workflow, and flexible enough to run against hosted or local
OpenAI-compatible models.
