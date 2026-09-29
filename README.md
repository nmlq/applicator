# Applicator

Applicator is a small Python CLI that applies an LLM prompt to one CSV column,
one row at a time, and writes the result into a new CSV column.

## Requirements

- Python 3.9 or newer
- One of:
  - An OpenAI-compatible model provider, plus an API key unless the provider
    does not require one
  - A Hugging Face model run locally (see [Local models](#local-models)),
    ideally on an NVIDIA or Apple Silicon GPU

Runtime dependencies are listed in `requirements.txt`. The legacy
`setup.py` reads that file when installing the package.

## Installation

Create and activate a virtual environment, then install Applicator:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

To run Hugging Face models locally, install the optional `local` extra. It
adds `langchain-huggingface`, `transformers`, `torch` and `accelerate`, which
are several gigabytes, so they are not part of the default install:

```bash
pip install -e '.[local]'
```

On Linux with an NVIDIA GPU, check that the installed `torch` build matches
your driver and GPU before running a model; see
[NVIDIA GPUs](#nvidia-gpus).

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

## Local models

Applicator can load a Hugging Face model into its own process and run it on
the local GPU instead of calling an API. The model is loaded once and reused
for every row, so there is no per-row network round trip. This requires the
`local` extra (see [Installation](#installation)).

Pass a Hugging Face model id, or a directory containing model weights, with
`--local-model`:

```bash
applicator run input.csv prompt.json \
  --column text \
  --output output.csv \
  --local-model Qwen/Qwen3-4B \
  --no-think \
  --max-new-tokens 256
```

The first run downloads the weights into `~/.cache/huggingface`; later runs
load them from disk. After the download, set `HF_HUB_OFFLINE=1` to guarantee
that no network requests are made.

The log reports where the model was placed. Check this line to confirm the
GPU is in use:

```text
INFO Model loaded on cuda:0
```

### Local model options

These options require `--local-model`. `--base-url` and `--api-key` cannot be
combined with `--local-model`, and `--model` is ignored.

| Option | Default | Description |
| --- | --- | --- |
| `--local-model` | — | Hugging Face model id or local model directory |
| `--device` | `auto` | Device map: `auto`, `cuda`, `cuda:N`, `mps` or `cpu` |
| `--dtype` | `auto` | Weight dtype: `auto`, `float16`, `bfloat16` or `float32` |
| `--max-new-tokens` | `512` | Maximum tokens generated per row |
| `--temperature` | greedy | Sampling temperature; omit or `0` for greedy decoding |
| `--top-p` | model default | Nucleus sampling probability (only with `--temperature`) |
| `--top-k` | model default | Top-k sampling cutoff (only with `--temperature`) |
| `--repetition-penalty` | model default | Penalty for repeated tokens, for example `1.1` |
| `--trust-remote-code` | off | Allow the model repository to run custom code |
| `--no-think` | off | Disable the thinking phase of reasoning models such as Qwen3 |

`--device auto` places the model on the best available device, but falls back
to the CPU silently when torch cannot use the GPU. Use `--device cuda` or
`--device mps` to fail with an error instead.

### Reasoning models

Reasoning models such as Qwen3 think before answering by default. The thinking
is written into the output column as a `<think>...</think>` block and counts
against `--max-new-tokens`, which makes each row much slower. `--no-think`
passes `enable_thinking=False` to the model's chat template, so the model
answers directly. Models whose chat template has no thinking switch ignore
the option.

### Choosing a model size

The weights must fit in GPU memory. Approximate size is the parameter count
multiplied by the bytes per weight: 4 bytes for `float32` and 2 for `float16`
or `bfloat16`. Leave a few gigabytes free for generation.

| Model | `float32` | `float16` / `bfloat16` |
| --- | --- | --- |
| `Qwen/Qwen3-4B` | ~16 GB | ~8 GB |
| `Qwen/Qwen3-8B` | ~32 GB | ~16 GB |
| `Qwen/Qwen3-14B` | ~56 GB | ~28 GB |

When a model does not fit, `--device auto` moves some layers to the CPU. The
run still completes, but slowly, and the log still reports the GPU device.
Choose a smaller model or a smaller dtype instead.

### Apple Silicon

`--device auto` uses the Metal (`mps`) backend and needs no extra setup. The
model shares unified memory with the rest of the system. Use `--dtype auto` or
`bfloat16`; loading with `--dtype float16` on `mps` has crashed during weight
loading in testing.

### NVIDIA GPUs

Each `torch` wheel is built for one CUDA version and a fixed list of GPU
architectures. The default `pip install torch` on Linux installs a recent CUDA
build, which needs a recent NVIDIA driver and no longer includes kernels for
older GPUs. Two checks tell you whether the installed build can use your GPU:

```bash
# Highest CUDA version the driver supports, shown top right
nvidia-smi

# torch build, whether CUDA works, and the architectures it was compiled for
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_arch_list())"
```

The CUDA version in the `torch` build suffix (for example `+cu126`) must not
be higher than the version `nvidia-smi` reports, and `is_available()` must
print `True`.

### Tesla P40

The Tesla P40 is a Pascal GPU (compute capability `sm_61`) with 24 GB of
memory. With driver 560 the highest supported CUDA version is 12.6. The
default Linux `torch` wheel does not work with it: it is built for CUDA 13
and its architecture list starts at `sm_75`, so `torch.cuda.is_available()`
returns `False` and the model silently runs on the CPU.

#### Install torch for the P40

Install a CUDA 12.6 build of torch. Remove the existing torch and NVIDIA
runtime packages first: installing over a CUDA 13 build leaves both sets of
`nvidia-*` libraries in the virtual environment, and torch then fails to
import with errors such as `undefined symbol: ncclCommResume`.

```bash
source .venv/bin/activate
pip install -e '.[local]'

# Remove torch and every NVIDIA runtime package
pip uninstall -y torch triton $(pip list --format=freeze | grep -iE '^nvidia-' | cut -d= -f1)

# Install the CUDA 12.6 build without reusing a cached wheel
pip install --no-cache-dir 'torch==2.14.0' --index-url https://download.pytorch.org/whl/cu126
```

To see which torch versions have a CUDA 12.6 build, run
`pip index versions torch --index-url https://download.pytorch.org/whl/cu126`.
Run it on the Linux server; the CUDA indexes have no macOS wheels.

#### Test the P40 setup

```bash
# Only -cu12 NVIDIA packages should be listed, and torch should end in +cu126
pip list 2>/dev/null | grep -iE "^torch|nvidia|triton"

# Expect: 2.14.0+cu126 True [... 'sm_60' ...]
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_arch_list())"

# Run a matrix multiply on the GPU; expect a number followed by "Tesla P40"
python -c "import torch; x = torch.randn(4096, 4096, device='cuda'); print((x @ x).sum().item(), torch.cuda.get_device_name(0))"
```

The architecture list includes `sm_60` but not `sm_61`. That is expected:
code compiled for `sm_60` runs on `sm_61` GPUs.

If `LD_LIBRARY_PATH` points at a system CUDA installation and torch fails to
import, run the check with it cleared (`LD_LIBRARY_PATH= python -c ...`). If
that works, remove the system CUDA directory from `LD_LIBRARY_PATH`; the pip
packages provide every CUDA library torch needs.

#### Run a model on the P40

The P40 has no `bfloat16` support and very low `float16` throughput, so use
`float32` with a model that fits in 24 GB:

```bash
applicator run tests/fixtures/input.example.csv tests/fixtures/prompt.example.json \
  --column text \
  --output out.csv \
  --local-model Qwen/Qwen3-4B \
  --device cuda \
  --dtype float32 \
  --no-think \
  --max-new-tokens 256
```

The log should report `Model loaded on cuda:0`. On the example fixture this
runs at about 1.7 seconds per row. To confirm the GPU is doing the work, watch
memory use and GPU utilization in another terminal while rows are processed:

```bash
watch -n1 nvidia-smi
```

`Qwen/Qwen3-8B` with `--dtype float16` also fits (about 16 GB) but may be
slower on this GPU. `Qwen/Qwen3-14B` does not fit in 24 GB at any of the
supported dtypes.

### Logging

`transformers` warnings are suppressed because many of them repeat for every
row. Set `TRANSFORMERS_VERBOSITY=warning` to show them when debugging:

```bash
TRANSFORMERS_VERBOSITY=warning applicator run ...
```

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
- Makes one LLM invocation per input row. A local model is loaded once
  before the first row and reused for every row.
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

The tests mock the LLM client and the Hugging Face libraries, so they do not
make network requests, download models, or require an API key or a GPU. The
suite covers:

- JSON prompt validation
- CSV transformation and validation
- LLM configuration and response conversion
- Local model loading, generation settings, and the `--no-think` switch
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
│   ├── core.py      # API and local model setup, CSV processing
│   └── readers.py   # JSON prompt and CSV readers
├── tests/
│   ├── fixtures/    # Reusable CSV and JSON test inputs
│   ├── conftest.py  # Shared pytest fixtures and test stubs
│   ├── test_cli.py
│   ├── test_core.py
│   └── test_readers.py
├── requirements.txt # Runtime dependencies
└── setup.py         # Package metadata, console entry point, `local` extra
```
