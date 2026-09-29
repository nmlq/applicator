# Applicator

Applicator is a small Python CLI that applies an LLM prompt to every value in
one CSV column and writes each result into a new CSV column. It can call an
OpenAI-compatible API, such as OpenAI or a local Ollama server, or run a
Hugging Face model on the local GPU, and it can process rows in batches.

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

`--load-in-4bit` also needs `bitsandbytes`, which is not part of the `local`
extra. Install it with `--no-deps`: `bitsandbytes` depends on `torch`, and a
normal install can replace a working CUDA build of torch with the default one:

```bash
pip install --no-deps bitsandbytes
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
| `--batch-size` | — | no | `1` | Rows processed together; see [Batching](#batching) |
| `--no-think` | — | no | off | Disable thinking in reasoning models; see [Reasoning models](#reasoning-models) |

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

For quantized models on older NVIDIA GPUs such as the Tesla P40, Ollama is
usually faster than a [local model](#local-models): its engine computes
directly on quantized weights and handles long prompts efficiently on those
GPUs.

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
| `--load-in-4bit` | off | Quantize weights to 4-bit with `bitsandbytes`; `--dtype` sets the compute dtype |

`--device auto` places the model on the best available device, but falls back
to the CPU silently when torch cannot use the GPU. Use `--device cuda` or
`--device mps` to fail with an error instead.

### Reasoning models

Reasoning models such as Qwen3 think before answering by default. Thinking
can generate hundreds of tokens per row and makes each row much slower.
`--no-think` turns it off:

- With `--local-model`, it passes `enable_thinking=False` to the model's chat
  template, so the model answers directly. Without it, the thinking is written
  into the output column as a `<think>...</think>` block and counts against
  `--max-new-tokens`. Models whose chat template has no thinking switch
  ignore the option.
- With an API, it sends `reasoning_effort="none"`. Ollama returns thinking
  in a separate field that Applicator does not save, so without `--no-think`
  the output column looks clean while each row still pays for the thinking.
  Ollama 0.34 ignores both a `/no_think` prefix in the prompt and its native
  `think: false` option on this endpoint; `reasoning_effort="none"` is the
  setting it honors. Hosted providers may reject `"none"` for models that do
  not support it.

In testing with Qwen3 on Ollama, `--no-think` was 4.6 times faster for eight
reports on Apple Silicon and 6.5 times faster for a single report on a Tesla
P40.

### Choosing a model size

The weights must fit in GPU memory. Approximate size is the parameter count
multiplied by the bytes per weight: 4 bytes for `float32` and 2 for `float16`
or `bfloat16`. Leave a few gigabytes free for generation.

| Model | `float32` | `float16` / `bfloat16` | `--load-in-4bit` |
| --- | --- | --- | --- |
| `Qwen/Qwen3-4B` | ~16 GB | ~8 GB | ~3–4 GB |
| `Qwen/Qwen3-8B` | ~33 GB | ~16 GB | ~6–7 GB |
| `Qwen/Qwen3-14B` | ~59 GB | ~30 GB | ~10–14 GB |

When a model does not fit, `--device auto` moves some layers to the CPU. The
run still completes, but slowly, and the log still reports the GPU device.
With `--device cuda` the run stops with `torch.OutOfMemoryError`. Choose a
smaller model, a smaller dtype, or `--load-in-4bit`.

Ollama's default model tags are already quantized to about 4 bits, so Ollama
can fit a model that fails here at full precision. `--load-in-4bit` quantizes
the full-precision Hugging Face weights while loading. It needs `bitsandbytes`
(`pip install bitsandbytes`) and an NVIDIA GPU. The 4-bit sizes above are
estimates: layers that are not quantized, such as the embeddings, stay at
`--dtype`, so the size varies with it.

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
slower on this GPU.

`Qwen/Qwen3-14B` does not fit in 24 GB at full precision. With
`--load-in-4bit --dtype float32` it loads and runs on the P40, but slowly:
about 5.2 seconds per report at `--batch-size 8`, and batch sizes 16 and 32
ran out of memory with real MRI reports. The same model through
[Ollama](#ollama) with `--no-think` ran at about 1.2 seconds per report. For
14B models on the P40, use Ollama. See
[Performance baselines](#performance-baselines) for the measurements.

### Batching

`--batch-size N` processes N rows at a time. With a local model, the N prompts
run through the GPU as one batch, which uses the GPU far better than one row
at a time. With an API, the N requests are sent concurrently.

```bash
applicator run input.csv prompt.json \
  --column text \
  --output output.csv \
  --local-model Qwen/Qwen3-4B \
  --device cuda \
  --no-think \
  --max-new-tokens 256 \
  --batch-size 8
```

In testing on Apple Silicon with Qwen3-14B, `--batch-size 4` processed rows
about three times faster than `--batch-size 1`.

Outputs can differ slightly between batch sizes, even with greedy decoding
(no `--temperature`). A batch is computed with different GPU kernels than a
single row, so the arithmetic rounds differently, and when two next words are
almost equally likely the model can pick the other one. The rest of the
answer then differs in wording. Smaller models are affected more often than
larger ones.

To find a good value, run the [batch size benchmark](#batch-size-benchmark)
or start at 4 and double it while the rows per second shown in the progress
bar keep improving. Larger batches use more GPU memory,
because each row in the batch needs its own working memory for generation, so
the limit is lower for long inputs and large `--max-new-tokens`. If a run
fails with `torch.OutOfMemoryError`, halve the batch size. Rows in a batch are
padded to the longest input in that batch, so inputs of very different lengths
waste some of the work.

Output is written and flushed after each batch, so an interrupted run loses
at most one batch of rows.

### Batch size benchmark

`scripts/benchmark_batch_size.py` loads a local model once and processes the
same rows at several batch sizes. By default it uses 64 synthetic MRI reports
from `tests/fixtures/mri_reports.example.csv`, so no external data is needed.
For each batch size it prints the time, rows per second, speedup over the
first batch size, peak GPU memory (NVIDIA only), and how many outputs match
the first batch size:

```bash
python scripts/benchmark_batch_size.py \
  --local-model Qwen/Qwen3-4B \
  --device cuda \
  --dtype float32 \
  --no-think \
  --batch-sizes 1 2 4 8 16 32
```

The benchmark accepts `--device`, `--dtype`, `--max-new-tokens` (default
256), `--no-think` and `--load-in-4bit` with the same meaning as for
`applicator run`. Use `--rows N` for a quicker run on the first N rows,
`--input`, `--prompt` and `--column` to benchmark other data, and
`--output-dir` to keep each run's CSV. When a batch size runs out of GPU
memory, the benchmark reports it and skips larger sizes.

### Logging

`transformers` warnings are suppressed because many of them repeat for every
row. Set `TRANSFORMERS_VERBOSITY=warning` to show them when debugging:

```bash
TRANSFORMERS_VERBOSITY=warning applicator run ...
```

## Performance baselines

These measurements were taken while developing local model support and
batching. They show relative differences between configurations; absolute
times depend on the GPU, the length of the inputs and answers, and the
prompt. Unless noted, answers were one-sentence summaries.

### Test machines

| | Apple Silicon Mac | Linux server |
| --- | --- | --- |
| GPU | Apple Silicon GPU (`mps`), unified memory | NVIDIA Tesla P40, 24 GB |
| Driver and CUDA | — | driver 560.35.05, CUDA 12.6 |
| Python | 3.14 | 3.12 |
| torch | 2.14.0 | 2.14.0+cu126 |
| transformers | 5.17.0 | 5.x |
| Ollama | 0.34.4 | not recorded |

Inputs:

- **Example fixture**: the two short rows in `tests/fixtures/input.example.csv`.
- **Synthetic MRI reports**: the 64 fictional reports in
  `tests/fixtures/mri_reports.example.csv`, 8 to 153 words each.
- **Real MRI reports**: a private set of reports, not included in the
  repository. A sampled report was about 440 tokens. In one batch of 16, the
  longest prompt, including the instructions and chat template, was about
  1,600 tokens.

### Qwen3-14B on the Tesla P40, real MRI reports

| Engine and settings | `--batch-size` | Seconds per report | Rows measured |
| --- | --- | --- | --- |
| `--local-model`, `float32` | — | does not fit (needs ~59 GB) | — |
| `--local-model`, `--load-in-4bit --dtype float32 --no-think` | 32 | out of GPU memory | — |
| `--local-model`, `--load-in-4bit --dtype float32 --no-think` | 16 | out of GPU memory | — |
| `--local-model`, `--load-in-4bit --dtype float32 --no-think` | 8 | 5.24 | 744 |
| Ollama `qwen3:14b`, thinking on | 4 | 7.72 | 112 |
| Ollama `qwen3:14b`, thinking on (`/no_think` in the prompt, ignored) | 4 | 9.30 | 8 |
| Ollama `qwen3:14b`, `--no-think` | 4 | **1.15** | 16 |

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

### Batch sizes on the Tesla P40, Qwen3-4B

`scripts/benchmark_batch_size.py` with `--local-model Qwen/Qwen3-4B --device
cuda --dtype float32 --no-think --max-new-tokens 256` on the 64 synthetic MRI
reports:

| `--batch-size` | Seconds | Rows per second | Speedup | Peak GPU memory | Outputs matching batch size 1 |
| --- | --- | --- | --- | --- | --- |
| 1 | 132.7 | 0.48 | 1.00× | 16.2 GB | 64/64 |
| 2 | 93.6 | 0.68 | 1.42× | 16.4 GB | 64/64 |
| 4 | 65.4 | 0.98 | 2.03× | 16.6 GB | 64/64 |
| 8 | 52.6 | 1.22 | 2.52× | 17.2 GB | 64/64 |
| 16 | 42.8 | 1.49 | 3.10× | 18.2 GB | 64/64 |
| 32 | 37.5 | 1.71 | 3.54× | 20.3 GB | 64/64 |

The gain per doubling shrinks as batches grow, and memory grows with batch
size. These reports are short; with the real reports, which are several
times longer, each row in a batch needs several times more memory, so the
largest batch size that fits is lower.

### Apple Silicon Mac

Qwen3-14B as a local model (`--dtype auto`, which loads `bfloat16` weights):

| Input | Settings | `--batch-size` | Result |
| --- | --- | --- | --- |
| Example fixture, 2 rows | thinking on, `--max-new-tokens 1024` | 1 | 31.2 s per row |
| Example fixture, 2 rows | `--no-think --max-new-tokens 256` | 1 | 2.08 s per row |
| 8 short rows | `--no-think --max-new-tokens 24` | 1 | 22 s for 8 rows |
| 8 short rows | `--no-think --max-new-tokens 24` | 4 | 7 s for 8 rows |

All 8 short-row outputs were identical at batch sizes 1 and 4.

SmolLM2-135M-Instruct as a local model, benchmarked on the first 16 synthetic
MRI reports with `--max-new-tokens 64`:

| `--batch-size` | Rows per second | Speedup | Outputs matching batch size 1 |
| --- | --- | --- | --- |
| 1 | 3.52 | 1.00× | 16/16 |
| 4 | 5.40 | 1.53× | 13/16 |
| 8 | 8.38 | 2.38× | 12/16 |
| 16 | 11.49 | 3.26× | 11/16 |

The mismatched outputs were reworded summaries of the same report, not
corrupted text; see [Batching](#batching) for why this happens.

Ollama `qwen3:1.7b` through Applicator, 8 synthetic MRI reports with
`--batch-size 4`: 13.6 seconds with thinking on and 2.95 seconds with
`--no-think`, including start-up.

### Model load times

| Model | Machine | Load time |
| --- | --- | --- |
| Qwen3-4B, `float32` | Tesla P40 | ~65 s on first run, including a 4 GB download |
| Qwen3-14B, `--load-in-4bit` | Tesla P40 | ~24 s from the local cache |
| Qwen3-14B, `bfloat16` | Apple Silicon | ~10 s from the local cache |

A local model is loaded once per run, so load time matters only for short
runs.

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
- **Batch rows.** On the P40, Qwen3-4B processed rows 3.5 times faster at
  `--batch-size 32` than at 1. Pick the largest batch size that fits in GPU
  memory with your real inputs; longer inputs lower the limit.
- **On older NVIDIA GPUs, run 14B models through Ollama.** On the P40,
  Ollama with `--no-think` and 4 parallel requests was about 4.5 times faster
  than a 4-bit local model, and fit comfortably in memory.
- **Use local models on Apple Silicon or recent NVIDIA GPUs,** or with models
  small enough to run at full precision, such as Qwen3-4B on the P40.

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
- Sends rows to the model in batches of `--batch-size` rows (default 1). A
  local model generates a batch together on the GPU; an API receives the
  batch's requests concurrently. A local model is loaded once before the first
  batch and reused for every batch.
- Preserves all original CSV columns.
- Adds the generated response to `applicator_output` by default.
- Writes and flushes each completed batch immediately.
- Prints progress for each processed row.
- Raises an error when the input CSV has no header or the requested column is
  missing.
- Does not implement agents or MCP.

## Tests

Run the test suite from the project root:

```bash
python3 -m pytest -q
```

The tests mock the LLM client and the Hugging Face libraries, so they do not
make network requests, download models, or require an API key or a GPU. The
suite covers:

- JSON prompt validation
- CSV transformation and validation, including batched processing
- LLM configuration and response conversion
- Local model loading, generation settings, and the `--no-think` switch for
  local models and APIs
- CLI parsing, validation, and dispatch
- End-to-end reader behavior using the example fixtures

## Test fixtures

Reusable example inputs are stored in `tests/fixtures/`:

- `tests/fixtures/input.example.csv` contains `id` and `text` columns with two
  sample rows.
- `tests/fixtures/prompt.example.json` contains a valid prompt using
  `{input}`.
- `tests/fixtures/mri_reports.example.csv` contains `id` and `report` columns
  with 64 synthetic MRI reports of 8 to 153 words. The reports are generated
  from fictional findings and contain no patient data. The batch size
  benchmark uses them by default.
- `tests/fixtures/mri_prompt.example.json` asks for a one-sentence summary of
  a report.

The fixture paths are exposed as the `example_input_csv` and
`example_prompt_json` pytest fixtures in `tests/conftest.py`.

## Project layout

```text
applicator/
├── .claude/
│   └── CLAUDE.md    # Rules for Claude Code in this repository
├── applicator/
│   ├── cli.py       # argparse command-line interface
│   ├── core.py      # API and local model setup, CSV processing
│   └── readers.py   # JSON prompt and CSV readers
├── scripts/
│   └── benchmark_batch_size.py # Local model throughput by batch size
├── tests/
│   ├── fixtures/    # Reusable CSV and JSON test inputs
│   ├── conftest.py  # Shared pytest fixtures and test stubs
│   ├── test_cli.py
│   ├── test_core.py
│   └── test_readers.py
├── requirements.txt # Runtime dependencies
└── setup.py         # Package metadata, console entry point, `local` extra
```
