#!/usr/bin/env bash
# Run applicator against a local Ollama server, once per prompt in PROMPTS_DIR
# and per column in COLUMNS.
set -euo pipefail

VENV_DIR="${VENV_DIR:-venv}"
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

INPUT_CSV="${INPUT_CSV:-input.csv}"
PROMPTS_DIR="${PROMPTS_DIR:-prompts}"
# Space-separated list of input CSV columns to run each prompt against.
COLUMNS="${COLUMNS:-ReportText ImpressionText}"

OLLAMA_BASE_URL="${OLLAMA_BASE_URL:-http://localhost:11434/v1}"
OLLAMA_MODEL="${OLLAMA_MODEL:-gemma3n:e2b}"
OLLAMA_API_KEY="${OLLAMA_API_KEY:-ollama}"
BATCH_SIZE="${BATCH_SIZE:-4}"

shopt -s nullglob
prompt_files=("$PROMPTS_DIR"/*.json)
if [ ${#prompt_files[@]} -eq 0 ]; then
  echo "No .json files found in $PROMPTS_DIR" >&2
  exit 1
fi

for prompt_json in "${prompt_files[@]}"; do
  for column in $COLUMNS; do
    output_csv="${prompt_json%.json}_${column}_output.csv"
    echo "Running $prompt_json on $column -> $output_csv"
    applicator run "$INPUT_CSV" "$prompt_json" \
      --column "$column" \
      --output "$output_csv" \
      --model "$OLLAMA_MODEL" \
      --base-url "$OLLAMA_BASE_URL" \
      --api-key "$OLLAMA_API_KEY" \
      --batch-size "$BATCH_SIZE" \
      --no-think
  done
done
