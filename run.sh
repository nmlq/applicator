#!/usr/bin/env bash
# Run applicator against a local Ollama server, once per entry in the RUNS
# array below. Each entry names its own input CSV, prompt JSON and output CSV.
set -euo pipefail

VENV_DIR="${VENV_DIR:-venv}"
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

APPLICATOR="${APPLICATOR:-applicator-v0.7.0-linux-x86_64-py312}"

# Input CSV column(s) that each prompt is applied to, separated by spaces.
# Override with e.g. COLUMN="ReportText" ./run.sh
COLUMN="${COLUMN:-ReportText ImpressionText}"
read -r -a COLUMN_ARGS <<< "$COLUMN"

OLLAMA_BASE_URL="${OLLAMA_BASE_URL:-http://localhost:11434/v1}"
OLLAMA_MODEL="${OLLAMA_MODEL:-medgemma:4b}"
OLLAMA_API_KEY="${OLLAMA_API_KEY:-ollama}"
BATCH_SIZE="${BATCH_SIZE:-4}"

# One line per run, in the form:
#   "input_csv|prompt_json|output_csv"
# Fields are separated by "|", so paths must not contain that character.
# Add, remove or edit lines to change what gets run.
RUNS=(
  "mapp.psma.csv|prompts/psma_petct_nodal_metastasis.json|output.mapp.psma.psma_petct_nodal_metastasis.csv"
  "mapp.psma.csv|prompts/psma_petct_bone_metastasis.json|output.mapp.psma.psma_petct_bone_metastasis.csv"
  "mapp.mriprostate.csv|prompts/pelvic_mri_scan_intent.json|output.mapp.mriprostate.pelvic_mri_scan_intent.csv"
  "mapp.mriprostate.csv|prompts/pelvic_mri_extracapsular_extension.json|output.mapp.mriprostate.pelvic_mri_extracapsular_extension.csv"
  "mapp.mriprostate.csv|prompts/pelvic_mri_seminal_vesicle_invasion.json|output.mapp.mriprostate.pelvic_mri_seminal_vesicle_invasion.csv"
  "mapp.mriprostate.csv|prompts/pelvic_mri_nodal_metastasis.json|output.mapp.mriprostate.pelvic_mri_nodal_metastasis.csv"
  "mapp.mriprostate.csv|prompts/pelvic_mri_bone_metastasis.json|output.mapp.mriprostate.pelvic_mri_bone_metastasis.csv"
  "mapp.mriprostate.csv|prompts/pelvic_mri_pirads_lesions.json|output.mapp.mriprostate.pelvic_mri_pirads_lesions.csv"
  "mapp.mriprostate.csv|prompts/pelvic_mri_prostate_volume.json|output.mapp.mriprostate.pelvic_mri_prostate_volume.csv"
  "mapp.pet_ct.csv|prompts/ct_abdomen_pelvis_nodal_metastasis.json|output.mapp.pet_ct.ct_abdomen_pelvis_nodal_metastasis.csv"
  "mapp.pet_ct.csv|prompts/ct_abdomen_pelvis_bone_metastasis.json|output.mapp.pet_ct.ct_abdomen_pelvis_bone_metastasis.csv"
  "resolve.tc_bonescan.csv|prompts/tc99m_bone_scan_bone_metastasis.json|output.resolve.tc_bonescan.tc99m_bone_scan_bone_metastasis.csv"
  "mapp.mriprostate.csv|prompts/entity_extractor_prompt.json|output.mapp.mriprostate.entity_extractor_prompt.csv"
  "mapp.mriprostate.csv|prompts/pirad_extractor_prompt.json|output.mapp.mriprostate.pirad_extractor_prompt.csv"
  "mapp.mripelvis.csv|prompts/pelvic_mri_scan_intent.json|output.mapp.mripelvis.pelvic_mri_scan_intent.csv"
  "mapp.mripelvis.csv|prompts/pelvic_mri_extracapsular_extension.json|output.mapp.mripelvis.pelvic_mri_extracapsular_extension.csv"
  "mapp.mripelvis.csv|prompts/pelvic_mri_seminal_vesicle_invasion.json|output.mapp.mripelvis.pelvic_mri_seminal_vesicle_invasion.csv"
  "mapp.mripelvis.csv|prompts/pelvic_mri_nodal_metastasis.json|output.mapp.mripelvis.pelvic_mri_nodal_metastasis.csv"
  "mapp.mripelvis.csv|prompts/pelvic_mri_bone_metastasis.json|output.mapp.mripelvis.pelvic_mri_bone_metastasis.csv"
  "mapp.mripelvis.csv|prompts/pelvic_mri_pirads_lesions.json|output.mapp.mripelvis.pelvic_mri_pirads_lesions.csv"
  "mapp.mripelvis.csv|prompts/pelvic_mri_prostate_volume.json|output.mapp.mripelvis.pelvic_mri_prostate_volume.csv"
  "mapp.mripelvis.csv|prompts/entity_extractor_prompt.json|output.mapp.mripelvis.entity_extractor_prompt.csv"
  "mapp.mripelvis.csv|prompts/pirad_extractor_prompt.json|output.mapp.mripelvis.pirad_extractor_prompt.csv"
)

# Check every entry in RUNS before anything runs, so a missing file
# fails fast instead of partway through a long batch. Reports every problem
# found, then exits non-zero if there were any.
validate_runs() {
  local errors=0 run input_csv prompt_json output_csv extra file stat_output

  if [ ${#RUNS[@]} -eq 0 ]; then
    echo "RUNS is empty: nothing to run" >&2
    return 1
  fi

  for run in "${RUNS[@]}"; do
    IFS='|' read -r input_csv prompt_json output_csv extra <<< "$run"

    if [ -z "$input_csv" ] || [ -z "$prompt_json" ] || [ -z "$output_csv" ] || [ -n "$extra" ]; then
      echo "Invalid entry (expected input_csv|prompt_json|output_csv): $run" >&2
      errors=$((errors + 1))
      continue
    fi

    # stat fails if the file is missing or can't be reached; print its error
    # output so the reason is shown alongside the entry.
    for file in "$input_csv" "$prompt_json"; do
      if ! stat_output=$(stat "$file" 2>&1); then
        echo "$stat_output" >&2
        errors=$((errors + 1))
      fi
    done
  done

  if [ "$errors" -gt 0 ]; then
    echo "Validation failed with $errors error(s); nothing was run." >&2
    return 1
  fi
  echo "Validated ${#RUNS[@]} run(s)."
}

# Run applicator once per entry in RUNS.
run_applicator() {
  local run input_csv prompt_json output_csv
  local index=0 total=${#RUNS[@]}

  for run in "${RUNS[@]}"; do
    index=$((index + 1))
    # Split the entry on "|" into its three fields.
    IFS='|' read -r input_csv prompt_json output_csv <<< "$run"

    # Make sure the output directory exists before writing to it.
    mkdir -p "$(dirname "$output_csv")"

    echo "[$index/$total] Running $prompt_json on $input_csv ($COLUMN) -> $output_csv"
    "$APPLICATOR" run "$input_csv" "$prompt_json" \
      --column "${COLUMN_ARGS[@]}" \
      --output "$output_csv" \
      --model "$OLLAMA_MODEL" \
      --base-url "$OLLAMA_BASE_URL" \
      --api-key "$OLLAMA_API_KEY" \
      --batch-size "$BATCH_SIZE" \
      --no-think
  done
}

main() {
  validate_runs
  run_applicator
}

main "$@"
