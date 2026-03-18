#!/usr/bin/env bash
set -u

SCRIPTS=(
  "scripts/eval/batch_evaluation2.sh"
  "scripts/eval/batch_evaluation3.sh"
  "scripts/eval/batch_evaluation4.sh"
  "scripts/eval/batch_evaluation5.sh"
  "scripts/eval/batch_evaluation6.sh"
  "scripts/eval/batch_evaluation7.sh"
)

for script in "${SCRIPTS[@]}"; do
  if [[ ! -f "${script}" ]]; then
    echo "[ERROR] Missing script: ${script}" >&2
    exit 1
  fi
done

declare -a PIDS
declare -a NAMES
FAIL_COUNT=0

for script in "${SCRIPTS[@]}"; do
  echo "[START] ${script}"
  sh "${script}" &
  PIDS+=("$!")
  NAMES+=("${script}")
done

for i in "${!PIDS[@]}"; do
  pid="${PIDS[$i]}"
  name="${NAMES[$i]}"
  if wait "${pid}"; then
    echo "[DONE] ${name}"
  else
    status=$?
    echo "[FAIL] ${name} (exit ${status})" >&2
    FAIL_COUNT=$((FAIL_COUNT + 1))
  fi
done

if [[ "${FAIL_COUNT}" -gt 0 ]]; then
  echo "[SUMMARY] ${FAIL_COUNT}/${#SCRIPTS[@]} scripts failed." >&2
  exit 1
fi

echo "[SUMMARY] All ${#SCRIPTS[@]} scripts finished successfully."
