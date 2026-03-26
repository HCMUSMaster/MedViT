#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET_DIR="${SCRIPT_DIR}/dataset/pcam"
PYTHON_BIN="${PYTHON:-python3}"

mkdir -p "${TARGET_DIR}"

download_one() {
  local file_id="$1"
  local file_name="$2"
  local gz_path="${TARGET_DIR}/${file_name}.gz"
  local h5_path="${TARGET_DIR}/${file_name}"

  if [[ -f "${h5_path}" ]]; then
    printf 'skip %s (already exists)\n' "${file_name}"
    return 0
  fi

  printf 'downloading %s\n' "${file_name}"
  (cd "${TARGET_DIR}" && "${PYTHON_BIN}" -m gdown --id "${file_id}" -O "${file_name}.gz")
  gzip -d -f "${gz_path}"
}

FILES=(
  "1Ka0XfEMiwgCYPdTI-vv6eUElOBnKFKQ2 camelyonpatch_level_2_split_train_x.h5"
  "1269yhu3pZDP8UYFQs-NYs3FPwuK-nGSG camelyonpatch_level_2_split_train_y.h5"
  "1hgshYGWK8V-eGRy8LToWJJgDU_rXWVJ3 camelyonpatch_level_2_split_valid_x.h5"
  "1bH8ZRbhSVAhScTS0p9-ZzGnX91cHT3uO camelyonpatch_level_2_split_valid_y.h5"
  "1qV65ZqZvWzuIVthK8eVDhIwrbnsJdbg_ camelyonpatch_level_2_split_test_x.h5"
  "17BHrSrwWKjYsOgTMmoqrIjDy6Fa2o_gP camelyonpatch_level_2_split_test_y.h5"
)

pids=()
for entry in "${FILES[@]}"; do
  file_id="${entry%% *}"
  file_name="${entry#* }"
  download_one "${file_id}" "${file_name}" &
  pids+=("$!")
done

for pid in "${pids[@]}"; do
  wait "${pid}"
done

printf 'all PCAM files are ready in %s\n' "${TARGET_DIR}"