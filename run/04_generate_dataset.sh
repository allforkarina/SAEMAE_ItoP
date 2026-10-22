#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON_BIN:-python}"

"$python_bin" "$project_root/tools/target_statistics.py" \
  --output-json "$project_root/outputs/dataset_pair.json" \
  "$@"
