#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON_BIN:-python}"

"$python_bin" "$project_root/tools/target_statistics.py" \
  --output-json "$project_root/outputs/dataset_pair_pilot.json" \
  --metadata-json "$project_root/outputs/metadata_pilot.json" \
  --limit 12 \
  --preview-dir "$project_root/outputs/previews" \
  "$@"
