#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON_BIN:-python}"
cphd_file="/data/dyn/capella/data/CAPELLA_C02_SP_CPHD_HH_20201201165018_20201201165040.cphd"

"$python_bin" "$project_root/tools/target_statistics.py" \
  --output-json "$project_root/outputs/dataset_pair.json" \
  --cphd-file "$cphd_file" \
  --cphd-parameters-json "$project_root/outputs/cphd_parameters.json" \
  "$@"
