#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
# shellcheck source=project-env.sh
source "${script_dir}/project-env.sh"

raw_dir="${TPCDS_RAW_DIR:-${repo_root}/benchmark-data/tpcds-sf10-raw}"
parquet_dir="${TPCDS_PARQUET_DIR:-${repo_root}/benchmark-data/tpcds-sf10-parquet}"
ddl="${repo_root}/.tools/tpcds-kit/tools/tpcds.sql"

if [[ ! -f "${ddl}" ]]; then
  echo "TPC-DS schema not found. Run scripts/setup-tpcds-kit.sh after reviewing its licence." >&2
  exit 1
fi

spark-submit \
  --master 'local[8]' \
  --driver-memory 6g \
  --conf spark.ui.enabled=false \
  --conf spark.sql.shuffle.partitions=16 \
  "${repo_root}/benchmarks/tpcds/prepare_data.py" \
  --raw-dir "${raw_dir}" \
  --parquet-dir "${parquet_dir}" \
  --ddl "${ddl}" \
  "$@"
