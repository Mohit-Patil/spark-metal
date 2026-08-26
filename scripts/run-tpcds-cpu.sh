#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
# shellcheck source=project-env.sh
source "${script_dir}/project-env.sh"

data_dir="${TPCDS_PARQUET_DIR:-${repo_root}/benchmark-data/tpcds-sf10-parquet}"
queries_dir="${repo_root}/.tools/spark-assets/sql/core/src/test/resources/tpcds"
run_id="$(date -u '+%Y%m%dT%H%M%SZ')"
output_dir="${TPCDS_RESULT_DIR:-${repo_root}/benchmark-results/cpu-${run_id}}"

if [[ ! -d "${queries_dir}" ]]; then
  echo "Pinned Spark queries are missing. Run scripts/fetch-spark-tpcds-assets.sh." >&2
  exit 1
fi

spark-submit \
  --master 'local[8]' \
  --driver-memory 6g \
  --conf spark.ui.enabled=false \
  --conf spark.sql.adaptive.enabled=true \
  --conf spark.sql.shuffle.partitions=16 \
  --conf spark.sql.parquet.enableVectorizedReader=true \
  "${repo_root}/benchmarks/tpcds/run_queries.py" \
  --data-dir "${data_dir}" \
  --queries-dir "${queries_dir}" \
  --output-dir "${output_dir}" \
  "$@"

echo "Results written under ${output_dir}."
