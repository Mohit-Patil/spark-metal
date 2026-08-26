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
spark_master="${TPCDS_MASTER:-local[8]}"

if [[ ! -d "${queries_dir}" ]]; then
  echo "Pinned Spark queries are missing. Run scripts/fetch-spark-tpcds-assets.sh." >&2
  exit 1
fi

spark-submit \
  --master "${spark_master}" \
  --driver-memory 6g \
  --conf spark.ui.enabled=false \
  --conf spark.sql.adaptive.enabled=false \
  --conf spark.sql.ansi.enabled=false \
  --conf spark.sql.shuffle.partitions=16 \
  --conf spark.sql.parquet.enableVectorizedReader=true \
  --conf spark.sql.parquet.columnarReaderBatchSize=1048576 \
  --conf spark.sql.columnVector.offheap.enabled=true \
  "${repo_root}/benchmarks/tpcds/run_queries.py" \
  --data-dir "${data_dir}" \
  --queries-dir "${queries_dir}" \
  --output-dir "${output_dir}" \
  --label cpu \
  "$@"

echo "Results written under ${output_dir}."
