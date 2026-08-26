#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
# shellcheck source=project-env.sh
source "${script_dir}/project-env.sh"

data_dir="${TPCDS_PARQUET_DIR:-${repo_root}/benchmark-data/tpcds-sf10-parquet}"
queries_dir="${repo_root}/.tools/spark-assets/sql/core/src/test/resources/tpcds"
run_id="$(date -u '+%Y%m%dT%H%M%SZ')"
output_dir="${TPCDS_RESULT_DIR:-${repo_root}/benchmark-results/metal-${run_id}}"
spark_master="${TPCDS_MASTER:-local[8]}"

if [[ ! -d "${data_dir}" ]]; then
  echo "TPC-DS SF10 Parquet data is missing: ${data_dir}" >&2
  echo "Generate it only after explicitly accepting the TPC-DS kit EULA." >&2
  exit 1
fi
if [[ ! -d "${queries_dir}" ]]; then
  echo "Pinned Spark queries are missing. Run scripts/fetch-spark-tpcds-assets.sh." >&2
  exit 1
fi

"${script_dir}/build-spark-plugin.sh"
build_root="${repo_root}/build/spark-plugin"

spark-submit \
  --master "${spark_master}" \
  --driver-memory 6g \
  --jars "${build_root}/spark-metal-plugin.jar" \
  --conf spark.ui.enabled=false \
  --conf spark.sql.adaptive.enabled=false \
  --conf spark.sql.ansi.enabled=false \
  --conf spark.sql.shuffle.partitions=16 \
  --conf spark.sql.parquet.enableVectorizedReader=true \
  --conf spark.sql.parquet.columnarReaderBatchSize=1048576 \
  --conf spark.sql.columnVector.offheap.enabled=true \
  --conf spark.sql.extensions=io.github.mohitpatil.sparkmetal.SparkMetalExtensions \
  --conf "spark.metal.nativeLibrary=${build_root}/libsparkmetal.dylib" \
  --conf "spark.metal.metalLibrary=${build_root}/kernels.metallib" \
  "${repo_root}/benchmarks/tpcds/run_queries.py" \
  --data-dir "${data_dir}" \
  --queries-dir "${queries_dir}" \
  --output-dir "${output_dir}" \
  --label metal \
  "$@"

echo "Results written under ${output_dir}."
