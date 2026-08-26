#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
# shellcheck source=project-env.sh
source "${script_dir}/project-env.sh"

data_dir="${1:-${repo_root}/benchmark-data/tpcds-sf10-parquet}"
queries_dir="${2:-${repo_root}/.tools/spark-assets/sql/core/src/test/resources/tpcds}"
selection="${3:-all}"

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

# Deliberately mirrors run-tpcds-metal.sh's Spark configuration (AQE off,
# ANSI off, off-heap column vectors on, shuffle partitions 16, vectorized
# reader on) but omits spark.sql.extensions: this probe wants to see the
# stock HashAggregateExec nodes GroupedAggregateShape.matchRegion classifies,
# not whatever SparkMetalColumnarRule would already have replaced them with.
spark-submit \
  --master 'local[8]' \
  --driver-memory 6g \
  --conf spark.ui.enabled=false \
  --conf spark.sql.adaptive.enabled=false \
  --conf spark.sql.ansi.enabled=false \
  --conf spark.sql.shuffle.partitions=16 \
  --conf spark.sql.parquet.enableVectorizedReader=true \
  --conf spark.sql.parquet.columnarReaderBatchSize=1048576 \
  --conf spark.sql.columnVector.offheap.enabled=true \
  --class io.github.mohitpatil.sparkmetal.GroupedAggregateShape \
  "${build_root}/spark-metal-plugin.jar" \
  "${data_dir}" "${queries_dir}" "${selection}"
