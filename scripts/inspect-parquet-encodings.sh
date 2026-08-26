#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
source "${script_dir}/project-env.sh"
"${script_dir}/build-spark-plugin.sh"
data_dir="${1:-${repo_root}/benchmark-data/tpcds-sf10-parquet/store_sales}"
columns="${2:-ss_sold_time_sk,ss_hdemo_sk,ss_store_sk}"
java -cp "${SPARK_HOME}/jars/*:${repo_root}/build/spark-plugin/spark-metal-plugin.jar" \
  io.github.mohitpatil.sparkmetal.ParquetEligibility "${columns}" \
  "${data_dir}"/*.parquet
