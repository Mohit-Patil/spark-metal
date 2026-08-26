#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
# shellcheck source=project-env.sh
source "${script_dir}/project-env.sh"

"${script_dir}/build-spark-plugin.sh"
build_root="${repo_root}/build/spark-plugin"
row_count="${SPARK_METAL_SYNTHETIC_ROWS:-33554432}"
core_count="${SPARK_METAL_CORES:-8}"
warmups="${SPARK_METAL_WARMUPS:-5}"
runs="${SPARK_METAL_RUNS:-11}"
batch_rows="${SPARK_METAL_BATCH_ROWS:-1048576}"
data_path="${repo_root}/.tools/synthetic-parquet-${row_count}"
result_path="${repo_root}/.tools/synthetic-results-${row_count}"
mkdir -p "${result_path}"

common=(
  --master "local[${core_count}]"
  --driver-memory 6g
  --conf spark.ui.enabled=false
  --conf spark.sql.adaptive.enabled=false
  --conf spark.sql.ansi.enabled=false
  --conf spark.sql.shuffle.partitions=16
  --conf "spark.sql.parquet.columnarReaderBatchSize=${batch_rows}"
  --conf spark.sql.columnVector.offheap.enabled=true
  --class io.github.mohitpatil.sparkmetal.SyntheticBenchmark
)

if [[ ! -d "${data_path}" ]]; then
  spark-submit "${common[@]}" \
    "${build_root}/spark-metal-plugin.jar" \
    generate "${data_path}" "${row_count}"
fi

spark-submit "${common[@]}" \
  "${build_root}/spark-metal-plugin.jar" \
  run "${data_path}" "${result_path}/cpu.json" "${warmups}" "${runs}"

spark-submit "${common[@]}" \
  --conf spark.sql.extensions=io.github.mohitpatil.sparkmetal.SparkMetalExtensions \
  --conf "spark.metal.nativeLibrary=${build_root}/libsparkmetal.dylib" \
  --conf "spark.metal.metalLibrary=${build_root}/kernels.metallib" \
  "${build_root}/spark-metal-plugin.jar" \
  run "${data_path}" "${result_path}/metal.json" "${warmups}" "${runs}"

python3 - "${result_path}/cpu.json" "${result_path}/metal.json" "${row_count}" <<'PYTHON'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as source:
    cpu = json.load(source)
with open(sys.argv[2], encoding="utf-8") as source:
    metal = json.load(source)
if cpu["result"] != metal["result"]:
    raise SystemExit("CPU and Metal results differ")
speedup = cpu["medianSeconds"] / metal["medianSeconds"]
print(json.dumps({
    "rows": int(sys.argv[3]),
    "cpuMedianSeconds": cpu["medianSeconds"],
    "metalMedianSeconds": metal["medianSeconds"],
    "speedup": speedup,
    "resultMatch": True,
}, indent=2))
PYTHON
