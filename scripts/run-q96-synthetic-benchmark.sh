#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
# shellcheck source=project-env.sh
source "${script_dir}/project-env.sh"

rows="${Q96_SYNTHETIC_ROWS:-33554432}"
data_root="${Q96_SYNTHETIC_DATA:-${repo_root}/.tools/q96-synthetic-${rows}}"
result_root="${Q96_SYNTHETIC_RESULTS:-${repo_root}/.tools/q96-synthetic-results-${rows}}"
warmups="${Q96_SYNTHETIC_WARMUPS:-3}"
runs="${Q96_SYNTHETIC_RUNS:-7}"
build_root="${repo_root}/build/spark-plugin"

"${script_dir}/build-spark-plugin.sh"
mkdir -p "${result_root}"

common=(
  --master 'local[8]'
  --driver-memory 6g
  --conf spark.ui.enabled=false
  --conf spark.sql.adaptive.enabled=false
  --conf spark.sql.ansi.enabled=false
  --conf spark.sql.shuffle.partitions=16
  --conf spark.sql.parquet.columnarReaderBatchSize=1048576
  --conf spark.sql.columnVector.offheap.enabled=true
  --class io.github.mohitpatil.sparkmetal.Q96SyntheticBenchmark
)

if [[ ! -f "${data_root}/store_sales/_SUCCESS" ]]; then
  spark-submit "${common[@]}" \
    "${build_root}/spark-metal-plugin.jar" \
    generate "${data_root}" "${rows}"
fi

spark-submit "${common[@]}" \
  "${build_root}/spark-metal-plugin.jar" \
  run "${data_root}" "${result_root}/cpu.json" "${warmups}" "${runs}" false

spark-submit "${common[@]}" \
  --conf spark.sql.extensions=io.github.mohitpatil.sparkmetal.SparkMetalExtensions \
  --conf "spark.metal.nativeLibrary=${build_root}/libsparkmetal.dylib" \
  --conf "spark.metal.metalLibrary=${build_root}/kernels.metallib" \
  "${build_root}/spark-metal-plugin.jar" \
  run "${data_root}" "${result_root}/metal.json" "${warmups}" "${runs}" true

python3 - "${result_root}/cpu.json" "${result_root}/metal.json" "${rows}" <<'PYTHON'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as source:
    cpu = json.load(source)
with open(sys.argv[2], encoding="utf-8") as source:
    metal = json.load(source)
if cpu["result"] != metal["result"]:
    raise SystemExit("CPU and Metal q96-shaped results differ")
print(json.dumps({
    "rows": int(sys.argv[3]),
    "cpuMedianSeconds": cpu["medianSeconds"],
    "metalMedianSeconds": metal["medianSeconds"],
    "speedup": cpu["medianSeconds"] / metal["medianSeconds"],
    "resultMatch": True,
}, indent=2))
PYTHON
