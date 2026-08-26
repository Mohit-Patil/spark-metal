#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
# shellcheck source=project-env.sh
source "${script_dir}/project-env.sh"

"${script_dir}/build-spark-plugin.sh"
build_root="${repo_root}/build/spark-plugin"
rows="${1:-1000003}"
data_root="${repo_root}/.tools/q96-membership-edge-${rows}"
result_root="${repo_root}/.tools/q96-membership-edge-results-${rows}"
mkdir -p "${result_root}"

common=(
  --master 'local[8]'
  --driver-memory 6g
  --conf spark.ui.enabled=false
  --conf spark.sql.adaptive.enabled=false
  --conf spark.sql.ansi.enabled=false
  --conf spark.sql.parquet.filterPushdown=false
  --conf spark.sql.columnVector.offheap.enabled=true
  --class io.github.mohitpatil.sparkmetal.Q96SyntheticBenchmark
)

if [[ ! -f "${data_root}/store_sales/_SUCCESS" ]]; then
  spark-submit "${common[@]}" \
    "${build_root}/spark-metal-plugin.jar" \
    generate-edge "${data_root}" "${rows}"
fi

spark-submit "${common[@]}" \
  "${build_root}/spark-metal-plugin.jar" \
  run "${data_root}" "${result_root}/cpu.json" 1 1 false

spark-submit "${common[@]}" \
  --conf spark.sql.extensions=io.github.mohitpatil.sparkmetal.SparkMetalExtensions \
  --conf "spark.metal.nativeLibrary=${build_root}/libsparkmetal.dylib" \
  --conf "spark.metal.metalLibrary=${build_root}/kernels.metallib" \
  "${build_root}/spark-metal-plugin.jar" \
  run "${data_root}" "${result_root}/metal.json" 1 1 true

spark-submit "${common[@]}" \
  --conf spark.sql.extensions=io.github.mohitpatil.sparkmetal.SparkMetalExtensions \
  --conf "spark.metal.nativeLibrary=${build_root}/libsparkmetal.dylib" \
  --conf "spark.metal.metalLibrary=${build_root}/kernels.metallib" \
  --conf spark.metal.parquetScan.enabled=false \
  "${build_root}/spark-metal-plugin.jar" \
  run "${data_root}" "${result_root}/metal-disabled.json" 1 1 true

spark-submit \
  --master 'local[8]' \
  --driver-memory 6g \
  --conf spark.ui.enabled=false \
  --conf spark.sql.adaptive.enabled=true \
  --conf spark.sql.ansi.enabled=false \
  --conf spark.sql.parquet.filterPushdown=false \
  --conf spark.sql.columnVector.offheap.enabled=true \
  --conf spark.sql.extensions=io.github.mohitpatil.sparkmetal.SparkMetalExtensions \
  --conf "spark.metal.nativeLibrary=${build_root}/libsparkmetal.dylib" \
  --conf "spark.metal.metalLibrary=${build_root}/kernels.metallib" \
  --class io.github.mohitpatil.sparkmetal.Q96SyntheticBenchmark \
  "${build_root}/spark-metal-plugin.jar" \
  run "${data_root}" "${result_root}/aqe-fallback.json" 1 1 false

python3 - "${result_root}/cpu.json" "${result_root}/metal.json" \
  "${result_root}/metal-disabled.json" "${result_root}/aqe-fallback.json" <<'PYTHON'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as source:
    cpu = json.load(source)
with open(sys.argv[2], encoding="utf-8") as source:
    metal = json.load(source)
with open(sys.argv[3], encoding="utf-8") as source:
    metal_disabled = json.load(source)
with open(sys.argv[4], encoding="utf-8") as source:
    aqe = json.load(source)
if not cpu["result"] == metal["result"] == metal_disabled["result"] == aqe["result"]:
    raise SystemExit(
        "edge-case mismatch: "
        f"CPU={cpu['result']}, Metal={metal['result']}, "
        f"MetalDisabled={metal_disabled['result']}, AQE={aqe['result']}"
    )
if "MetalParquetMembershipCount" not in metal["plan"]:
    raise SystemExit(
        f"expected MetalParquetMembershipCount in metal plan, got:\n{metal['plan']}"
    )
if "MetalFusedMembershipCount" not in metal_disabled["plan"]:
    raise SystemExit(
        "expected MetalFusedMembershipCount in metal-disabled plan "
        f"(spark.metal.parquetScan.enabled=false), got:\n{metal_disabled['plan']}"
    )
if "MetalParquetMembershipCount" in metal_disabled["plan"]:
    raise SystemExit(
        "expected MetalParquetMembershipCount to be absent when "
        f"spark.metal.parquetScan.enabled=false, got:\n{metal_disabled['plan']}"
    )
print(json.dumps({
    "cpuResult": cpu["result"],
    "metalResult": metal["result"],
    "metalDisabledResult": metal_disabled["result"],
    "aqeFallbackResult": aqe["result"],
    "duplicatesAndNullsMatch": True,
    "metalUsesParquetOperator": True,
    "metalDisabledUsesFusedOperator": True,
    "aqeFallsBack": not aqe["accelerated"],
}, indent=2))
PYTHON
