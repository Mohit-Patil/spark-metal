#!/usr/bin/env bash

# Standalone per-phase benchmark of the Metal grouped-aggregate pipeline:
# prices planning, split enumeration, dimension collect, GroupSpace build,
# row-group read, CPU page parse, Spark's own scan, and the end-to-end query
# (with the operators' fine-grained SQL metrics) for each requested query,
# in one Metal-enabled leg and one plain-CPU leg.
#
#   scripts/run-phase-benchmark.sh [--queries q3,q12,...] [--warmups N] [--runs N]

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
# shellcheck source=project-env.sh
source "${script_dir}/project-env.sh"

data_dir="${TPCDS_PARQUET_DIR:-${repo_root}/benchmark-data/tpcds-sf10-parquet}"
queries_dir="${repo_root}/.tools/spark-assets/sql/core/src/test/resources/tpcds"
run_id="$(date -u '+%Y%m%dT%H%M%SZ')"
output_dir="${PHASE_RESULT_DIR:-${repo_root}/benchmark-results/phase-${run_id}}"
spark_master="${PHASE_MASTER:-local[8]}"
# Default set: the three grouped-aggregate winners plus representative losers
# (q3: PLAIN-keyed single region; q12/q20: small web/catalog regions below
# parity; q74 is the worst loser but needs the region gate off to fire).
queries="${PHASE_QUERIES:-q3,q12,q20,q53,q63,q89}"
warmups="${PHASE_WARMUPS:-2}"
runs="${PHASE_RUNS:-5}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --queries)
      queries="$2"
      shift 2
      ;;
    --warmups)
      warmups="$2"
      shift 2
      ;;
    --runs)
      runs="$2"
      shift 2
      ;;
    *)
      echo "Unknown phase-benchmark option: $1" >&2
      exit 2
      ;;
  esac
done

if [[ ! -d "${data_dir}" ]]; then
  echo "TPC-DS SF10 Parquet data is missing: ${data_dir}" >&2
  exit 1
fi
if [[ ! -d "${queries_dir}" ]]; then
  echo "Pinned Spark queries are missing. Run scripts/fetch-spark-tpcds-assets.sh." >&2
  exit 1
fi

"${script_dir}/build-spark-plugin.sh"
build_root="${repo_root}/build/spark-plugin"
mkdir -p "${output_dir}"

common=(
  --master "${spark_master}"
  --driver-memory 6g
  --conf spark.ui.enabled=false
  --conf spark.sql.adaptive.enabled=false
  --conf spark.sql.ansi.enabled=false
  --conf spark.sql.shuffle.partitions=16
  --conf spark.sql.parquet.enableVectorizedReader=true
  --conf spark.sql.parquet.columnarReaderBatchSize=1048576
  --conf spark.sql.columnVector.offheap.enabled=true
  --class io.github.mohitpatil.sparkmetal.PhaseBenchmark
)

# Optional extra --conf overrides for the Metal leg (e.g.
# spark.metal.parquetAggregate.maxRegions=99 to un-gate the multi-region
# losers so their phases can be measured at all).
extra_conf_args=()
if [[ -n "${PHASE_METAL_EXTRA_CONF:-}" ]]; then
  for entry in ${PHASE_METAL_EXTRA_CONF}; do
    extra_conf_args+=(--conf "${entry}")
  done
fi

spark-submit "${common[@]}" \
  --conf spark.sql.extensions=io.github.mohitpatil.sparkmetal.SparkMetalExtensions \
  --conf "spark.metal.nativeLibrary=${build_root}/libsparkmetal.dylib" \
  --conf "spark.metal.metalLibrary=${build_root}/kernels.metallib" \
  "${extra_conf_args[@]+"${extra_conf_args[@]}"}" \
  "${build_root}/spark-metal-plugin.jar" \
  run "${data_dir}" "${queries_dir}" "${output_dir}/metal.json" "${warmups}" "${runs}" "${queries}"

spark-submit "${common[@]}" \
  "${build_root}/spark-metal-plugin.jar" \
  run "${data_dir}" "${queries_dir}" "${output_dir}/cpu.json" "${warmups}" "${runs}" "${queries}"

python3 - "${output_dir}/metal.json" "${output_dir}/cpu.json" "${output_dir}/phases.json" <<'PYTHON'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as source:
    metal = json.load(source)
with open(sys.argv[2], encoding="utf-8") as source:
    cpu = json.load(source)

summary = {}
for name, metal_query in metal.items():
    cpu_query = cpu.get(name, {})
    summary[name] = {
        "cpu_end_to_end_median_ms": cpu_query.get("end_to_end_median_ms"),
        "metal_end_to_end_median_ms": metal_query.get("end_to_end_median_ms"),
        "speedup": (
            cpu_query["end_to_end_median_ms"] / metal_query["end_to_end_median_ms"]
            if cpu_query.get("end_to_end_median_ms") and metal_query.get("end_to_end_median_ms")
            else None
        ),
        "metal": metal_query,
    }

with open(sys.argv[3], "w", encoding="utf-8") as output:
    json.dump(summary, output, indent=2, sort_keys=True)
    output.write("\n")
print(json.dumps({name: {k: v for k, v in entry.items() if k != "metal"}
                  for name, entry in summary.items()}, indent=2, sort_keys=True))
PYTHON

echo "Phase benchmark written under ${output_dir}."
