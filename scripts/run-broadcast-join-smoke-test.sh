#!/usr/bin/env bash

# Correctness smoke for the columnar broadcast-join tier: runs a query set
# once with the plugin + spark.metal.parquetJoin.enabled=true and once with
# vanilla Spark, then requires (a) identical result hashes / row counts /
# schemas and (b) a MetalParquetBroadcastJoin operator in every Metal-leg
# plan. The other tiers stay enabled in the Metal leg, so tier precedence
# (aggregate/membership first, join second) is exercised too.

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
# shellcheck source=project-env.sh
source "${script_dir}/project-env.sh"

data_dir="${TPCDS_PARQUET_DIR:-${repo_root}/benchmark-data/tpcds-sf10-parquet}"
queries_dir="${repo_root}/.tools/spark-assets/sql/core/src/test/resources/tpcds"
run_id="$(date -u '+%Y%m%dT%H%M%SZ')"
output_dir="${JOIN_SMOKE_RESULT_DIR:-${repo_root}/benchmark-results/broadcast-join-smoke-${run_id}}"
queries="${JOIN_SMOKE_QUERIES:-q97,q21,q22,q62,q99,q43}"
# Expect a cpuFallbackRowGroups/numRowGroups relation per stage of the plan:
#   all  -> every row group recomputed on CPU (Task 2, no GPU path yet)
#   none -> zero CPU fallbacks (Task 4, GPU decode path wired)
expect_fallback="${JOIN_SMOKE_EXPECT_FALLBACK:-all}"

if [[ ! -d "${data_dir}" ]]; then
  echo "TPC-DS SF10 Parquet data is missing: ${data_dir}" >&2
  exit 1
fi

"${script_dir}/build-spark-plugin.sh"
build_root="${repo_root}/build/spark-plugin"
mkdir -p "${output_dir}"

TPCDS_RESULT_DIR="${output_dir}/vanilla" "${script_dir}/run-tpcds-cpu.sh" \
  --queries "${queries}" --warmups 0 --runs 1
TPCDS_METAL_EXTRA_CONF="spark.metal.parquetJoin.enabled=true" \
  TPCDS_RESULT_DIR="${output_dir}/metal" "${script_dir}/run-tpcds-metal.sh" \
  --queries "${queries}" --warmups 0 --runs 1

python3 - "${output_dir}" "${expect_fallback}" <<'PYTHON'
import json
import sys
from pathlib import Path

output_dir = Path(sys.argv[1])
expect_fallback = sys.argv[2]
with open(output_dir / "vanilla" / "summary.json", encoding="utf-8") as source:
    vanilla = json.load(source)["queries"]
with open(output_dir / "metal" / "summary.json", encoding="utf-8") as source:
    metal = json.load(source)["queries"]

failures = []
for name, vanilla_result in vanilla.items():
    metal_result = metal[name]
    for field in ("row_count", "sha256", "schema_json"):
        if vanilla_result[field] != metal_result[field]:
            failures.append(f"{name}: {field} mismatch")
    plan = (output_dir / "metal" / f"{name}-plan.txt").read_text(encoding="utf-8")
    if "MetalParquetBroadcastJoin" not in plan:
        failures.append(f"{name}: no MetalParquetBroadcastJoin operator in the Metal plan")
    join_metrics = [
        metrics for node, metrics in metal_result.get("accelerator_metrics", {}).items()
        if node.startswith("MetalParquetBroadcastJoin")
    ]
    if not join_metrics:
        failures.append(f"{name}: no MetalParquetBroadcastJoin metrics harvested")
    for metrics in join_metrics:
        fallbacks = metrics.get("cpuFallbackRowGroups", -1)
        row_groups = metrics.get("numRowGroups", -2)
        if expect_fallback == "all" and fallbacks != row_groups:
            failures.append(
                f"{name}: expected every row group on CPU, got {fallbacks}/{row_groups}")
        if expect_fallback == "none" and fallbacks != 0:
            failures.append(f"{name}: expected zero CPU fallbacks, got {fallbacks}/{row_groups}")

if failures:
    raise SystemExit("BROADCAST JOIN SMOKE FAILED:\n  " + "\n  ".join(failures))
print(f"broadcast-join smoke OK: {len(vanilla)} queries match, "
      f"operator present, fallback expectation '{expect_fallback}' holds")
PYTHON

echo "Broadcast-join smoke results under ${output_dir}."
