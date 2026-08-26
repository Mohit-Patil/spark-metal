#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
run_id="$(date -u '+%Y%m%dT%H%M%SZ')"
comparison_root="${TPCDS_COMPARISON_DIR:-${repo_root}/benchmark-results/comparison-${run_id}}"
cpu_root="${comparison_root}/cpu"
metal_root="${comparison_root}/metal"
queries="${TPCDS_QUERIES:-q96}"
warmups="${TPCDS_WARMUPS:-1}"
runs="${TPCDS_RUNS:-5}"

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
      echo "Unknown comparison option: $1" >&2
      exit 2
      ;;
  esac
done
mkdir -p "${comparison_root}"

TPCDS_RESULT_DIR="${cpu_root}" "${script_dir}/run-tpcds-cpu.sh" \
  --queries "${queries}" --warmups "${warmups}" --runs "${runs}"
TPCDS_RESULT_DIR="${metal_root}" "${script_dir}/run-tpcds-metal.sh" \
  --queries "${queries}" --warmups "${warmups}" --runs "${runs}"

python3 - "${cpu_root}/summary.json" "${metal_root}/summary.json" \
  "${comparison_root}/comparison.json" <<'PYTHON'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as source:
    cpu = json.load(source)
with open(sys.argv[2], encoding="utf-8") as source:
    metal = json.load(source)

comparison = {"queries": {}, "all_results_match": True, "success_gate_met": False}
for name, cpu_result in cpu["queries"].items():
    metal_result = metal["queries"].get(name)
    if metal_result is None:
        raise SystemExit(f"Metal result is missing query {name}")
    matches = (
        cpu_result["row_count"] == metal_result["row_count"]
        and cpu_result["sha256"] == metal_result["sha256"]
        and cpu_result["schema_json"] == metal_result["schema_json"]
    )
    speedup = cpu_result["median_seconds"] / metal_result["median_seconds"]
    comparison["queries"][name] = {
        "cpu_median_seconds": cpu_result["median_seconds"],
        "metal_median_seconds": metal_result["median_seconds"],
        "speedup": speedup,
        "result_match": matches,
        # Either membership-count operator counts: the GPU Parquet scan
        # (MetalParquetMembershipCount) replaces the fused one wherever the
        # fact-side scan is eligible, so hardcoding the fused name reported a
        # false negative for every accelerated plan that used the Parquet path.
        "metal_operator_present": any(
            operator in open(
                f"{sys.argv[2].rsplit('/', 1)[0]}/{name}-plan.txt", encoding="utf-8"
            ).read()
            for operator in ("MetalFusedMembershipCount", "MetalParquetMembershipCount")
        ),
        "accelerator_metrics": metal_result.get("accelerator_metrics", {}),
    }
    comparison["all_results_match"] &= matches

comparison["success_gate_met"] = any(
    result["result_match"] and result["metal_operator_present"] and result["speedup"] >= 1.10
    for result in comparison["queries"].values()
)
with open(sys.argv[3], "w", encoding="utf-8") as output:
    json.dump(comparison, output, indent=2, sort_keys=True)
    output.write("\n")
print(json.dumps(comparison, indent=2, sort_keys=True))
PYTHON

echo "Comparison written under ${comparison_root}."
