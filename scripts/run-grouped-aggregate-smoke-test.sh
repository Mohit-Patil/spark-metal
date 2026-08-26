#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
# shellcheck source=project-env.sh
source "${script_dir}/project-env.sh"

# Default grouped-aggregate query: q31 (partial Sum over a date_dim/
# customer_address star-join, keyed on ss_addr_sk/ss_sold_date_sk, grouped on
# dimension attributes). q96: the existing count-only, zero-group-key region
# -- matchRegion rejects this shape outright, so it must keep planning
# MetalParquetMembershipCount, never MetalParquetGroupedAggregate. Both sets
# are run through the SAME CPU/Metal comparison so a single comparison.json
# carries every result_match.
#
# NOT q3/q52 (the task brief's originally-named demonstration queries):
# both join store_sales to item on ss_item_sk, and ss_item_sk is PLAIN-
# encoded -- not dictionary-encoded -- across every one of the 30
# store_sales files in benchmark-data/tpcds-sf10-parquet (verified via
# `scripts/inspect-parquet-encodings.sh "" ss_item_sk`; SF10's item table has
# ~204K distinct values, enough to overflow a row group's dictionary page).
# The native decoder requires join/group keys to be dictionary-encoded
# (MetalParquetGroupedAggregateExec.decodeKeyColumn), so ParquetEligibility.
# check() correctly and permanently rejects q3/q52 against this dataset --
# NonFatal-safe fallback to a vanilla HashAggregateExec, not a bug. This is a
# real data-generation characteristic of the current SF10 Parquet files, not
# a planner defect: ss_addr_sk/ss_store_sk/ss_customer_sk/ss_sold_date_sk are
# all dictionary-encoded across every file, which is why q31 (and other
# non-item-joining star queries) still demonstrate the operator end-to-end.
# Override via TPCDS_QUERIES to re-check q3/q52 once that is addressed
# (regenerating the dataset with a larger dictionary page size, or relaxing
# the native decoder's key-dictionary requirement) -- see task-6-report.md.
grouped_queries="${TPCDS_QUERIES:-q31}"
# q96/q88/q90: all three are the existing count-only, zero-group-key shape
# -- matchRegion rejects this shape outright (before any ParquetEligibility
# work), so they must keep planning the existing membership operators, never
# MetalParquetGroupedAggregate. All three are asserted; per the controller's
# instruction, if one of q88/q90 doesn't actually plan a membership operator
# this script reports what it sees rather than silently forcing it.
boundary_queries="${TPCDS_BOUNDARY_QUERIES:-q96,q88,q90}"
warmups="${TPCDS_WARMUPS:-1}"
runs="${TPCDS_RUNS:-2}"
all_queries="${grouped_queries},${boundary_queries}"

run_id="$(date -u '+%Y%m%dT%H%M%SZ')"
comparison_root="${TPCDS_COMPARISON_DIR:-${repo_root}/benchmark-results/grouped-aggregate-smoke-${run_id}}"
disabled_root="${comparison_root}/metal-disabled"
aqe_root="${comparison_root}/metal-aqe"
mkdir -p "${comparison_root}"

TPCDS_COMPARISON_DIR="${comparison_root}" \
  TPCDS_QUERIES="${all_queries}" \
  TPCDS_WARMUPS="${warmups}" \
  TPCDS_RUNS="${runs}" \
  "${script_dir}/run-tpcds-comparison.sh"

# Flag-off leg (spark.metal.parquetAggregate.enabled=false): only the
# grouped-aggregate queries need rerunning here -- this leg exists to prove
# the new operator disappears (falls back to a vanilla HashAggregateExec)
# when the flag is off, mirroring run-q96-membership-smoke-test.sh's
# metal-disabled leg for spark.metal.parquetScan.enabled.
TPCDS_RESULT_DIR="${disabled_root}" \
  TPCDS_METAL_EXTRA_CONF="spark.metal.parquetAggregate.enabled=false" \
  "${script_dir}/run-tpcds-metal.sh" \
  --queries "${grouped_queries}" --warmups "${warmups}" --runs "${runs}"

# AQE leg (spark.sql.adaptive.enabled=true, overriding run-tpcds-metal.sh's
# own --conf spark.sql.adaptive.enabled=false -- spark-submit's --conf
# processing takes the last value for a repeated key): the grouped-aggregate
# branch is gated on !adaptiveEnabled (same as the membership branch), so
# under AQE the plan must be entirely vanilla (no Metal operator of any
# kind) and the result must still match the CPU baseline.
TPCDS_RESULT_DIR="${aqe_root}" \
  TPCDS_METAL_EXTRA_CONF="spark.sql.adaptive.enabled=true" \
  "${script_dir}/run-tpcds-metal.sh" \
  --queries "${grouped_queries}" --warmups "${warmups}" --runs "${runs}"

python3 - "${comparison_root}/comparison.json" "${comparison_root}/cpu/summary.json" \
  "${comparison_root}/metal" "${disabled_root}" "${aqe_root}" "${grouped_queries}" "${boundary_queries}" <<'PYTHON'
import json
import sys
from pathlib import Path

(comparison_path, cpu_summary_path, metal_dir, disabled_dir, aqe_dir,
 grouped_csv, boundary_csv) = sys.argv[1:8]

with open(comparison_path, encoding="utf-8") as source:
    comparison = json.load(source)
with open(cpu_summary_path, encoding="utf-8") as source:
    cpu_summary = json.load(source)
with open(f"{disabled_dir}/summary.json", encoding="utf-8") as source:
    disabled_summary = json.load(source)
with open(f"{aqe_dir}/summary.json", encoding="utf-8") as source:
    aqe_summary = json.load(source)

metal_dir = Path(metal_dir)
disabled_dir = Path(disabled_dir)
aqe_dir = Path(aqe_dir)
grouped_queries = [name for name in grouped_csv.split(",") if name]
boundary_queries = [name for name in boundary_csv.split(",") if name]

def result_key(summary, name):
    entry = summary["queries"][name]
    return (entry["row_count"], entry["sha256"])

report = {}

for name in grouped_queries:
    result = comparison["queries"].get(name)
    if result is None:
        raise SystemExit(f"{name}: missing from comparison.json")
    if not result["result_match"]:
        raise SystemExit(f"{name}: CPU/Metal result mismatch: {result}")

    plan_text = (metal_dir / f"{name}-plan.txt").read_text(encoding="utf-8")
    if "MetalParquetGroupedAggregate" not in plan_text:
        raise SystemExit(
            f"{name}: expected MetalParquetGroupedAggregate in the (flag-enabled) metal plan, got:\n{plan_text}"
        )

    disabled_plan_text = (disabled_dir / f"{name}-plan.txt").read_text(encoding="utf-8")
    if "Metal" in disabled_plan_text:
        raise SystemExit(
            f"{name}: expected a vanilla plan (no Metal operator) with "
            f"spark.metal.parquetAggregate.enabled=false, got:\n{disabled_plan_text}"
        )
    if result_key(cpu_summary, name) != result_key(disabled_summary, name):
        raise SystemExit(
            f"{name}: flag-disabled result diverges from the CPU baseline: "
            f"cpu={result_key(cpu_summary, name)} disabled={result_key(disabled_summary, name)}"
        )

    aqe_plan_text = (aqe_dir / f"{name}-plan.txt").read_text(encoding="utf-8")
    if "Metal" in aqe_plan_text:
        raise SystemExit(
            f"{name}: expected a vanilla plan (no Metal operator) under "
            f"spark.sql.adaptive.enabled=true, got:\n{aqe_plan_text}"
        )
    if result_key(cpu_summary, name) != result_key(aqe_summary, name):
        raise SystemExit(
            f"{name}: AQE-leg result diverges from the CPU baseline: "
            f"cpu={result_key(cpu_summary, name)} aqe={result_key(aqe_summary, name)}"
        )

    report[name] = {
        "result_match": True,
        "metal_plans_grouped_aggregate": True,
        "disabled_plan_is_vanilla": True,
        "disabled_result_matches_cpu": True,
        "aqe_plan_is_vanilla": True,
        "aqe_result_matches_cpu": True,
    }

for name in boundary_queries:
    result = comparison["queries"].get(name)
    if result is None:
        raise SystemExit(f"{name}: missing from comparison.json")
    if not result["result_match"]:
        raise SystemExit(f"{name}: CPU/Metal result mismatch: {result}")

    plan_text = (metal_dir / f"{name}-plan.txt").read_text(encoding="utf-8")
    if "MetalParquetGroupedAggregate" in plan_text:
        raise SystemExit(
            f"{name}: MetalParquetGroupedAggregate must not plan for a count-only, "
            f"zero-group-key region, got:\n{plan_text}"
        )
    has_membership_operator = (
        "MetalParquetMembershipCount" in plan_text or "MetalFusedMembershipCount" in plan_text
    )
    # Empirically confirmed true for q96/q88/q90 on this dataset (all three
    # plan a membership operator) before this was made a hard assertion --
    # see task-6-report.md's fix-round addendum. A future regression here
    # should fail loudly, not be silently downgraded to a note.
    if not has_membership_operator:
        raise SystemExit(f"{name}: expected a membership-count Metal operator in plan, got:\n{plan_text}")

    report[name] = {
        "result_match": True,
        "metal_plans_membership_count": True,
        "grouped_aggregate_absent": True,
    }

print(json.dumps(report, indent=2, sort_keys=True))
PYTHON

echo "Grouped-aggregate smoke results written under ${comparison_root}."
