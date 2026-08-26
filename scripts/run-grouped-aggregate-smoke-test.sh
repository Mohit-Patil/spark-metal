#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
# shellcheck source=project-env.sh
source "${script_dir}/project-env.sh"

# Winner query: q3 (partial Sum over a date_dim/item star-join, keyed on
# ss_sold_date_sk/ss_item_sk) has exactly one grouped-aggregate region and
# clears the accelerator's default budget (spark.metal.parquetAggregate.
# maxRegions, default 1) -- proof the region-count gate does not over-block a
# single-region query. It is asserted below not just for result_match and
# the operator's presence in the plan, but for cpuFallbackRowGroups == 0.
#
# q3 is the task brief's originally-named demonstration query, and was
# excluded here until Task 6b: it joins store_sales to item on ss_item_sk,
# and ss_item_sk is PLAIN-encoded -- not dictionary-encoded -- across every
# one of the 30 store_sales files in benchmark-data/tpcds-sf10-parquet
# (verified via `scripts/inspect-parquet-encodings.sh "" ss_item_sk`; SF10's
# item table has ~204K distinct values, enough to overflow a row group's
# dictionary page). Task 6 (see task-6-report.md) required join/group keys to
# be dictionary-encoded, so this query permanently fell back to a vanilla
# HashAggregateExec on this dataset. Task 6b (see task-6b-report.md) relaxed
# that requirement to admit PLAIN key chunks via dense VALUE-space code
# tables, recovering q3 (and, per Task 1's ELIGIBLE list, most of the other
# item-keyed queries this tier targets): its cpuFallbackRowGroups == 0 is
# proof the PLAIN ss_item_sk chunks decoded and aggregated entirely on the
# GPU, not via the operator's per-row-group CPU fallback.
winner_queries="${TPCDS_QUERIES:-q3}"
# Grouped queries whose accelerator_metrics must show zero per-row-group CPU
# fallback -- proof a PLAIN key chunk (not just a dictionary one) ran end to
# end on the GPU. See the q3 comment above.
zero_fallback_queries="${TPCDS_ZERO_FALLBACK_QUERIES:-q3}"
# Multi-region query: q31 (six MetalParquetGroupedAggregate regions, per
# Task 7's coverage table -- repeated CTE references over store_sales/
# web_sales multiply the operator's per-region fixed cost until it loses to
# CPU by 0.17x-0.28x). Task 7b's region-count gate must decline ALL of a
# query's grouped regions once the count exceeds
# spark.metal.parquetAggregate.maxRegions (default 1): under the default
# config q31 must plan entirely vanilla (falling through to plain
# HashAggregateExec, since its regions are not the membership operators'
# count-only/zero-group-key shape either) while still matching the CPU
# result exactly, and only with maxRegions raised to 0 (unlimited) does it
# plan MetalParquetGroupedAggregate again -- the original, pre-gate behavior.
multi_region_query="${TPCDS_MULTI_REGION_QUERY:-q31}"
# q96/q88/q90: all three are the existing count-only, zero-group-key shape
# -- matchRegion rejects this shape outright (before any ParquetEligibility
# work), so they must keep planning the existing membership operators, never
# MetalParquetGroupedAggregate, regardless of the region-count gate (which
# only governs the grouped-aggregate branch). All three are asserted; per
# the controller's instruction, if one of q88/q90 doesn't actually plan a
# membership operator this script reports what it sees rather than silently
# forcing it.
boundary_queries="${TPCDS_BOUNDARY_QUERIES:-q96,q88,q90}"
warmups="${TPCDS_WARMUPS:-1}"
runs="${TPCDS_RUNS:-2}"
all_queries="${winner_queries},${multi_region_query},${boundary_queries}"

run_id="$(date -u '+%Y%m%dT%H%M%SZ')"
comparison_root="${TPCDS_COMPARISON_DIR:-${repo_root}/benchmark-results/grouped-aggregate-smoke-${run_id}}"
disabled_root="${comparison_root}/metal-disabled"
aqe_root="${comparison_root}/metal-aqe"
unlimited_root="${comparison_root}/metal-unlimited-regions"
mkdir -p "${comparison_root}"

# Default-config run (spark.metal.parquetAggregate.maxRegions defaults to
# 1): carries the winner leg, the multi-region gate leg, and the boundary
# leg through the SAME CPU/Metal comparison so a single comparison.json
# carries every result_match.
TPCDS_COMPARISON_DIR="${comparison_root}" \
  TPCDS_QUERIES="${all_queries}" \
  TPCDS_WARMUPS="${warmups}" \
  TPCDS_RUNS="${runs}" \
  "${script_dir}/run-tpcds-comparison.sh"

# Flag-off leg (spark.metal.parquetAggregate.enabled=false): only the
# winner query needs rerunning here -- this leg exists to prove the new
# operator disappears (falls back to a vanilla HashAggregateExec) when the
# flag is off, mirroring run-q96-membership-smoke-test.sh's metal-disabled
# leg for spark.metal.parquetScan.enabled.
TPCDS_RESULT_DIR="${disabled_root}" \
  TPCDS_METAL_EXTRA_CONF="spark.metal.parquetAggregate.enabled=false" \
  "${script_dir}/run-tpcds-metal.sh" \
  --queries "${winner_queries}" --warmups "${warmups}" --runs "${runs}"

# AQE leg (spark.sql.adaptive.enabled=true, overriding run-tpcds-metal.sh's
# own --conf spark.sql.adaptive.enabled=false -- spark-submit's --conf
# processing takes the last value for a repeated key): the grouped-aggregate
# branch is gated on !adaptiveEnabled (same as the membership branch), so
# under AQE the plan must be entirely vanilla (no Metal operator of any
# kind) and the result must still match the CPU baseline.
TPCDS_RESULT_DIR="${aqe_root}" \
  TPCDS_METAL_EXTRA_CONF="spark.sql.adaptive.enabled=true" \
  "${script_dir}/run-tpcds-metal.sh" \
  --queries "${winner_queries}" --warmups "${warmups}" --runs "${runs}"

# Unlimited-regions leg (spark.metal.parquetAggregate.maxRegions=0): only
# the multi-region query needs rerunning here -- this is q31's original,
# pre-gate leg (Task 7 planned MetalParquetGroupedAggregate for q31 under
# every config), now reached only by explicitly raising the region budget.
TPCDS_RESULT_DIR="${unlimited_root}" \
  TPCDS_METAL_EXTRA_CONF="spark.metal.parquetAggregate.maxRegions=0" \
  "${script_dir}/run-tpcds-metal.sh" \
  --queries "${multi_region_query}" --warmups "${warmups}" --runs "${runs}"

python3 - "${comparison_root}/comparison.json" "${comparison_root}/cpu/summary.json" \
  "${comparison_root}/metal" "${disabled_root}" "${aqe_root}" "${unlimited_root}" \
  "${winner_queries}" "${multi_region_query}" "${boundary_queries}" "${zero_fallback_queries}" <<'PYTHON'
import json
import sys
from pathlib import Path

(comparison_path, cpu_summary_path, metal_dir, disabled_dir, aqe_dir, unlimited_dir,
 winner_csv, multi_region_csv, boundary_csv, zero_fallback_csv) = sys.argv[1:11]

with open(comparison_path, encoding="utf-8") as source:
    comparison = json.load(source)
with open(cpu_summary_path, encoding="utf-8") as source:
    cpu_summary = json.load(source)
with open(f"{disabled_dir}/summary.json", encoding="utf-8") as source:
    disabled_summary = json.load(source)
with open(f"{aqe_dir}/summary.json", encoding="utf-8") as source:
    aqe_summary = json.load(source)
with open(f"{unlimited_dir}/summary.json", encoding="utf-8") as source:
    unlimited_summary = json.load(source)

metal_dir = Path(metal_dir)
disabled_dir = Path(disabled_dir)
aqe_dir = Path(aqe_dir)
unlimited_dir = Path(unlimited_dir)
winner_queries = [name for name in winner_csv.split(",") if name]
multi_region_queries = [name for name in multi_region_csv.split(",") if name]
boundary_queries = [name for name in boundary_csv.split(",") if name]
zero_fallback_queries = {name for name in zero_fallback_csv.split(",") if name}

def result_key(summary, name):
    entry = summary["queries"][name]
    return (entry["row_count"], entry["sha256"])

report = {}

for name in winner_queries:
    result = comparison["queries"].get(name)
    if result is None:
        raise SystemExit(f"{name}: missing from comparison.json")
    if not result["result_match"]:
        raise SystemExit(f"{name}: CPU/Metal result mismatch: {result}")

    plan_text = (metal_dir / f"{name}-plan.txt").read_text(encoding="utf-8")
    if "MetalParquetGroupedAggregate" not in plan_text:
        raise SystemExit(
            f"{name}: expected MetalParquetGroupedAggregate in the (flag-enabled, default "
            f"maxRegions) metal plan -- the region-count gate should not block a single-region "
            f"winner, got:\n{plan_text}"
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

    if name in zero_fallback_queries:
        # Task 6b: proves a PLAIN-encoded key chunk (e.g. q3's ss_item_sk)
        # decoded and aggregated entirely on the GPU, via the operator's own
        # cpuFallbackRowGroups metric -- not just that the plan carries
        # MetalParquetGroupedAggregate (which a plan running entirely via the
        # per-row-group CPU fallback would also show).
        # nodeName() strips the "Exec" suffix (same reason the plan-text
        # checks above look for "MetalParquetGroupedAggregate", not
        # "...Exec") -- accelerator_metrics is keyed the same way.
        node_metrics = result.get("accelerator_metrics", {}).get("MetalParquetGroupedAggregate", {})
        cpu_fallback_row_groups = node_metrics.get("cpuFallbackRowGroups")
        if cpu_fallback_row_groups != 0:
            raise SystemExit(
                f"{name}: expected cpuFallbackRowGroups == 0 (PLAIN key decode entirely on the GPU), "
                f"got {cpu_fallback_row_groups} (accelerator_metrics={result.get('accelerator_metrics')})"
            )
        report[name]["cpu_fallback_row_groups"] = cpu_fallback_row_groups

for name in multi_region_queries:
    result = comparison["queries"].get(name)
    if result is None:
        raise SystemExit(f"{name}: missing from comparison.json")
    if not result["result_match"]:
        raise SystemExit(f"{name}: CPU/Metal result mismatch: {result}")

    # Task 7b: q31's six regions exceed the default maxRegions=1 budget, so
    # the grouped branch must decline ALL of its regions and the query must
    # fall all the way through to a vanilla (non-Metal) plan -- q31's regions
    # are not the membership operators' shape either, so no other Metal
    # operator should appear.
    plan_text = (metal_dir / f"{name}-plan.txt").read_text(encoding="utf-8")
    if "Metal" in plan_text:
        raise SystemExit(
            f"{name}: expected a vanilla plan (no Metal operator) under the default "
            f"maxRegions=1 budget -- {name} has more than one grouped-aggregate region, got:\n{plan_text}"
        )

    unlimited_plan_text = (unlimited_dir / f"{name}-plan.txt").read_text(encoding="utf-8")
    if "MetalParquetGroupedAggregate" not in unlimited_plan_text:
        raise SystemExit(
            f"{name}: expected MetalParquetGroupedAggregate in the metal plan with "
            f"spark.metal.parquetAggregate.maxRegions=0 (unlimited), got:\n{unlimited_plan_text}"
        )
    if result_key(cpu_summary, name) != result_key(unlimited_summary, name):
        raise SystemExit(
            f"{name}: unlimited-maxRegions result diverges from the CPU baseline: "
            f"cpu={result_key(cpu_summary, name)} unlimited={result_key(unlimited_summary, name)}"
        )

    unlimited_metrics = unlimited_summary["queries"][name].get("accelerator_metrics", {})
    node_metrics = unlimited_metrics.get("MetalParquetGroupedAggregate", {})
    cpu_fallback_row_groups = node_metrics.get("cpuFallbackRowGroups")
    if cpu_fallback_row_groups != 0:
        raise SystemExit(
            f"{name}: expected cpuFallbackRowGroups == 0 under maxRegions=0, "
            f"got {cpu_fallback_row_groups} (accelerator_metrics={unlimited_metrics})"
        )

    report[name] = {
        "result_match": True,
        "default_config_plan_is_vanilla": True,
        "unlimited_max_regions_plans_grouped_aggregate": True,
        "unlimited_max_regions_result_matches_cpu": True,
        "unlimited_max_regions_cpu_fallback_row_groups": cpu_fallback_row_groups,
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
