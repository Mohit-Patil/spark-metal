# TPC-DS q96 target

## Why q96 is first

The pinned q96 text has a narrow shape that is valuable for an initial GPU
vertical slice: a large `store_sales` scan, three filtered dimension tables,
three integer equi-joins, and a global `count(*)`. The dimension predicates are
selective and their keys are suitable for broadcast-style membership maps. The
fact path otherwise crosses into rows and probes three CPU hash tables.

The proposed replacement keeps dimension filtering in Spark, collects only the
three filtered integer key columns, and replaces the fact-side joins plus
partial count with one columnar operator. It does not rewrite the SQL text.

```text
Vanilla Spark
Parquet columnar scan -> rows -> hash join -> hash join -> hash join -> partial count

Spark Metal
Parquet columnar scan -> fused GPU membership x3 + partial count
```

## Current capability contract

The rule activates only when all of these are true:

- exactly three inner broadcast hash joins, each building the right side;
- one integer equi-key per join and three distinct fact attributes;
- no residual join condition;
- a global, non-distinct partial `count(1)` with no aggregate filter;
- intervening projections contain only attributes;
- intervening filters contain only conjunctions of `is not null` checks;
- adaptive query execution is disabled.

The Metal path preserves null non-matches and build-side duplicate
multiplicity. Unsupported physical plans remain unchanged. At execution time,
on-heap columns and overly wide dense key domains use an exact CPU fallback.

## Evidence before the licensed run

The exact q96 SQL text runs against a generated q96-shaped dataset with
33,554,432 fact rows. With five warm-ups and eleven measured observations on the
first host, CPU median time is 201.910 ms and Metal median time is 178.462 ms,
for a 1.13x end-to-end speedup and identical result.

This is a plan-shape and crossover experiment, not a TPC-DS benchmark result.
The project goal is satisfied only after the same mechanism runs an unmodified
q96 against scale-factor-10 TPC-DS Parquet data, matches the controlled CPU
result, appears in the captured physical plan, and sustains at least a 10%
median improvement.

## Reproduction

The synthetic check does not require the TPC-DS kit:

```bash
scripts/run-q96-membership-smoke-test.sh
Q96_SYNTHETIC_WARMUPS=5 Q96_SYNTHETIC_RUNS=11 scripts/run-q96-synthetic-benchmark.sh
```

The official local comparison requires prior explicit acceptance of the kit
licence and generated scale-factor-10 Parquet data:

```bash
scripts/run-tpcds-comparison.sh --queries q96 --warmups 2 --runs 7
```
