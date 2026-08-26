# GPU grouped-aggregate spec (star-schema tier)

## Goal

Generalize the accelerated region from "three membership joins + partial
count" to "N broadcast joins + partial grouped aggregate (SUM / COUNT / AVG)"
over an eligible fact scan, so the GPU Parquet decode and membership
machinery built for q96 applies to the broad star-aggregate class of TPC-DS
queries instead of three of them.

## Evidence base (full-suite run, 2026-08-26)

The complete 103-query SF10 comparison
(`benchmark-results/comparison-20260826T124359Z` plus the q95–q99 tail in
`comparison-20260826T141241Z`) established:

- **Correctness:** 103/103 exact result-hash, row-count, and schema matches
  between vanilla Spark and the Metal build.
- **Current coverage:** the accelerator fires on exactly three queries and
  wins on all of them — q96 1.72x, q90 1.60x, q88 1.58x — with zero CPU
  fallbacks. All other plans are untouched (verified: no Metal operators in
  any other executed plan).
- **Addressable pool for this tier:** a plan scan finds **41 queries** whose
  physical shape is broadcast-only joins over one or more fact tables feeding
  a partial HashAggregate with only SUM/COUNT/AVG — including q67 (11.2s CPU
  median), q22 (8.0s), q21 (1.7s), q48, q13, q61, q18, q68, q79, q46, q66,
  q15, q7, q60, q19, q27, q36, q56, q71, q33, q98, q89, q99, q63, q53, q76,
  q26, q43, q20, q86, q62, q45, q3, q91, q52, q12, q55, q42 (38 beyond the
  three already covered).
- Queries requiring sort-merge joins (q14, q64, q95, …) are **out of scope**
  for this tier.

## Key design observations (from the SF10 plans)

1. **Decimal SUM is int64 at the partial level.** Spark plans
   `sum(ss_ext_sales_price)` as `partial_sum(UnscaledValue(col))` — a 64-bit
   sum of unscaled 10^-2 integers. The GPU only needs int64 accumulation;
   scale/overflow handling stays in Spark's final aggregate, which we do not
   replace. `avg` is `partial_avg` = (sum, count) — the same machinery.
2. **Group keys are dimension attributes** (e.g. `d_year, i_brand_id,
   i_brand` in q3), not fact columns. Because every dimension is
   broadcast-small, the join+group can be fused: on the driver, assign each
   *distinct combination of group-key values that survives the dimension
   filters* a dense integer group id, and build per-column lookup tables
   `factKeyValue → (member?, groupContribution)` exactly like today's
   membership tables. The GPU then aggregates into `groups[groupId]` — a
   dense array — and the driver maps group ids back to attribute values.
   String group keys never touch the GPU.
3. **The fact side contributes at most: K int32 join-key columns + M measure
   columns.** Measures are int32/int64/decimal(7,2)-as-int64. The existing
   page decoder already handles int32 dictionary pages; it needs a PLAIN
   int32/int64 page path for measures (measure columns are often not
   dictionary-encoded), plus decimal-as-int64 widening.
4. **Fact-side filters** beyond IsNotNull (e.g. q13/q48 have range predicates
   on fact columns) disqualify a region in the first version, exactly as
   today; the fused-batch operator remains the fallback. A follow-up can
   evaluate simple int predicates in the kernel.

## Architecture

Planner: extend `SparkMetalColumnarRule` with a matcher for
`HashAggregate(partial: sums/counts/avgs, keys: dimension attrs) ← Project ←
BroadcastHashJoin* ← [Filter(IsNotNull)] ← eligible Parquet scan`, where every
join is Inner/BuildRight-or-Left on one int32 fact key, every group key is a
dimension attribute, and every aggregate input is a fact measure column (or
literal 1 for count). Replacement operator: `MetalParquetGroupedAggregateExec`
producing the partial-aggregate output rows (group keys + partial sums/counts)
so Spark's exchange + final aggregate run unchanged.

Execution per (file, rowGroup) split — reusing the existing stream:

1. Driver collects each dimension's (joinKey → groupKeyAttrs) rows, builds the
   dense group-id space (cross product of per-dimension distinct contributions,
   capped — see eligibility), and per-column tables
   `dictId/value → int32 code` where code packs (member bit, per-dimension
   group component); the fact row's group id is a sum/base-multiply of its
   per-column components, computed in the kernel.
2. GPU decodes join-key columns (existing path) and measure columns (new
   PLAIN int32/int64 decoder) into row-group planes.
3. New kernel `fused_grouped_aggregate`: per row — null/membership gate as
   today, compose groupId from per-column codes, then
   `atomic add` (or threadgroup-local then merged) into
   `partials[groupId * numAggs + a]` int64 accumulators.
4. Stream finish returns the dense partial table; the driver emits one
   partial-aggregate InternalRow per non-empty group id, mapping ids back to
   attribute values. Multiplicity semantics (duplicate dimension join keys)
   multiply contributions exactly as today's multiplicity kernel.

## Eligibility (all planner-time, else fused/CPU fallback)

- Everything the current parquet path requires (V1 dictionary pages for the
  join keys, SNAPPY/UNCOMPRESSED, maxRep 0, maxDef ≤ 1), plus PLAIN or
  RLE_DICTIONARY int32/int64 pages for measures.
- Aggregates only `partial_sum(int/long/UnscaledValue)`, `partial_count`,
  `partial_avg` (sum+count).
- Dense group-id space ≤ 2^20 groups × numAggs (dense table ≤ ~64MB);
  larger group domains fall back.
- Broadcast joins only; single fact table per region; no fact-side
  non-IsNotNull filters in v1.

## Verification gates (same discipline as before)

1. Edge dataset extended with duplicate dimension keys, null fact keys, null
   measures, and negative decimals — exact partial-output match against Spark.
2. Full 103-query SF10 suite: exact hash match on all queries, accelerator
   firing recorded per query with zero unexplained fallbacks.
3. Per-query gate: report per-accelerated-query medians honestly; a query
   whose region is replaced but does not beat CPU must be reported as such
   (candidates for a planner cost threshold if that occurs).

## Non-goals

Sort-merge joins, string/date group keys on the GPU (ids only), fact-side
predicates beyond IsNotNull, window functions, rollup/cube grouping sets
(q67's rollup runs in Spark's final aggregate — only its partial is ours),
distributed execution.
