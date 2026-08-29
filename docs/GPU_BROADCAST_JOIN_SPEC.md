# GPU broadcast-join spec (columnar-output tier)

## Goal

Add a third accelerated tier that replaces "eligible Parquet fact scan + N
inner broadcast hash joins" with a GPU operator that emits **columnar
batches** — joined fact columns plus dimension attributes — to whatever
operator sits above it. The two existing tiers only produce reductions (a
count scalar; grouped-aggregate partials), which caps them at regions ending
in a supported partial aggregate. Columnar output is the missing
infrastructure for every future tier (fact-to-fact joins, windows), and
scan + broadcast star join is the most common heavy pattern in real Spark
workloads — more common than this suite shows, since production plans with
AQE and realistic broadcast thresholds convert many sort-merge joins into
broadcast joins.

## Evidence base (full-suite run, 2026-08-29)

`benchmark-results/comparison-20260829T050219Z` (103/103 exact matches) plus
a plan-shape scan of every executed plan:

- **Direct pool:** 22 not-yet-accelerated queries are pure broadcast-join
  shapes (no sort-merge join, window, rollup/Expand, or nested-loop join):
  q13 (2.30s), q48 (2.26s), q21 (2.00s), q66, q26, q7, q60, q46, q56, q68,
  q71, q15, q33, q79, q76, q19, q99, q45, q43, q62, q91, q41 — **28.0s,
  7.7% of suite time**. The pool is wide but shallow: no member exceeds
  2.3s, so the suite-level ceiling of this tier alone is modest and is
  stated as such up front.
- **Queries containing sort-merge joins hold 80% of suite time.** They are
  out of scope for this tier, except that an eligible scan+broadcast-join
  subregion feeding a sort-merge join may still be replaced (the sort/merge
  itself stays on CPU).
- The 2026-08-29 phase benchmark (`scripts/run-phase-benchmark.sh`)
  established that decode-side CPU costs are already characterized
  (native parse ~130ms per store_sales-scale region, staging ~40ms,
  encode ~90ms) and that multi-region concurrency, not kernel time, is
  what sinks ungated UNION queries — so this tier keeps the same
  one-region-at-a-time economics that make the current winners win.

## Region shape and eligibility (v1)

A replaced region is:

    [Project]? <- BroadcastHashJoin(inner) x N <- [Filter IsNotNull]? <- eligible Parquet fact scan

- Joins are **inner**, keyed on a single int32 fact column per dimension,
  at most 4 dimensions (kernel key cap, as today).
- The fact scan passes the existing `ParquetEligibility` checks (bucketless,
  unpartitioned, V1 dictionary/PLAIN int32 chunks for every consumed fact
  column), under the existing ANSI-off / AQE-off constraints.
- Every output column of the region is either (a) an int32-decodable fact
  column, or (b) a dimension attribute whose type
  `GroupSpace.isSupportedAttributeType` accepts.
- Every dimension's join-key domain must fit the value-space guard
  (`[0, MaxValueSpaceKey]`), checked at runtime exactly as in the
  grouped-aggregate tier.
- **Unique build-side join keys.** A duplicate dimension key fans a fact row
  out into several output rows; v1 does not replicate rows on the GPU.
  Duplicate keys are a runtime data condition and route the whole operator
  to its CPU fallback, exactly like `GroupSpace.build` rejections today.
- Config: `spark.metal.parquetJoin.enabled`, **default false** until the
  success gate below is met on this hardware.

Out of scope for v1 (explicit non-goals): outer/semi/anti joins, duplicate
build keys, fact-side predicates beyond IsNotNull, string/decimal fact
columns, sort-merge joins, AQE.

## Architecture

Reused unchanged: `ParquetEligibility` (+ its memoised footers/splits), the
dimension-collect machinery with the 2026-08-29 execution-scoped caches
(`collectedSubplanCache`), the streamed GPU Parquet page decoder
(`parquetDecodePage` / `parquetDecodeMeasurePage`, staging pool, per-stream
sub-timers), and the CPU row-group fallback reader pattern.

New pieces:

1. **Planner** (`SparkMetalColumnarRule`): match the region above; collect
   per-dimension keyPlans `(joinKey, outputAttrs...)` like the
   grouped-aggregate matcher, but keep the region's full output schema and
   the mapping of each output column to (fact column | dimension, attribute
   ordinal).
2. **Driver prepare** (per region, cached per execution like
   `preparedRegionCache`): collect dimensions once; build per-dimension
   dense **value-space row-index tables** `key -> build-row index`
   (`-1` = non-member) — the same construction as the grouped tier's code
   tables with the premultiplied group code replaced by the dimension's own
   row index — plus the duplicate-key check; broadcast the tables.
3. **Probe and compaction — staged.** The GPU's measured leverage is the
   Parquet decode (phase benchmark, 2026-08-29: native parse/encode
   dominate the accelerated stages; a table probe is one array read per
   row, ~1ms per million rows on CPU). v1 therefore runs the existing GPU
   page decode, reads the decoded key and fact planes back through the
   existing `parquetRowGroupRead` / `parquetRowGroupReadMeasure` surface,
   and fuses probe + compaction + gather into the single CPU pass that
   builds the output vectors (which must walk every surviving row anyway).
   A `fused_join_compact` kernel (GPU probe, prefix-sum compaction,
   scatter) is a v1.1 follow-up, justified only if the extended phase
   benchmark shows the CPU probe/compact pass as a first-order term.
4. **Output** (`supportsColumnar = true`): per row group, the JVM wraps the
   compacted fact columns into off-heap `ColumnVector`s and CPU-gathers each
   dimension-attribute output column from the collected dimension rows by
   build-row index (attributes are arbitrary supported types — strings and
   decimals never touch the GPU, exactly like group tuples today). One
   `ColumnarBatch` per row group, capped by the reader batch size.
5. **Fallbacks**: a page/row group the native decoder rejects recomputes on
   CPU via parquet-mr (row-by-row probe of the same tables, emitting the
   same batch layout); duplicate build keys or a domain violation route the
   whole operator to that CPU path for every row group. `numMetalCommands`
   remains the proof of GPU execution.
6. **Metrics**: the grouped tier's full set (decode umbrella + native
   staging/parse/encode split, row-group read, page submit, output build)
   plus `numOutputBatches` and per-region survival counts.

## Verification and success gate

- Full-suite `run-tpcds-comparison.sh --queries all`: **103/103 exact
  result-hash, row-count, and schema matches** with the tier enabled, zero
  silent fallbacks on accelerated queries, no regression on the 14 queries
  the existing tiers accelerate.
- Phase benchmark extended to price the new operator's stages standalone.
- Gate to flip the default to true: **>= 1.10x median end-to-end speedup on
  at least 3 pool queries under all three measurement protocols** (batched
  strict, isolated rerun, winners-only), per `docs/BENCHMARK_PROTOCOL.md`.
- Staged delivery: the CPU-only operator lands first (planner + columnar
  output + fallback path, verified 103/103) before any kernel work, so
  correctness plumbing is never entangled with GPU debugging.

## Outcome (2026-08-29)

v1 shipped correct — 103/103 exact matches with the tier enabled, firing on
40 queries with zero CPU fallbacks — but the speed gate was NOT met: under
the warmed batched-strict protocol every candidate lost (0.37–0.70x,
`comparison-20260829T075044Z`); cold-run wins were JIT-warmup artifacts.
`spark.metal.parquetJoin.enabled` therefore remains false. The measured
causes and the v1.1 plan (fused probe+compact kernel, selectivity-aware
gate) are recorded in docs/ROADMAP.md M6.
