# Prototype results

## GPU grouped aggregation across the TPC-DS SF10 suite (2026-08-27)

`MetalParquetGroupedAggregateExec` generalizes the accelerated region from
"three membership joins + partial `count(*)`" to "N broadcast joins + partial
grouped aggregate (SUM/COUNT/AVG)" over an eligible Parquet fact scan. It is
controlled by `spark.metal.parquetAggregate.enabled` (default true) and, like
the membership operators, declines under ANSI mode and adaptive execution.

### Suite outcome

The full 103-query comparison
(`benchmark-results/task7-full`, one warm-up, three measured runs, both legs in
one JVM each) completed without a wedge:

- **103/103 exact result-hash, row-count, and schema matches**
  (`all_results_match: true`).
- The accelerator now fires on **24 of 103 queries**: 21 through
  `MetalParquetGroupedAggregate` and the pre-existing 3 through
  `MetalParquetMembershipCount`.
- **`cpuFallbackRowGroups = 0` on every one of the 24.** No row group in the
  whole suite fell back to CPU, and no whole-operator fallback fired.
- Four `OutOfMemoryError`s occurred (one in the CPU leg at q64, three in the
  Metal leg around q10 and q95), all retried successfully by Spark with exact
  final results. Unlike the 2026-08-26 run, neither leg wedged; the documented
  q14/q64/q95 6 GB heap hazard remains real but did not require a tail rerun.

### Correctness is unambiguous; performance is not

This is the honest headline: **the operator is exactly correct everywhere it
fires and faster than Spark on only a handful of queries.** Measured three
ways — the suite leg, a 24-query strict batch (5 warm-ups, 11 runs,
`benchmark-results/task7-strict`), and an isolated 8-query strict batch
(`benchmark-results/task7-winners`) — only **q53, q63, and q89 clear the 1.10x
gate under every protocol.** Everything else sits at parity or loses, several
of them badly.

The three protocols disagree by more than the effect being measured for the
near-parity queries, so all three are shown rather than the most flattering
one. The spread has a measured cause: the same untouched
`MetalParquetMembershipCount` operator ran q96 in 122 ms inside the full suite,
132 ms in an isolated 3-query batch, and 188 ms inside the back-to-back
24-query batch — a 43% swing from run context alone on this fanless M5 Air.
Sustained GPU batches throttle; the CPU leg throttles less.

### Coverage table — every query where the accelerator fired

`regions` counts `MetalParquetGroupedAggregate` nodes in the executed plan.
`metalTime` and `decodeParseTime` are the strict batch's accumulators summed
across tasks; for multi-region queries the harness records only one node's
metrics (its `accelerator_metrics` dict is keyed by node name), so those two
columns understate total accelerator work there.

| Query | regions | suite 1/3 | strict 24q 5/11 | isolated 5/11 | cpuFallbackRowGroups | metalTime | decodeParseTime |
|---|---:|---:|---:|---:|---:|---:|---:|
| q89 | 1 | 1.52x | 1.32x | 1.13x | 0 | 72 ms | 583 ms |
| q53 | 1 | 1.26x | 1.30x | 1.16x | 0 | 60 ms | 376 ms |
| q70 | 1 | 1.03x | 1.23x | 0.90x | 0 | 71 ms | 338 ms |
| q63 | 1 | 1.27x | 1.17x | 1.14x | 0 | 234 ms | 930 ms |
| q55 | 1 | 1.37x | 1.11x | 0.93x | 0 | 30 ms | 314 ms |
| q52 | 1 | 1.39x | 0.99x | 0.95x | 0 | 27 ms | 318 ms |
| q77 | 5 | 0.50x | 0.94x | — | 0 | 46 ms | 1127 ms |
| q98 | 1 | 0.70x | 0.75x | — | 0 | 28 ms | 676 ms |
| q3 | 1 | 1.03x | 0.74x | 1.05x | 0 | 62 ms | 368 ms |
| q20 | 1 | 0.64x | 0.59x | — | 0 | 20 ms | 93 ms |
| q12 | 1 | 0.51x | 0.57x | — | 0 | 20 ms | 71 ms |
| q47 | 3 | 0.54x | 0.44x | — | 0 | 46 ms | 523 ms |
| q57 | 3 | 0.47x | 0.44x | — | 0 | 48 ms | 411 ms |
| q42 | 1 | 0.93x | 0.35x | 1.07x | 0 | 37 ms | 2437 ms |
| q58 | 3 | 0.43x | 0.33x | — | 0 | 128 ms | 957 ms |
| q56 | 3 | 0.41x | 0.31x | — | 0 | 36 ms | 1334 ms |
| q60 | 3 | 0.24x | 0.29x | — | 0 | 38 ms | 1913 ms |
| q83 | 3 | 0.17x | 0.19x | — | 0 | 73 ms | 82 ms |
| q31 | 6 | 0.28x | 0.17x | — | 0 | 53 ms | 1846 ms |
| q33 | 3 | 0.21x | 0.15x | — | 0 | 56 ms | 4208 ms |
| q74 | 4 | 0.11x | 0.09x | — | 0 | 133 ms | 4861 ms |

**Every query with three or more accelerated regions loses, without
exception**, and the loss deepens with region count (q31, six regions, 0.17x).
Every query that wins has exactly one. The operator's per-region fixed cost —
driver-side dimension collection, group-space construction, split planning, and
row-at-a-time materialization of the partial-aggregate output — is paid once
per region, and TPC-DS's `UNION ALL` over `store_sales`/`catalog_sales`/
`web_sales` (q33, q56, q60, q83) or repeated CTE references (q31, q74)
multiply it. Spark's CPU plan shares far more work across those regions than
this operator does.

The second failure mode is group-space size. q74 (500,000 groups, 449,790
output rows) is the worst result in the suite at 0.09x: emitting nearly half a
million partial-aggregate `InternalRow`s one at a time costs more than the
entire join and aggregate it replaced.

**These are planner-cost-threshold candidates, not tuning targets.** The
operator should decline a region when the plan already contains other
accelerated regions over the same fact tables, and when the estimated group
space is large relative to the fact scan. That gate does not exist yet.

### The bottleneck is not the GPU, so the ledgered kernel optimization was not taken

The plan carried one ledgered optimization for this tier: threadgroup-local
pre-aggregation ahead of the global atomics, to be implemented only if a fired
query was slower than CPU *and* profiling attributed it to atomic contention.
The first condition holds; the second does not, so it was not implemented.

Across all 21 accelerated queries `metalTime` ranges from **20 ms to 234 ms**
while `decodeParseTime` — CPU-side Parquet page parsing inside the JNI bridge —
ranges from 71 ms to 4,861 ms. On the worst query, q74, the GPU kernel accounts
for 133 ms of a 20,400 ms execution: **0.65%**. Atomic contention cannot explain
any of these losses, and a faster kernel would not move them. q74 also has by
far the largest group space (500,000), which is precisely where atomic
contention would show up if it were the problem — and it does not.

### Membership operators: absolute times held, ratios did not

q96/q88/q90 were untouched by this tier and are boundary-asserted (the
count-only, zero-group-key shape is rejected by the grouped matcher by design,
verified in `run-grouped-aggregate-smoke-test.sh`). Their recorded results were
q96 1.72x, q90 1.60x, q88 1.58x. Re-measured in isolation
(`benchmark-results/task7-membership`, 5 warm-ups, 11 runs):

| Query | Recorded CPU → Metal | Recorded | Now CPU → Metal | Now |
|---|---|---:|---|---:|
| q96 | 211.2 → 122.9 ms | 1.72x | 180.5 → 132.5 ms | 1.36x |
| q88 | 1710.2 → 1080.2 ms | 1.58x | 1215.2 → 827.1 ms | 1.47x |
| q90 | 197.7 → 123.3 ms | 1.60x | 160.8 → 118.5 ms | 1.36x |

**The ratios fell, but the Metal operator did not get slower.** q88's Metal
median improved 23% (1080 → 827 ms) and q90's improved 4%; q96's 132.5 ms sits
inside its own historical 122.9–153.8 ms spread across six same-code runs. Both
legs got faster and the CPU leg got faster by more, partly because the earlier
figures used one warm-up and these use five — additional warm-ups favour
Spark's JIT-compiled CPU path. Reported as a ratio regression against the
recorded numbers, with the absolute evidence that the operator itself is
unchanged.

### Eligibility reconciliation — all 25 Task-1-eligible queries accounted for

The planning-time shape probe (`scripts/inspect-grouped-aggregates.sh`)
classified 25 of 103 queries as eligible on shape alone. **21 fired; the other
4 were rejected by a documented planner cap**, re-confirmed by re-running the
probe against the shipped build:

| Query | Probe shape | Fired? | Reason |
|---|---|---|---|
| q3, q12, q20, q31, q33, q42, q47, q52, q53, q55, q56, q57, q58, q60, q63, q70, q74, q77, q83, q89, q98 | — | yes | — |
| q7 | joins=4 groups=1 aggs=[avg×4] | no | Internal aggregate slot cap: 4 `avg`s cost 2 slots each plus 1 occupancy slot = 9 > 8 (`buildGroupedAggregateExec`) |
| q26 | joins=4 groups=1 aggs=[avg×4] | no | Same slot cap: 9 > 8 |
| q61 | joins=6 groups=0 aggs=[sum] | no | Kernel key cap: 6 joins > 4 |
| q91 | joins=6 groups=5 aggs=[sum] | no | Kernel key cap: 6 joins > 4 |

No eligible query was lost to an encoding rejection, a group-space cap, or a
build reject — the group-space and encoding paths accepted every region they
were offered, which is why `cpuFallbackRowGroups` is 0 across the board. The
four misses are both deterministic and shape-only: they do not depend on the
data, and raising either cap is a bounded kernel change rather than a
correctness question.

### Spec corrections found by implementation

Two claims in `GPU_GROUPED_AGGREGATE_SPEC.md` were wrong and are corrected
here rather than quietly dropped.

1. **q67 is not addressable, and it was the largest prize in the spec.** The
   spec's non-goals said "q67's rollup runs in Spark's final aggregate — only
   its partial is ours", and its evidence section led the addressable pool with
   q67 at an 11.2 s CPU median. Both are wrong: Spark places the `Expand` node
   for `GROUP BY ROLLUP` *below* the partial aggregate, between it and the
   scan, so the rollup expansion is inside the region this tier would replace,
   not above it. Every rollup/cube query (q5, q14a, q18, q22, q27, q36, q67,
   q80, q86) is rejected for this reason. The spec's "41 addressable queries"
   figure was a plan-shape estimate that did not model attribute lineage; the
   probe that did model it found 25, and 21 of those fired.

2. **Duplicate dimension join keys cannot be handled by a multiplicity
   factor when the dimension carries group-key attributes.** The spec said
   multiplicity semantics "multiply contributions exactly as today's
   multiplicity kernel". That is only valid for an attribute-free dimension.
   If an attributed dimension has two rows with the same join key but
   different attribute tuples, a matching fact row must fan out into *several
   distinct groups* — an effect no scalar multiplier on a single group id can
   express. `GroupSpace.build` therefore rejects duplicate keys in an
   attributed dimension, and the operator takes a whole-operator CPU fallback
   that performs a real hash join with Cartesian fan-out. This is validated
   against Spark's own join result by the
   `whole-operator-cpu-duplicate-key` case in
   `scripts/run-parquet-decode-smoke-test.sh`.

### The `ss_item_sk` discovery and the value-space answer

Planner integration initially excluded every item-keyed query — q3, q42, q52,
q55 and others — because `ParquetEligibility.check` requires join-key columns
to be dictionary-encoded, and **`ss_item_sk` is PLAIN-encoded in all 30 SF10
`store_sales` files**. SF10's `item` table has roughly 204,000 distinct values —
enough to overflow a row group's dictionary page and force parquet-mr to fall
back to PLAIN for that column across the whole table — so the single
most-joined fact column in TPC-DS never gets a dictionary page at this scale
factor. The key-column decode path in the JNI
bridge hard-required dictionary framing: it always parsed the value section as
an RLE/bit-packed id stream, and would have silently misread a PLAIN page's raw
packed int32s.

The fix was to decode a PLAIN key chunk into a dense **value-space** code table
— indexed by the raw key value rather than by dictionary id — leaving the
kernel completely unchanged, since it only ever reads "code at index". The
dictionary case remains a dictionary-id-space table; only the table's index
domain differs. A PLAIN key carries one extra runtime obligation the
planning-time check cannot see: the dimension's join-key domain must fit the
value-space table's bound, enforced by a driver-side domain guard once the
dimension rows are collected. This recovered q3, q42, q52 and q55 — including
q52, q55 and q42, three of the queries that come closest to the gate.

### Bottom line

The grouped-aggregate tier is a **correctness success and a performance
disappointment.** It extends exact GPU execution from 3 queries to 24 with zero
CPU fallbacks and zero result mismatches across 103 queries, which is the
harder half of the problem. But it beats Spark on three queries (q53 1.16x,
q63 1.14x, q89 1.13x under the most conservative protocol available for each);
of the remaining 18, sixteen are below parity in the 24-query strict batch and
the worst is 11x slower than CPU. `spark.metal.parquetAggregate.enabled` defaults to
true and, on this evidence, should not: without a planner cost threshold that
declines multi-region and large-group-space plans, enabling this operator makes
the median accelerated TPC-DS query slower, not faster. The next unit of work
is that threshold, not a faster kernel — the kernel is already 1% of the time.

## Full TPC-DS SF10 suite validation (2026-08-26)

All 103 pinned Spark TPC-DS queries were run through both configurations
(one warm-up, three measured runs; `benchmark-results/comparison-20260826T124359Z`
for q1–q94 and `comparison-20260826T141241Z` for q95–q99 after a Metal-leg
JVM wedge — see below).

- **103/103 queries produce exact result-hash, row-count, and schema matches.**
- The accelerator fires on exactly three queries and wins on all three with
  zero CPU fallbacks: **q96 1.72x, q90 1.60x, q88 1.58x**. q88 — which lost
  to CPU under the earlier fused operator — clears the gate under GPU Parquet
  decode. No other executed plan contains a Metal operator; the remaining
  queries' timing deltas scatter symmetrically within run-to-run noise.
- One `OutOfMemoryError` occurred during the Metal leg's q14a and was retried
  successfully by Spark (final result exact); the long-lived Metal-leg JVM
  later wedged at q95 after 94 queries. Analysis attributes both to the 6 GB
  local-mode heap running at its ceiling on TPC-DS's three heaviest
  sort-merge-join queries (q14, q64, q95 — the only queries showing heap
  warnings), aggravated by the Metal leg always running second on a
  memory-degraded host: the failing stacks are entirely vanilla Spark sorter
  allocations, q14a's plans are byte-identical across configurations, and the
  pressure began before any accelerated query had executed. Suite runs should
  use an 8 GB driver or per-query JVMs for those three queries.
- A plan-shape scan of the suite identifies **41 broadcast-only star-aggregate
  queries** (SUM/COUNT/AVG partials over fact scans, led by q67 at 11.2 s and
  q22 at 8.0 s) as the addressable pool for the next accelerator tier; see
  `GPU_GROUPED_AGGREGATE_SPEC.md`.

Captured on the first Apple M5 host on 2026-08-26. These results are feasibility
measurements, not TPC-DS results and not comparable to official TPC benchmark
results.

## Native fused operation

Workload:

```text
SUM(IF(value > 100, value * 3 + 7, 0))
```

Each configuration used two warm-ups and seven measured observations. The GPU
time below includes an entire input copy into Metal shared memory, command
encoding, synchronization, and final partial-sum merge.

| Rows | CPU median | Metal copy-inclusive median | Speedup |
|---:|---:|---:|---:|
| 65,536 | 0.159 ms | 0.295 ms | 0.54x |
| 262,144 | 0.653 ms | 0.337 ms | 1.94x |
| 1,048,576 | 2.935 ms | 0.684 ms | 4.29x |
| 4,194,304 | 10.731 ms | 1.292 ms | 8.31x |
| 8,388,608 | 20.759 ms | 2.277 ms | 9.12x |

The crossover for this standalone workload lies between 65,536 and 262,144 rows.

This speedup is relative to the simple scalar-style Swift CPU reference in this
prototype. The MLX results below show that a compiled/vectorized CPU
implementation is substantially stronger, so these native numbers establish
Metal feasibility but are not the project's final CPU comparison.

## MLX CPU/GPU comparison

MLX 0.32.1 executes the same input and exact integer expression using a compiled
graph. Each configuration used five warm-ups and 25 observations.

| Rows | MLX CPU | MLX GPU, input ready | MLX GPU, materialization included | Ready GPU speedup |
|---:|---:|---:|---:|---:|
| 65,536 | 0.028 ms | 0.566 ms | 0.262 ms | 0.05x |
| 262,144 | 0.070 ms | 0.301 ms | 0.302 ms | 0.23x |
| 1,048,576 | 0.238 ms | 0.445 ms | 0.422 ms | 0.54x |
| 4,194,304 | 0.999 ms | 1.099 ms | 1.367 ms | 0.91x |
| 8,388,608 | 1.923 ms | 1.978 ms | 2.204 ms | 0.97x |

There is no MLX GPU crossover in the tested range. Materialization timings can
occasionally be lower than input-ready timings because the observations are
independent and Apple GPU scheduling is noisy; the medians should not be
algebraically combined.

## Core ML capability result

A Core ML 9.0 ML Program can represent the `int32` form of the expression, and
its 4,194,304-row compute plan supports CPU and GPU and prefers GPU. None of the
operators supports the Neural Engine. Core ML cannot cast the intermediate to
`int64`, so the graph returns -1,166,372,896 instead of the required
3,128,594,400. It is therefore rejected as a Spark SQL backend for this vertical
slice regardless of its execution time.

## Spark SQL synthetic comparison

The Spark test reads 33,554,432 integers from the same Parquet dataset in two
separate local Spark processes. Both use eight local cores, off-heap column
vectors, one-million-row reader batches, five warm-ups, and eleven measured
runs. The only execution difference is the registered Metal columnar rule.

| Configuration | Median end-to-end time | Result |
|---|---:|---:|
| Vanilla Spark CPU | 76.298 ms | 1,688,850,044,797,455 |
| Spark Metal | 80.599 ms | 1,688,850,044,797,455 |

Observed end-to-end speedup: **0.95x**. The Metal physical operator and exact
result are verified, but this is not yet a performance win. This negative result
is retained because it shows that a fast kernel alone does not overcome Parquet
decode, column ownership, synchronization, scheduling, and final Spark work.

## Spark SQL q96-shaped comparison

This second Spark test executes the unmodified q96 SQL text over synthetic
tables with 33,554,432 `store_sales` rows and small dimension tables. Vanilla
Spark uses three broadcast hash joins and a partial global count. The plugin
replaces that complete fact-side region with a single columnar operator that
performs three dense membership checks and the count using Metal.

Both configurations use separate local Spark processes, eight local cores,
off-heap column vectors, one-million-row Parquet batches, AQE disabled, five
warm-ups, and eleven measured runs.

| Configuration | Median end-to-end time | Result |
|---|---:|---:|
| Vanilla Spark CPU | 200.281 ms | 720 |
| Spark Metal | 179.344 ms | 720 |

Observed end-to-end speedup: **1.12x**. A preceding strict run of the same
implementation measured 189.449 ms CPU and 172.130 ms Metal, or **1.10x**.
Both cross the project's provisional
10% performance threshold on the synthetic plan shape. It is not the goal result:
the success gate explicitly requires the licensed TPC-DS scale-factor-10 data.

The q96 edge-case test also matches at 60 rows after introducing null fact keys
and a duplicated matching dimension key. The implementation uses a compact
one-byte presence-map kernel when build keys are unique and a separate
multiplicity kernel when they are not.

The current bridge prepares membership maps and the partial-count buffer once
per partition. It maps the enclosing virtual-memory pages of Spark's off-heap
columns into Metal and supplies the column displacement as a buffer offset. The
latest strict run processed 32 fact batches with `inputCopyFallbacks = 0`.

## Correctness evidence

- Native CPU and Metal results match for non-power-of-two input sizes.
- JNI CPU and Metal results match with every seventeenth value marked null.
- Spark physical plans contain `MetalFusedSum` after transition insertion.
- Spark SQL results match independent references for both nullable and non-null
  one-million-row inputs.
- Spark ANSI mode is verified to leave the expression on the CPU.
- MLX CPU and GPU results match the independent 64-bit reference.
- The Core ML probe deliberately demonstrates and records its 32-bit semantic
  mismatch instead of treating an overflowing result as valid.
- The q96-shaped CPU and Metal results match for the normal dataset and for a
  dataset containing null fact keys and duplicate build keys.
- With adaptive execution enabled, the q96-shaped rule is verified to remain on
  the CPU rather than attempting an unsafe adaptive-plan replacement.

## TPC-DS scale-factor-10 q96 result (2026-08-26)

The licensed SF10 comparison (five warm-ups, eleven measured runs per
configuration, identical Spark settings) after introducing streamed
asynchronous submission, dictionary-aware membership tables, and the shared
prepared-map cache:

| Configuration | Median end-to-end time |
|---|---:|
| Vanilla Spark CPU | 206.4 ms |
| Spark Metal | 135.7 ms |

Observed end-to-end speedup: **1.52x**, with an exact result-hash, row-count,
and schema match, `MetalFusedMembershipCount` in the executed plan,
`numMetalCommands = 30` for 30 fact batches over 28,800,991 rows, and zero
copy fallbacks. Raw data: `benchmark-results/comparison-20260826T084008Z`.

Two findings invalidated the earlier near-tie measurements. First, Spark keeps
these Parquet fact columns dictionary-encoded in the off-heap vectors, so the
earlier address-based GPU path silently fell back to the CPU multiplicity loop
while a metric bug reported the batches as Metal commands. Second, Spark
reuses each partition's off-heap vector memory for the following batch, so
zero-copy references held across `next()` read overwritten data; the streamed
path therefore copies each batch into pooled Metal staging buffers at submit
time.

## GPU Parquet decode on TPC-DS scale-factor-10 q96 (2026-08-26)

`MetalParquetMembershipCount` replaces the whole fact-side region — the Parquet
scan included — for eligible plans, decoding dictionary-encoded data pages on the
GPU instead of consuming Spark's vectorized reader output. Same protocol as
above: five warm-ups, eleven measured runs per configuration, identical Spark
settings, separate local Spark processes.

| Configuration | Median end-to-end time | Speedup |
|---|---:|---:|
| Vanilla Spark CPU | 204.3 ms | — |
| Spark Metal, GPU Parquet decode | 125.9 ms | **1.62x** |

`all_results_match: true`, `metal_operator_present: true`,
`success_gate_met: true`, `MetalParquetMembershipCount` in the executed plan,
`numRowGroups = 30`, `numPagesDecoded = 4380`, `cpuFallbackRowGroups = 0`,
`splitPlanTime = 0`. Raw data:
`benchmark-results/comparison-20260826T120124Z`. An immediately preceding run of
the same build measured 205.4 ms against 133.3 ms, or **1.54x**
(`benchmark-results/comparison-20260826T115726Z`); run-to-run spread on this
host is roughly ±0.06x.

**This is above the project's 1.10x success gate but below the 1.8x checkpoint
and the 2.0x target set for this milestone.** It is reported as what it is.

### Head-to-head against the previous fused path

Both operators measured on the same `-O2` build and the same host state. The earlier 1.52x headline result (above) was measured against a JNI library unknowingly compiled at `-O0` and under a different host state; same-build head-to-head comparisons are the only directly comparable numbers:

| Workload | CPU | GPU Parquet decode | Fused (`parquetScan.enabled=false`) |
|---|---:|---:|---:|
| TPC-DS SF10 q96 | 198.6 ms | **133.3 ms (1.49x)** | 165.5 ms (1.20x) |
| 33.5M-row synthetic q96 | 170.1 ms | 111.5 ms (1.53x) | **97.1 ms (1.75x)** |

The GPU Parquet path wins on real SF10 data by 24% of wall time and loses on the
synthetic shape. The synthetic fact columns have small dictionaries, no nulls,
and long runs, so Spark's vectorized reader is already close to optimal there and
the extra per-page CPU staging is pure overhead. SF10 `store_sales` has ~4.6%
nulls and larger dictionaries — about 1,650 value runs and 2,200 definition-level
segments per 20,000-value page — which is where decoding on the GPU pays.
`spark.metal.parquetScan.enabled` therefore stays **default true**, since SF10 is
the benchmark the project is measured against. The fused path remains one config
flag away.

### What actually cost the time

The first working version of this path measured **0.39x on SF10 and 0.69x on the
synthetic shape — slower than CPU** — with only ~13 ms of `metalTime`. None of
the eight fixes that followed touched a GPU kernel's arithmetic. Measured one at
a time:

| Change | SF10 q96 median | Synthetic median |
|---|---:|---:|
| Starting point (Task 6 as delivered) | 508 ms (0.39x) | 229 ms (0.69x) |
| One command buffer per row group, not per page | — | 204 ms (0.79x) |
| Shared `Configuration`, parallel footer reads | — | 152 ms (1.04x) |
| Bucketed staging pool, chunked commits, parser scratch reuse | — | 141 ms (1.10x) |
| Prefix-only staging reclaim | — | 135 ms (1.12x) |
| `setRequestedSchema` — read 3 columns, not 23 | 425 ms (0.47x) | — |
| Build the native library with `-O2` | 343 ms (0.58x) | — |
| Size threadgroups to the page's real runs | 275 ms (0.72x) | — |
| Cache footer reads across executions | 126-133 ms (1.54-1.62x) | 112 ms (1.53x) |

The two largest single wins were not in the design at all:

- **The JNI library had no `-O` flag**, so clang defaulted to `-O0`. The bridge
  does real per-page CPU work (`parseDataPageV1` walks every run of every page),
  and unoptimized that cost ~300 µs per 20,000-value page — 95% of the path's
  per-task budget, roughly 35-50 ns for a single `vector::push_back`. Adding
  `-O2` cut native page submit from 1370 ms to 323 ms.
- **`ParquetEligibility.check` runs during planning**, so it re-opened and
  re-read all 30 Parquet footers on *every execution of every query*: ~130 ms of
  driver time per run, against a ~200 ms query. Memoising verdicts and row-group
  enumeration on `(path, length, modification time)` took the SF10 median from
  275 ms to 133 ms in one change.

The per-page lesson generalises: at 4,380 pages per query, anything costing tens
of microseconds per page is a first-order term. A per-page `MTLCommandBuffer`
commit cost ~87 µs. A staging pool served first-fit from a flat list handed each
page's ~1 KB work-item request the recycled ~32 KB page buffer, so every page
buffer had to be allocated fresh. `parseDataPageV1` freed both its output
vectors per page by resetting with `out = PageRuns{}`. A fixed 256-wide
threadgroup per run left 95% of its threads idle on data whose runs average 11
values. None of these are visible in `metalTime`.

## Success gate

The gate — an unmodified TPC-DS scale-factor-10 query with a correct result
and at least a 10% median end-to-end improvement — **is met by q96 at 1.62x**
with the GPU Parquet decode path, and was previously met at 1.52x by the fused
path. The 1.8x checkpoint and 2.0x target for the GPU Parquet decode milestone
were **not** reached.
