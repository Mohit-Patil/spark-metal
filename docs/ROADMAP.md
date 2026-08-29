# Roadmap

## M0 — Repository and decision record

- [x] Establish project goal and success threshold.
- [x] Record first-host hardware and missing dependencies.
- [x] Define roles for Metal, MLX, Core ML/ANE, Spark ML, and SynapseML.
- [x] Publish the repository.

## M1 — Reproducible CPU baseline

- [x] Select and pin ARM64 JDK.
- [x] Select and pin Spark/Scala version.
- [x] Add a licence-gated, pinned TPC-DS generator setup.
- [ ] Generate scale-factor-10 Parquet data.
- [x] Add query runner, plan capture, metrics, and correctness hashes.
- [ ] Run and publish the vanilla CPU baseline.

## M2 — Apple backend feasibility

- [x] Implement a common representative workload.
- [x] Run CPU reference.
- [x] Run MLX CPU/GPU prototype.
- [x] Run direct Metal prototype.
- [ ] Evaluate MPSGraph if it reduces implementation work.
- [x] Run Core ML compute-unit and integer-semantics experiment.
- [x] Record the direct-Metal crossover batch sizes.
- [x] Select direct Metal as the primary SQL backend; retain MLX and Core ML as controls.

## M3 — Spark columnar vertical slice

- [x] Register a Spark columnar rule.
- [ ] Add capability tagging and fallback explanations.
- [x] Add native fixed-width column bridge.
- [x] Implement integer filter and projection.
- [x] Implement global partial reductions.
- [x] Fuse filter, projection, and reduction.
- [x] Fuse three integer broadcast-membership joins and partial count.
- [x] Preserve null and duplicate-key join semantics for that slice.
- [x] Reuse prepared membership maps and partial buffers per partition.
- [x] Add page-offset shared-memory mapping with observable copy fallback.

## M4 — TPC-DS proof

- [x] Select q96's three broadcast joins plus partial count as the first candidate.
- [ ] Execute at least one region on the Apple GPU.
- [ ] Validate exact output.
- [ ] Tune batching and memory reuse.
- [ ] Demonstrate the required median end-to-end speedup.

## M5 — Expansion

- [x] Null masks for the initial fused integer expression.
- [x] Limited fixed-point decimals (unscaled int64 partial sums; decimal p<=9
      fact columns decode as int32).
- [x] Grouped aggregation (`MetalParquetGroupedAggregateExec`).
- [x] Integer equi-join, v1 (`MetalParquetBroadcastJoinExec`,
      `spark.metal.parquetJoin.enabled`, default false — see below).
- [ ] Wider TPC-DS coverage.
- [ ] Extend zero-copy coverage and revisit Parquet decoding and local shuffle.

## M6 — Broadcast-join tier v1.1: fused probe+compact kernel

The v1 columnar broadcast-join tier (2026-08-29, docs/GPU_BROADCAST_JOIN_SPEC.md)
is CORRECT on all 103 queries (fires on 40, zero CPU fallbacks) but loses
0.37-0.70x on every candidate under the warmed batched-strict protocol
(`comparison-20260829T075044Z`); its cold-run wins were Spark JIT-warmup
artifacts. Measured causes (phase runs `phase-join-20260829T073320Z`,
`phase-join2-*`):

1. **Full-plane readback + JVM probe**: every decoded plane (all rows) is
   copied back and probed on the CPU; for a selective join almost all of
   that volume is discarded (q60: 39k survivors of 28.8M rows).
2. **Materialization boundary**: Spark fuses scan->join->agg in one codegen
   stage and never materializes the join output; this tier materializes
   full batches, and dimension-string gather dominates low-selectivity
   regions (q22: 26.5M output rows, outputBuildTime 7-10s).

A `fused_join_compact` kernel (GPU probe against the row-index tables,
prefix-sum compaction, scatter of survivor fact values + per-dimension
build-row indices; readback of survivors only) attacks cause 1 directly —
readback and probe shrink by the selectivity factor. Cause 2 bounds the
tier to selective regions regardless; a selectivity-aware gate (runtime,
after the first row groups report survival) should decline regions that
emit a large fraction of their input. Neither is speculative: both are
sized by the phase data above.

**Status (2026-08-29, kernel landed):** `fused_join_compact` shipped
(atomic-slot compaction — join output order carries no meaning, so no
prefix sum needed), with unified per-column page walks (a key that is also
a fact output decodes into both plane types from one read), per-partition
output scratch reuse, and 103/103 exact matches with the tier firing on 40
queries. It moved the warmed batched-strict range from 0.37–0.70x to
0.96–1.28x. The default-flip gate (>=1.10x on >=3 queries under ALL three
protocols) is STILL not met — conservative-of-three after ~3h of sustained
thermal load: q62 1.09x, q58 1.08x, q66 0.96x, q77 0.99x
(`comparison-20260829T084052Z`, `comparison-isolated-*`,
`comparison-join-winners`). Next levers, in measured order: the
selectivity-aware runtime gate (cause 2 still burns the unselective
regions the tier fires on), trimming rowGroupReadTime (parquet-mr page
read + decompress is now the largest remaining stage), and a rested-machine
protocol pass — the near-misses sit inside this machine's documented
±40% thermal swing.
