# Spark Metal

Spark Metal is a research project exploring whether Apache Spark SQL workloads can be accelerated on the integrated GPU in Apple Silicon Macs.

The initial success target is deliberately narrow and measurable:

> Produce a correct, reproducible end-to-end speedup for at least one query in a local TPC-DS scale-factor-10 benchmark, compared with a controlled vanilla Spark CPU baseline.

The project evaluates four related but distinct paths before committing the Spark execution layer to one of them:

- **Direct Metal compute** for general-purpose columnar SQL kernels.
- **MLX** for rapid GPU prototypes and crossover measurements.
- **Core ML and the Apple Neural Engine** for model-graph workloads and as a boundary study for SQL-shaped graphs.
- **Spark ML and SynapseML patterns** for batching, transformer lifecycle, and integration with Spark pipelines.

Direct Metal is the selected primary backend for SQL execution because it exposes
general GPU compute, explicit buffers, and command scheduling. MLX remains a
useful numerical reference; Core ML/ANE and Spark ML remain relevant to model
inference, but are not substitutes for a Spark SQL execution engine.

## Current status

- Private repository and research charter: established.
- First test host detected: 16 GB MacBook Air with Apple M5.
- Xcode and the Metal compiler: available.
- OpenJDK 21 and Apache Spark 4.2.0: installed and ARM64 smoke-tested.
- Two Spark 4.2 columnar plan replacements, a JNI bridge, and three Metal kernels: working.
- Integer null semantics: validated through Spark and independently through JNI.
- **TPC-DS scale-factor-10 q96: 1.62x median end-to-end speedup** (CPU 204.3 ms
  vs Metal 125.9 ms; five warm-ups, eleven measured runs per configuration)
  with an exact result-hash, row-count, and schema match — the project's 1.10x
  success gate is met. The 1.8x checkpoint and 2.0x target set for the GPU
  Parquet decode milestone were not reached.
- The q96 fact-side region — the Parquet scan included — runs as a single
  operator that decodes dictionary-encoded data pages on the GPU
  (`MetalParquetMembershipCount`, `spark.metal.parquetScan.enabled`, default
  true). It reads `(file, row group)` splits directly, expands each page's
  RLE/bit-packed runs into GPU-resident id and validity planes, and counts
  membership over those planes; any page the parser rejects falls back to a CPU
  recount of that row group.
- The earlier fused operator remains, one flag away, for plans whose fact-side
  scan is ineligible. It streams Spark's own vectorized batches to the GPU and
  met the same gate at 1.52x.
- **Grouped aggregation on the GPU: correct everywhere, faster in three
  places.** `MetalParquetGroupedAggregate`
  (`spark.metal.parquetAggregate.enabled`, default true) generalizes the
  accelerated region to N broadcast joins plus a partial SUM/COUNT/AVG over an
  eligible Parquet fact scan. Across the full 103-query SF10 suite it fires on
  21 queries — 24 counting the three membership queries — with
  **103/103 exact result matches and `cpuFallbackRowGroups = 0` on every
  accelerated query**. Only q53, q63 and q89 beat Spark's CPU under every
  measurement protocol (1.16x / 1.14x / 1.13x at their most conservative);
  most of the rest lose, the worst by 11x.
- The losses are not GPU losses. `metalTime` is 20-234 ms across all 21
  queries while CPU-side Parquet page parsing runs to 4.9 s; on the worst
  query the kernel is 0.65% of execution. Every query with three or more
  accelerated regions loses, because the operator's per-region driver cost is
  paid once per region while Spark's plan shares that work. The next unit of
  work is a planner cost threshold, not a faster kernel — so the ledgered
  threadgroup-local pre-aggregation optimization was deliberately **not**
  implemented.
- Handling PLAIN-encoded join keys was required to reach the item-keyed
  queries at all: `ss_item_sk` has no dictionary page in any SF10
  `store_sales` file. Key decode now builds a dense value-space code table for
  PLAIN chunks, with the GPU kernel unchanged.
- Head-to-head on one build: the GPU Parquet path wins on SF10 (1.49x vs 1.20x)
  and loses on the 33.5-million-row synthetic q96 shape (1.53x vs 1.75x), whose
  small dictionaries, absent nulls, and long runs already suit Spark's
  vectorized reader.
- MLX 0.32.1 comparison: exact, but its GPU did not beat its compiled CPU path through 8.4 million rows.
- Core ML 9.0 capability probe: CPU/GPU execution is possible, but the tested SQL-shaped graph cannot use the Neural Engine or produce Spark's required 64-bit sum.

## Project principles

1. End-to-end query time matters more than isolated kernel time.
2. Accelerated results must match Spark semantics and output.
3. Unsupported operations must fall back safely to Spark.
4. CPU/GPU conversion, synchronization, compilation, and memory pressure are included in measurements.
5. The first implementation will be a small vertical slice, not an attempted port of all NVIDIA RAPIDS functionality.

## Documents

- [Project charter](docs/PROJECT_CHARTER.md)
- [Backend evaluation](docs/BACKEND_EVALUATION.md)
- [Proposed architecture](docs/ARCHITECTURE.md)
- [Benchmark protocol](docs/BENCHMARK_PROTOCOL.md)
- [Direct Metal prototype](docs/METAL_PROTOTYPE.md)
- [Prototype results](docs/PROTOTYPE_RESULTS.md)
- [TPC-DS q96 target](docs/TPCDS_Q96_TARGET.md)
- [Roadmap](docs/ROADMAP.md)
- [First host environment](docs/environment/first-host.md)

## Near-term milestone

The next milestone is the licensed, reproducible TPC-DS q96 proof:

1. Review and explicitly accept the TPC-DS kit licence.
2. Generate TPC-DS scale factor 10 in Parquet.
3. Run q96 through the controlled CPU and Metal configurations.
4. Verify the result hash and that the Metal operator appears in the physical plan.
5. Confirm or reject the required 10% median end-to-end improvement, then expand
   to the wider query set where practical.

## Local setup

```bash
source scripts/project-env.sh
scripts/smoke-test-spark.sh
scripts/fetch-spark-tpcds-assets.sh
```

The TPC-DS generator is intentionally not vendored. After reviewing its licence,
build the pinned generator with an explicit acknowledgement:

```bash
TPCDS_EULA_ACCEPTED=yes scripts/setup-tpcds-kit.sh
scripts/generate-tpcds-raw.sh
scripts/prepare-tpcds-parquet.sh
scripts/run-tpcds-cpu.sh --queries q1 --warmups 1 --runs 5
```

The native and Spark integration checks do not require TPC-DS data:

```bash
scripts/run-metal-microbenchmark.sh
scripts/setup-mlx.sh
scripts/run-mlx-microbenchmark.sh
scripts/setup-coreml.sh
scripts/run-coreml-capability-probe.sh
scripts/run-jni-smoke-test.sh
scripts/run-spark-plugin-smoke-test.sh
scripts/run-spark-synthetic-benchmark.sh
scripts/run-q96-membership-smoke-test.sh
scripts/run-parquet-decode-smoke-test.sh
scripts/run-parquet-runs-test.sh
Q96_SYNTHETIC_WARMUPS=5 Q96_SYNTHETIC_RUNS=11 scripts/run-q96-synthetic-benchmark.sh
```

After scale-factor-10 Parquet data exists, run the paired comparison with:

```bash
scripts/run-tpcds-comparison.sh --queries q96 --warmups 2 --runs 7
```

The grouped-aggregate correctness smoke and the planning-time eligibility
probe also need that dataset:

```bash
scripts/run-grouped-aggregate-smoke-test.sh
scripts/inspect-grouped-aggregates.sh
```

Performance numbers produced here are research results and are not comparable to
official TPC benchmark results.

This repository is private while the feasibility work is underway.
