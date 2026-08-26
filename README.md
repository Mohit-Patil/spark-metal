# Spark Metal

Spark Metal is a research project exploring whether Apache Spark SQL workloads can be accelerated on the integrated GPU in Apple Silicon Macs.

The initial success target is deliberately narrow and measurable:

> Produce a correct, reproducible end-to-end speedup for at least one query in a local TPC-DS scale-factor-10 benchmark, compared with a controlled vanilla Spark CPU baseline.

The project evaluates four related but distinct paths before committing the Spark execution layer to one of them:

- **Direct Metal compute** for general-purpose columnar SQL kernels.
- **MLX** for rapid GPU prototypes and crossover measurements.
- **Core ML and the Apple Neural Engine** for model-graph workloads and as a boundary study for SQL-shaped graphs.
- **Spark ML and SynapseML patterns** for batching, transformer lifecycle, and integration with Spark pipelines.

Direct Metal is the provisional primary candidate for SQL execution because it exposes general GPU compute. The other paths remain part of the research and benchmark matrix rather than being treated as interchangeable APIs.

## Current status

- Private repository and research charter: established.
- First test host detected: 16 GB MacBook Air with Apple M5.
- Xcode and the Metal compiler: available.
- OpenJDK 21 and Apache Spark 4.2.0: installed and ARM64 smoke-tested.
- A Spark 4.2 columnar rule, JNI bridge, and fused Metal partial aggregate: working.
- Integer null semantics: validated through Spark and independently through JNI.
- Controlled 32-million-row synthetic Spark comparison: correct but currently about 5% slower than CPU.
- MLX 0.32.1 comparison: exact, but its GPU did not beat its compiled CPU path through 8.4 million rows.
- Core ML 9.0 capability probe: CPU/GPU execution is possible, but the tested SQL-shaped graph cannot use the Neural Engine or produce Spark's required 64-bit sum.
- TPC-DS baseline: not yet generated.

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
- [Roadmap](docs/ROADMAP.md)
- [First host environment](docs/environment/first-host.md)

## Near-term milestone

The next milestone is a reproducible vanilla Spark baseline:

1. Review and explicitly accept the TPC-DS kit licence.
2. Generate TPC-DS scale factor 10 in Parquet.
3. Execute the full query set where practical.
4. Capture plans, correctness hashes, Spark metrics, and repeated runtimes.
5. Select one numerical query fragment for MLX and Metal feasibility experiments.

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
```

Performance numbers produced here are research results and are not comparable to
official TPC benchmark results.

This repository is private while the feasibility work is underway.
