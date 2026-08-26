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

- Repository foundation and research charter: in progress.
- First test host detected: 16 GB MacBook Air with Apple M5.
- Xcode and the Metal compiler: available.
- Java and Apache Spark: not yet installed.
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
- [Roadmap](docs/ROADMAP.md)
- [First host environment](docs/environment/first-host.md)

## Near-term milestone

The next milestone is a reproducible vanilla Spark baseline:

1. Install and pin an ARM64 JDK and Spark distribution.
2. Generate TPC-DS scale factor 10 in Parquet.
3. Execute the full query set where practical.
4. Capture plans, correctness hashes, Spark metrics, and repeated runtimes.
5. Select one numerical query fragment for MLX and Metal feasibility experiments.

This repository is private while the feasibility work is underway.

