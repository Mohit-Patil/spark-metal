# Project charter

## Problem

Apache Spark can schedule accelerator resources, but an accelerator becomes useful only when a software layer translates Spark physical operators and expressions into compatible device operations. NVIDIA's cuDF plugin for Spark supplies that layer for CUDA GPUs. Apple Silicon provides an integrated GPU with unified memory, but there is no equivalent Spark SQL execution backend for Metal.

## Objective

Build a research prototype that accelerates a supported subset of Spark SQL columnar execution on an integrated Apple Silicon GPU and demonstrates a correct end-to-end improvement on at least one TPC-DS scale-factor-10 query.

## Primary research questions

1. Which portion of local Spark query time remains GPU-accelerable after Parquet decoding, Spark planning, shuffle, and result materialization?
2. Can Spark columnar batches reach Metal shared buffers with sufficiently low conversion cost?
3. At what batch size does the Apple GPU outperform Spark's CPU execution for filter, projection, reduction, and grouped aggregation?
4. Which Spark SQL types and semantics can be implemented exactly with Metal's available numeric operations?
5. Can adjacent supported operators remain in one accelerated columnar stage?
6. Do MLX, Core ML, Spark ML, or SynapseML provide reusable components or integration patterns that materially reduce implementation effort?

## In scope

- Local Spark on ARM64 macOS.
- Apple Silicon integrated GPU, initially Apple M5 with M4 compatibility considered.
- TPC-DS scale factor 10 stored in Parquet.
- Spark SQL/DataFrame physical-plan interception.
- Columnar filter, projection, and partial aggregation as the first operator family.
- Correct null handling and limited exact fixed-point decimals.
- CPU fallback and explainable eligibility decisions.
- Direct Metal, MLX, Core ML/ANE, and Spark ML/SynapseML feasibility comparisons.

## Initially out of scope

- Full compatibility with every TPC-DS query.
- Multi-node macOS clusters.
- GPU-native Parquet decoding.
- GPU-aware network shuffle.
- Complete joins, windows, complex strings, and arbitrary-precision decimals.
- Apple private APIs or undocumented Neural Engine interfaces.
- Production support commitments.

## Definition of success

The first project goal is achieved when all conditions below hold:

1. A TPC-DS scale-factor-10 dataset and baseline are reproducible from repository instructions.
2. At least one unmodified TPC-DS query contains a Spark physical-plan region executed on the Apple GPU.
3. Its output matches the vanilla Spark result under the documented comparison rules.
4. Its median end-to-end runtime improves by at least 10% across at least five measured runs after warm-up.
5. Setup, compilation, conversion, synchronization, and fallback costs are included rather than hidden.
6. The physical plan and metrics clearly identify what ran on the GPU and what remained on the CPU.

The 10% threshold protects the result from ordinary run-to-run noise. Smaller improvements will still be recorded as research findings but will not complete the goal.

