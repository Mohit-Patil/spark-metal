# Apple backend evaluation

Core ML, MLX, Spark ML, SynapseML, and Metal occupy different layers. The project will test their usefulness without treating them as equivalent compute backends.

## Decision matrix

| Path | Device | General kernels | SQL fit | Spark integration value | Planned role |
|---|---|---:|---:|---:|---|
| Direct Metal compute | Apple GPU | High | High, if kernels are implemented | Requires a native bridge and Spark plugin | Selected primary SQL execution backend |
| MLX | CPU and Apple GPU | Tensor/array operations | Medium for numerical prototypes; low for full SQL semantics | Python and C++ APIs can prototype batch operations | Feasibility and crossover experiments |
| Core ML | CPU, GPU, and Neural Engine selected by the runtime | Model graphs | Low for relational SQL | Useful for model inference transformers | Boundary study and possible ML inference extension |
| Apple Neural Engine | Neural Engine | Restricted through supported public APIs | Very low for general SQL | No public general-purpose Spark kernel path | Not the primary TPC-DS backend |
| Spark ML / MLlib | CPU Spark execution plus algorithm-specific implementations | Not a device backend | Not an SQL accelerator | Native Spark pipeline abstraction | Integration and lifecycle reference |
| SynapseML ONNXModel | ONNX Runtime, commonly CPU/CUDA in its Spark API | Model inference | Low for SQL; high for inference | Demonstrates batched native inference in Spark | Integration reference and possible experiment |
| MPS / MPSGraph | Apple GPU numerical and graph operations | Numerical/graph oriented | Medium for selected primitives | Requires native bridging | Evaluate only if it shortens Metal implementation |

## Observed evidence on the first host

### MLX 0.32.1

The same deterministic integer workload used by the direct Metal prototype now
has a compiled MLX implementation. MLX executes exact `int32` element-wise work
and an `int64` reduction on either its CPU or GPU backend. With five warm-ups and
25 measured runs, the GPU did not beat MLX's compiled CPU path at any tested
size through 8,388,608 rows. At that largest size the medians were 1.923 ms on
CPU, 1.978 ms on GPU with a pre-existing MLX array, and 2.204 ms on GPU including
Python-array materialization.

This is useful negative evidence. It confirms that unified memory alone does not
remove dispatch, framework, synchronization, or ownership costs, and it exposes
that the original Swift loop is not a sufficient best-CPU control. The
end-to-end vanilla Spark baseline remains the authoritative comparison.

MLX currently documents CPU and GPU devices, shared array storage, lazy
evaluation, and graph compilation; it does not expose the Neural Engine as a
general execution device. See the official [MLX overview](https://ml-explore.github.io/mlx/),
[unified-memory guide](https://ml-explore.github.io/mlx/build/html/usage/unified_memory.html),
and [compilation guide](https://ml-explore.github.io/mlx/build/html/usage/compile.html).

### Core ML 9.0 and the Neural Engine

The executable Core ML probe builds the same static predicate, projection, and
reduction as an ML Program. For 4,194,304 rows, Core ML's compute plan lists CPU
and GPU support for every non-constant operation and prefers the GPU. It does
not list the Neural Engine for comparison, multiplication, addition, selection,
or reduction, even though the Neural Engine is present on the host.

The graph also cannot cast to `int64`; its supported cast types stop at `int32`.
Consequently the correct Spark-compatible sum 3,128,594,400 becomes
-1,166,372,896 through 32-bit overflow. Chunking and merging on the host could
avoid this one overflow, but would not turn Core ML into a general SQL execution
engine or provide ANE execution for these operators.

Core ML remains relevant for a future Spark ML inference transformer. Apple
documents that `ComputeUnit.ALL` permits CPU, GPU, and Neural Engine and that the
runtime partitions supported model graphs across them; placement is not a
general kernel API. See [Core ML compute units](https://apple.github.io/coremltools/docs-guides/source/model-prediction.html)
and [typed execution](https://apple.github.io/coremltools/docs-guides/source/typed-execution.html).

### Spark ML / MLlib

Spark ML is a schema-aware DataFrame API for feature transforms, algorithms,
pipelines, tuning, and model persistence. It can define a future Core ML-backed
inference transformer and teach us executor lifecycle patterns, but it does not
replace Catalyst/Tungsten SQL operators or select Apple hardware. The official
[MLlib guide](https://spark.apache.org/docs/latest/ml-guide) describes this
library boundary.

## Evaluation criteria

Each executable backend prototype will be measured on the same representative workloads:

1. Element-wise integer predicate.
2. Predicate plus compaction or selection.
3. Fixed-point projection using scaled integers.
4. Global reduction.
5. Fused predicate, projection, and reduction.

For each workload, record:

- supported data types;
- null handling;
- compilation or graph-build cost;
- input preparation cost;
- synchronization cost;
- kernel or prediction time;
- complete batch time;
- peak memory;
- correctness differences;
- minimum batch size that beats the CPU control.

## Conclusions

### Metal

Metal is the only supported Apple API in this set that provides the general compute control needed to reproduce cuDF-style relational primitives. It also permits shared buffers on Apple Silicon. The cost is that the project must implement the columnar database operations.

### MLX

MLX makes numerical experiments much faster to write and provides a useful
compiled CPU/GPU control. It does not provide Spark-compatible joins, decimals,
null masks, Catalyst physical-plan replacement, or Neural Engine execution by
itself, so it is not the production SQL path.

### Core ML and the Neural Engine

Core ML is optimized around compiled model graphs. The runtime controls placement across permitted compute units, and the public API is not a general SQL kernel interface. The completed capability probe confirms that this SQL-shaped integer graph neither reaches the Neural Engine nor preserves the required 64-bit aggregate semantics.

### Spark ML and SynapseML

These projects are valuable at the integration layer: model/session reuse, batching, schema-aware transformers, and executor lifecycle. They do not replace the need for a columnar SQL execution library. A later project extension could expose Core ML inference as a Spark ML transformer even if it does not accelerate TPC-DS.

## Decision gate

Direct Metal is the primary SQL backend because it is the only evaluated path
that provides the necessary kernel, integer, memory, and Spark integration
control. The first single-expression Spark experiment is 0.95x, while the wider
q96-shaped fusion is 1.13x, demonstrating why whole-plan-region replacement is
the useful level of acceleration. The official TPC-DS success gate remains open.
MLX remains a compact feasibility/reference backend; Core ML, the Neural Engine,
Spark ML, and SynapseML remain model-inference or integration-layer options
rather than TPC-DS execution engines.
