# Apple backend evaluation

Core ML, MLX, Spark ML, SynapseML, and Metal occupy different layers. The project will test their usefulness without treating them as equivalent compute backends.

## Decision matrix

| Path | Device | General kernels | SQL fit | Spark integration value | Planned role |
|---|---|---:|---:|---:|---|
| Direct Metal compute | Apple GPU | High | High, if kernels are implemented | Requires a native bridge and Spark plugin | Provisional primary execution backend |
| MLX | CPU and Apple GPU | Tensor/array operations | Medium for numerical prototypes; low for full SQL semantics | Python and C++ APIs can prototype batch operations | Feasibility and crossover experiments |
| Core ML | CPU, GPU, and Neural Engine selected by the runtime | Model graphs | Low for relational SQL | Useful for model inference transformers | Boundary study and possible ML inference extension |
| Apple Neural Engine | Neural Engine | Restricted through supported public APIs | Very low for general SQL | No public general-purpose Spark kernel path | Not the primary TPC-DS backend |
| Spark ML / MLlib | CPU Spark execution plus algorithm-specific implementations | Not a device backend | Not an SQL accelerator | Native Spark pipeline abstraction | Integration and lifecycle reference |
| SynapseML ONNXModel | ONNX Runtime, commonly CPU/CUDA in its Spark API | Model inference | Low for SQL; high for inference | Demonstrates batched native inference in Spark | Integration reference and possible experiment |
| MPS / MPSGraph | Apple GPU numerical and graph operations | Numerical/graph oriented | Medium for selected primitives | Requires native bridging | Evaluate only if it shortens Metal implementation |

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

## Provisional conclusions to validate

### Metal

Metal is the only supported Apple API in this set that provides the general compute control needed to reproduce cuDF-style relational primitives. It also permits shared buffers on Apple Silicon. The cost is that the project must implement the columnar database operations.

### MLX

MLX should make early numerical experiments much faster to write. It may demonstrate whether unified-memory GPU execution is promising before the Spark plugin exists. It is not expected to provide Spark-compatible joins, decimals, null masks, or physical-plan replacement by itself.

### Core ML and the Neural Engine

Core ML is optimized around compiled model graphs. The runtime controls placement across permitted compute units, and the public API is not a general SQL kernel interface. The project will still test whether a small static numerical graph can express a representative fused operation, but this is a research comparison rather than the assumed implementation.

### Spark ML and SynapseML

These projects are valuable at the integration layer: model/session reuse, batching, schema-aware transformers, and executor lifecycle. They do not replace the need for a columnar SQL execution library. A later project extension could expose Core ML inference as a Spark ML transformer even if it does not accelerate TPC-DS.

## Decision gate

Direct Metal becomes the committed SQL backend only after it beats the controlled CPU implementation for the fused representative workload including buffer and synchronization costs. MLX or another path may be retained as a prototype backend if it wins materially or reduces implementation risk.

