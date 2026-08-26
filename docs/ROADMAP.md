# Roadmap

## M0 — Repository and decision record

- [x] Establish project goal and success threshold.
- [x] Record first-host hardware and missing dependencies.
- [x] Define roles for Metal, MLX, Core ML/ANE, Spark ML, and SynapseML.
- [x] Publish the private repository.

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
- [ ] Select the primary backend after the remaining comparisons.

## M3 — Spark columnar vertical slice

- [x] Register a Spark columnar rule.
- [ ] Add capability tagging and fallback explanations.
- [x] Add native fixed-width column bridge.
- [x] Implement integer filter and projection.
- [x] Implement global partial reductions.
- [x] Fuse filter, projection, and reduction.

## M4 — TPC-DS proof

- [ ] Select candidate TPC-DS plan regions from the baseline.
- [ ] Execute at least one region on the Apple GPU.
- [ ] Validate exact output.
- [ ] Tune batching and memory reuse.
- [ ] Demonstrate the required median end-to-end speedup.

## M5 — Expansion

- [x] Null masks for the initial fused integer expression.
- [ ] Limited fixed-point decimals.
- [ ] Grouped aggregation.
- [ ] Integer equi-join.
- [ ] Wider TPC-DS coverage.
- [ ] Revisit zero-copy buffers, Parquet decoding, and local shuffle.
