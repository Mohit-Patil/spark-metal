#include <metal_stdlib>
using namespace metal;

struct FusedParameters {
    uint count;
    int threshold;
    int multiplier;
    int addend;
    uint has_nulls;
};

kernel void fused_filter_project_sum(
    device const int *input [[buffer(0)]],
    device const uchar *validity [[buffer(1)]],
    device long *partial_sums [[buffer(2)]],
    constant FusedParameters &parameters [[buffer(3)]],
    uint global_index [[thread_position_in_grid]],
    uint local_index [[thread_index_in_threadgroup]],
    uint group_index [[threadgroup_position_in_grid]])
{
    threadgroup long scratch[256];

    long contribution = 0;
    if (global_index < parameters.count &&
        (parameters.has_nulls == 0 || validity[global_index] == 0)) {
        int value = input[global_index];
        if (value > parameters.threshold) {
            int projected = value * parameters.multiplier + parameters.addend;
            contribution = long(projected);
        }
    }
    scratch[local_index] = contribution;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = 128; stride > 0; stride >>= 1) {
        if (local_index < stride) {
            scratch[local_index] += scratch[local_index + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (local_index == 0) {
        partial_sums[group_index] = scratch[0];
    }
}
