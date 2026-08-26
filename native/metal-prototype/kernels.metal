#include <metal_stdlib>
using namespace metal;

struct FusedParameters {
    uint count;
    int threshold;
    int multiplier;
    int addend;
    uint has_nulls;
};

struct MembershipCountParameters {
    uint count;
    uint null_mask;
    int key_min_0;
    int key_min_1;
    int key_min_2;
    uint key_span_0;
    uint key_span_1;
    uint key_span_2;
};

inline bool dense_contains(device const uchar *keys, int minimum, uint span, int value)
{
    long offset = long(value) - long(minimum);
    return offset >= 0 && ulong(offset) < ulong(span) && keys[offset] != 0;
}

inline uint dense_multiplicity(device const uint *keys, int minimum, uint span, int value)
{
    long offset = long(value) - long(minimum);
    return offset >= 0 && ulong(offset) < ulong(span) ? keys[offset] : 0u;
}

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

kernel void fused_membership_count_3_unique(
    device const int *input_0 [[buffer(0)]],
    device const int *input_1 [[buffer(1)]],
    device const int *input_2 [[buffer(2)]],
    device const uchar *nulls_0 [[buffer(3)]],
    device const uchar *nulls_1 [[buffer(4)]],
    device const uchar *nulls_2 [[buffer(5)]],
    device const uchar *keys_0 [[buffer(6)]],
    device const uchar *keys_1 [[buffer(7)]],
    device const uchar *keys_2 [[buffer(8)]],
    device long *partial_counts [[buffer(9)]],
    constant MembershipCountParameters &parameters [[buffer(10)]],
    uint global_index [[thread_position_in_grid]],
    uint local_index [[thread_index_in_threadgroup]],
    uint group_index [[threadgroup_position_in_grid]])
{
    threadgroup long scratch[256];
    long contribution = 0;
    if (global_index < parameters.count) {
        bool valid =
            ((parameters.null_mask & 1u) == 0u || nulls_0[global_index] == 0) &&
            ((parameters.null_mask & 2u) == 0u || nulls_1[global_index] == 0) &&
            ((parameters.null_mask & 4u) == 0u || nulls_2[global_index] == 0);
        if (valid) {
            contribution =
                dense_contains(keys_0, parameters.key_min_0, parameters.key_span_0, input_0[global_index]) &&
                dense_contains(keys_1, parameters.key_min_1, parameters.key_span_1, input_1[global_index]) &&
                dense_contains(keys_2, parameters.key_min_2, parameters.key_span_2, input_2[global_index])
                ? 1 : 0;
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
        partial_counts[group_index] = scratch[0];
    }
}

kernel void fused_membership_count_3_multiplicity(
    device const int *input_0 [[buffer(0)]],
    device const int *input_1 [[buffer(1)]],
    device const int *input_2 [[buffer(2)]],
    device const uchar *nulls_0 [[buffer(3)]],
    device const uchar *nulls_1 [[buffer(4)]],
    device const uchar *nulls_2 [[buffer(5)]],
    device const uint *keys_0 [[buffer(6)]],
    device const uint *keys_1 [[buffer(7)]],
    device const uint *keys_2 [[buffer(8)]],
    device long *partial_counts [[buffer(9)]],
    constant MembershipCountParameters &parameters [[buffer(10)]],
    uint global_index [[thread_position_in_grid]],
    uint local_index [[thread_index_in_threadgroup]],
    uint group_index [[threadgroup_position_in_grid]])
{
    threadgroup long scratch[256];
    long contribution = 0;
    if (global_index < parameters.count) {
        bool valid =
            ((parameters.null_mask & 1u) == 0u || nulls_0[global_index] == 0) &&
            ((parameters.null_mask & 2u) == 0u || nulls_1[global_index] == 0) &&
            ((parameters.null_mask & 4u) == 0u || nulls_2[global_index] == 0);
        if (valid) {
            uint multiplicity_0 = dense_multiplicity(
                keys_0, parameters.key_min_0, parameters.key_span_0, input_0[global_index]);
            uint multiplicity_1 = dense_multiplicity(
                keys_1, parameters.key_min_1, parameters.key_span_1, input_1[global_index]);
            uint multiplicity_2 = dense_multiplicity(
                keys_2, parameters.key_min_2, parameters.key_span_2, input_2[global_index]);
            contribution = long(multiplicity_0) * long(multiplicity_1) * long(multiplicity_2);
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
        partial_counts[group_index] = scratch[0];
    }
}
