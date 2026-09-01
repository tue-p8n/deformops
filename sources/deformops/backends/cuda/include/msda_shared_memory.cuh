// Dynamic shared-memory partition of the backward kernel, and the predicate
// that selects which reduction strategy a launch uses.
#pragma once

#include "msda_args.cuh"

namespace msda_split {

// Extents of the backward kernel's dynamic shared-memory partition. The device
// offsets and the host total both read them here, including each extent's
// grouping and integer-division order, so they cannot disagree.

template <int64_t channels_per_thread>
struct BackwardSharedExtents {
    // One of two attention tiles: pre-softmax weight gradients, then the
    // softmaxed weights. The kernel strides between them by this size.
    __host__ __device__ static constexpr int64_t
    attention_tile(int64_t queries_per_block, int64_t G, int64_t L, int64_t K) {
        return queries_per_block * G * L * K;
    }
    // Both attention-shaped tiles together.
    __host__ __device__ static constexpr int64_t
    attention(int64_t queries_per_block, int64_t G, int64_t L, int64_t K) {
        return attention_tile(queries_per_block, G, L, K) * 2;
    }
    // Per-lane sampling-parameter gradients; shared-reduction path only.
    // ``channels_per_thread`` is a template parameter, so the divide is by a
    // constant instead of a 64-bit division.
    __host__ __device__ static constexpr int64_t parameters(int64_t queries_per_block,
                                                            int64_t G, int64_t D) {
        return queries_per_block * G * D / channels_per_thread * 3;
    }
    // Per-head location broadcast slot; shared-reduction path only.
    __host__ __device__ static constexpr int64_t locations(int64_t queries_per_block,
                                                           int64_t G) {
        return queries_per_block * G * 2;
    }

    template <typename scalar_t>
    static constexpr int64_t bytes(int64_t queries_per_block, int64_t G, int64_t D,
                                   int64_t L, int64_t K, bool use_warp_reduction) {
        int64_t elements = attention(queries_per_block, G, L, K);
        if (!use_warp_reduction)
            elements +=
                parameters(queries_per_block, G, D) + locations(queries_per_block, G);
        return elements * static_cast<int64_t>(sizeof(accum_t<scalar_t>));
    }
};

inline bool check_backward_warp(int channels_per_thread, int64_t D) {
    int64_t n_group_threads = D / channels_per_thread;
    return (n_group_threads <= kWarpSize) && (kWarpSize % n_group_threads == 0);
}

} // namespace msda_split
