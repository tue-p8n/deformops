#pragma once

#include <type_traits>

#include <torch/headeronly/util/BFloat16.h>
#include <torch/headeronly/util/Half.h>

const int64_t kWarpSize = 32;

// The arithmetic type: float32 for half precision, else the operand dtype.
template <typename scalar_t>
struct accum_type {
    using type = scalar_t;
};
template <>
struct accum_type<c10::Half> {
    using type = float;
};
template <>
struct accum_type<c10::BFloat16> {
    using type = float;
};
template <typename scalar_t>
using accum_t = typename accum_type<scalar_t>::type;

namespace deform_utils {

// Whether the dtype is 16-bit, and so eligible for half2 atomics.
template <typename T>
struct is_16bit {
    static constexpr bool value = false;
};
template <>
struct is_16bit<c10::Half> {
    static constexpr bool value = true;
};
template <>
struct is_16bit<c10::BFloat16> {
    static constexpr bool value = true;
};

// The 16-bit overloads are always declared so nvcc's host pass resolves them. Below
// the minimum architecture the body is empty, and
// ``_msda_bridge.require_native_atomic`` refuses such launches. See docs/precision.md.
template <typename scalar_t>
__device__ inline void atomicAdd_scalar(scalar_t *address, scalar_t val) {
    atomicAdd(address, val);
}

__device__ inline void atomicAdd_scalar(c10::Half *address, c10::Half val) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 700)
    atomicAdd(reinterpret_cast<__half *>(address),
              *reinterpret_cast<const __half *>(&val));
#endif
}

__device__ inline void atomicAdd_half2(c10::Half *address, c10::Half val1,
                                       c10::Half val2) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 700)
    __half2 val;
    val.x = *reinterpret_cast<const __half *>(&val1);
    val.y = *reinterpret_cast<const __half *>(&val2);
    atomicAdd(reinterpret_cast<__half2 *>(address), val);
#endif
}

__device__ inline void atomicAdd_scalar(c10::BFloat16 *address, c10::BFloat16 val) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    atomicAdd(reinterpret_cast<__nv_bfloat16 *>(address),
              *reinterpret_cast<const __nv_bfloat16 *>(&val));
#endif
}

__device__ inline void atomicAdd_half2(c10::BFloat16 *address, c10::BFloat16 val1,
                                       c10::BFloat16 val2) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    __nv_bfloat162 val;
    val.x = *reinterpret_cast<const __nv_bfloat16 *>(&val1);
    val.y = *reinterpret_cast<const __nv_bfloat16 *>(&val2);
    atomicAdd(reinterpret_cast<__nv_bfloat162 *>(address), val);
#endif
}

// Copy `channels_per_thread` elements in one vector load/store when a vector type
// of that byte width exists; otherwise fall back to an unrolled scalar copy.
template <typename scalar_t, int64_t channels_per_thread>
struct channel_vector_type {
    static constexpr int64_t bytes = sizeof(scalar_t) * channels_per_thread;
    using type = std::conditional_t<
        bytes == 2, scalar_t,
        std::conditional_t<
            bytes == 4, uint,
            std::conditional_t<
                bytes == 8, uint2,
                std::conditional_t<
                    bytes == 16, uint4,
                    std::conditional_t<bytes == 32, ulonglong4, scalar_t>>>>>;
};
template <typename scalar_t, int64_t channels_per_thread>
using channel_vector_t_for =
    typename channel_vector_type<scalar_t, channels_per_thread>::type;

template <typename scalar_t, typename channel_vector_t, int64_t channels_per_thread>
__device__ __forceinline__ void vectorized_copy(scalar_t *dst, const scalar_t *src) {
    if constexpr (sizeof(channel_vector_t) == sizeof(scalar_t) * channels_per_thread) {
        *(channel_vector_t *)(dst) = *(channel_vector_t *)(src);
    } else {
#pragma unroll
        for (int i = 0; i < channels_per_thread; ++i) {
            dst[i] = src[i];
        }
    }
}

// Call `fn` (`[&]<int64_t V>() { ... }`) with the `V` in `Values` equal to `value`,
// so loop bounds unroll; returns `false`, dispatching nothing, when none matches.
template <int64_t... Values, typename Fn>
bool dispatch_constant(int64_t value, Fn &&fn) {
    return ((value == Values ? (fn.template operator()<Values>(), true) : false) ||
            ...);
}

__device__ __forceinline__ unsigned get_lane_mask(int lanes_per_head,
                                                  int head_index_in_warp) {
    unsigned mask = (lanes_per_head == 32) ? ~0u : (1u << lanes_per_head) - 1u;
    return mask << (head_index_in_warp * lanes_per_head);
}
template <typename acc_t>
__device__ __forceinline__ void softmax_forward_thread0(acc_t *attention_weight,
                                                        int64_t N) {
    acc_t softmax_max = -1e100;
    acc_t softmax_sum = 0.0;
    for (int j = 0; j < N; j++) {
        softmax_max = max(softmax_max, attention_weight[j]);
    }
    for (int j = 0; j < N; j++) {
        acc_t exp_results = exp(attention_weight[j] - softmax_max);
        attention_weight[j] = exp_results;
        softmax_sum += exp_results;
    }
    for (int j = 0; j < N; j++) {
        attention_weight[j] /= softmax_sum;
    }
}

template <typename acc_t>
__device__ __forceinline__ void softmax_forward_warp(acc_t *attention_weight, int64_t N,
                                                     unsigned lane_mask) {
    acc_t local_max = -1e100;
    for (int j = threadIdx.x; j < N; j += blockDim.x) {
        local_max = max(local_max, attention_weight[j]);
    }
    for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
        local_max = max(local_max, __shfl_xor_sync(lane_mask, local_max, offset));
    }

    acc_t local_sum = 0.0;
    for (int j = threadIdx.x; j < N; j += blockDim.x) {
        acc_t exp_results = exp(attention_weight[j] - local_max);
        attention_weight[j] = exp_results;
        local_sum += exp_results;
    }
    for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
        local_sum += __shfl_xor_sync(lane_mask, local_sum, offset);
    }

    for (int j = threadIdx.x; j < N; j += blockDim.x) {
        attention_weight[j] /= local_sum;
    }
}

template <typename acc_t>
__device__ __forceinline__ void
softmax_backward_thread0(acc_t *grad_attention_weight, const acc_t *attention_weight,
                         const acc_t *head_grad_attention_weight, int64_t N) {
    for (int i = 0; i < N; ++i) {
        acc_t grad_i = 0.;
        for (int j = 0; j < N; ++j) {
            if (i != j) {
                grad_i -= head_grad_attention_weight[j] * attention_weight[i] *
                          attention_weight[j];
            } else {
                grad_i += head_grad_attention_weight[i] * attention_weight[i] *
                          (1 - attention_weight[i]);
            }
        }
        grad_attention_weight[i] = grad_i;
    }
}

template <typename acc_t>
__device__ __forceinline__ void
warp_reduce_grad_sampling_parameters(acc_t *grad_sampling_parameters,
                                     unsigned lane_mask) {
    for (uint32_t offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
        grad_sampling_parameters[0] +=
            __shfl_down_sync(lane_mask, grad_sampling_parameters[0], offset);
        grad_sampling_parameters[1] +=
            __shfl_down_sync(lane_mask, grad_sampling_parameters[1], offset);
        grad_sampling_parameters[2] +=
            __shfl_down_sync(lane_mask, grad_sampling_parameters[2], offset);
    }
}

// Shared-memory `warp_reduce_grad_sampling_parameters`: thread 0 of a head sums the
// per-lane (x, y, weight) gradients in the head's stride-3 tile at `head_slot`.
template <typename acc_t>
__device__ __forceinline__ void
block_reduce_grad_sampling_parameters(const acc_t *shared_grad_sampling_parameters,
                                      int64_t head_slot, acc_t out[3]) {
    const int64_t base = head_slot * blockDim.x * 3;
    out[0] = shared_grad_sampling_parameters[base];
    out[1] = shared_grad_sampling_parameters[base + 1];
    out[2] = shared_grad_sampling_parameters[base + 2];
    for (int lane = 1; lane < blockDim.x; ++lane) {
        out[0] += shared_grad_sampling_parameters[base + 3 * lane];
        out[1] += shared_grad_sampling_parameters[base + 3 * lane + 1];
        out[2] += shared_grad_sampling_parameters[base + 3 * lane + 2];
    }
}

// Launch geometry for every kernel: block x = channel lane, y = head,
// z = packed (batch, query) units.
struct BlockGeometry {
    dim3 grid;
    dim3 block;
    int64_t queries_per_block;
};

inline BlockGeometry compute_block_geometry(int64_t B, int64_t Q, int64_t G, int64_t D,
                                            int64_t channels_per_thread,
                                            int64_t threads_per_block) {
    // A degenerate request yields `queries_per_block == 0`, which launch sites
    // reject with a message; dividing by it here would kill the process.
    const int64_t lanes_per_head =
        channels_per_thread > 0 ? D / channels_per_thread : 0;
    const int64_t queries_per_block =
        (lanes_per_head > 0 && G > 0) ? threads_per_block / lanes_per_head / G : 0;
    BlockGeometry geometry;
    geometry.queries_per_block = queries_per_block;
    geometry.grid = dim3(queries_per_block > 0 ? B * Q / queries_per_block : 0);
    geometry.block = dim3(lanes_per_head, G, queries_per_block);
    return geometry;
}

} // namespace deform_utils
