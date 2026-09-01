// Forward pass of the split-layout MSDA kernel, shared by every configuration.
// See docs/kernels.md.
#pragma once

#include "msda_args.cuh"

namespace msda_split {

// The largest block in the tighter tier, where typical launches land and fp32 runs
// faster. Neither tier spills registers.
inline constexpr int64_t kForwardThreadTier = 512;

template <typename scalar_t, int64_t channels_per_thread, typename channel_vector_t,
          int64_t L, int64_t K, int apply_softmax, int64_t max_threads_per_block>
__global__ void __launch_bounds__(max_threads_per_block)
    forward_kernel(const ForwardArgs<scalar_t> args) {
    const scalar_t *const __restrict__ value = args.value;
    const int64_t *const __restrict__ spatial_shapes = args.spatial_shapes;
    const int64_t *const __restrict__ level_start_index = args.level_start_index;
    const accum_t<scalar_t> *const __restrict__ location = args.location;
    const scalar_t *const __restrict__ attention_weight = args.attention_weight;
    scalar_t *const __restrict__ output = args.output;
    const int64_t N = args.N;
    const int64_t G = args.G;
    const int64_t D = args.D;
    const int64_t Q = args.Q;
    const int64_t queries_per_block = args.queries_per_block;

    const int64_t batch_query_index = blockIdx.x * queries_per_block + threadIdx.z;
    const int64_t qi = batch_query_index % Q;
    const int64_t bi = batch_query_index / Q;

    // 32-bit per-point offsets (see interpolate.cuh); base pointers and query
    // bookkeeping are 64-bit.
    const int32_t channel_offset = threadIdx.x * channels_per_thread;
    const int32_t gi = threadIdx.y;
    // Lane 0 reads and broadcasts the location when the head tiles a warp evenly.
    // Otherwise every lane reads it: a shuffle mask that excludes the caller is
    // undefined.
    const bool broadcast_location =
        blockDim.x <= kWarpSize && kWarpSize % blockDim.x == 0;
    const int64_t heads_per_warp = broadcast_location ? kWarpSize / blockDim.x : 1;
    const int64_t head_index_in_warp = (threadIdx.z * G + threadIdx.y) % heads_per_warp;
    const unsigned lane_mask =
        broadcast_location ? deform_utils::get_lane_mask(blockDim.x, head_index_in_warp)
                           : 0u;
    accum_t<scalar_t> output_accumulator[channels_per_thread];
    for (int i = 0; i < channels_per_thread; i++)
        output_accumulator[i] = 0.0;
    const scalar_t *attention_weight_pointer =
        attention_weight + (((bi * Q + qi) * G + gi) * L) * K;

    // The normalizing path keeps the max logit and the shifted exponential sum, not an
    // `L * K` array. The max spans the whole block first: a running max would read
    // `-inf - -inf`, NaN, while the prefix is masked.
    accum_t<scalar_t> softmax_max = -1e100;
    accum_t<scalar_t> softmax_sum = 0;
    if constexpr (apply_softmax == 1) {
        for (int i = 0; i < L * K; i++)
            softmax_max =
                max(softmax_max, (accum_t<scalar_t>)attention_weight_pointer[i]);
    }
    const int32_t w_stride = G * D;
    const int32_t base_offset = gi * D + channel_offset;
    const accum_t<scalar_t> *location_pointer =
        location + (((bi * Q + qi) * G + gi) * L) * K * 2;
    for (int li = 0; li < L; li++) {
        const int32_t spatial_h = spatial_shapes[li * 2];
        const int32_t spatial_w = spatial_shapes[li * 2 + 1];
        const int64_t level_start = level_start_index[li];
        const scalar_t *value_pointer = value + (bi * N + level_start) * G * D;
        for (int ki = 0; ki < K; ki++) {
            // Block-uniform, so the shuffle is warp-collective; overwritten lanes
            // start at zero.
            accum_t<scalar_t> location_w = 0, location_h = 0;
            if (broadcast_location) {
                if (threadIdx.x == 0) {
                    location_w =
                        (accum_t<scalar_t>)location_pointer[li * K * 2 + ki * 2];
                    location_h =
                        (accum_t<scalar_t>)location_pointer[li * K * 2 + ki * 2 + 1];
                }
                location_w =
                    __shfl_sync(lane_mask, location_w, head_index_in_warp * blockDim.x);
                location_h =
                    __shfl_sync(lane_mask, location_h, head_index_in_warp * blockDim.x);
            } else {
                location_w = (accum_t<scalar_t>)location_pointer[li * K * 2 + ki * 2];
                location_h =
                    (accum_t<scalar_t>)location_pointer[li * K * 2 + ki * 2 + 1];
            }
            // The weight enters the softmax sum before the bounds test, so out-of-level
            // points still count in the denominator.
            accum_t<scalar_t> point_weight =
                (accum_t<scalar_t>)attention_weight_pointer[li * K + ki];
            if constexpr (apply_softmax == 1) {
                point_weight = exp(point_weight - softmax_max);
                softmax_sum += point_weight;
            }
            const accum_t<scalar_t> pixel_h = location_h * spatial_h - 0.5;
            const accum_t<scalar_t> pixel_w = location_w * spatial_w - 0.5;
            if (pixel_h >= -1 && pixel_w >= -1 && pixel_h < spatial_h &&
                pixel_w < spatial_w) {
                deform_interpolate::im2col_bilinear<scalar_t, channel_vector_t,
                                                    channels_per_thread>(
                    output_accumulator, value_pointer, spatial_h, spatial_w, pixel_h,
                    pixel_w, point_weight, w_stride, base_offset);
            }
        }
    }
    if constexpr (apply_softmax == 1) {
        const accum_t<scalar_t> inverse_sum = accum_t<scalar_t>(1) / softmax_sum;
        for (int i = 0; i < channels_per_thread; i++)
            output_accumulator[i] *= inverse_sum;
    }

    int64_t output_index = ((bi * Q + qi) * G + gi) * D + channel_offset;
    alignas(channel_vector_t) scalar_t output_values[channels_per_thread];
    for (int ds = 0; ds < channels_per_thread; ds++)
        output_values[ds] = (scalar_t)output_accumulator[ds];
    deform_utils::vectorized_copy<scalar_t, channel_vector_t, channels_per_thread>(
        output + output_index, output_values);
}

template <typename scalar_t, typename channel_vector_t, int64_t K,
          int64_t channels_per_thread, int apply_softmax>
void _forward_launch(cudaStream_t stream, ForwardArgs<scalar_t> args, const int64_t B,
                     const int64_t L, const int64_t threads_per_block) {
    CHECK_DIVISIBLE(args.D, channels_per_thread);
    const auto geometry = deform_utils::compute_block_geometry(
        B, args.Q, args.G, args.D, channels_per_thread, threads_per_block);
    args.queries_per_block = geometry.queries_per_block;
    // A zero or non-dividing block size would silently drop the tail queries.
    TORCH_CHECK(geometry.queries_per_block >= 1, "threads_per_block (",
                threads_per_block, ") is too small for one query at G=", args.G,
                ", D=", args.D, ", channels_per_thread=", channels_per_thread);
    CHECK_DIVISIBLE((B * args.Q), geometry.queries_per_block);

    // The bound follows the block this launch needs. Two tiers only, since each
    // tier duplicates every instantiation.
    const int64_t block_threads = block_thread_count(geometry.block);
    TORCH_CHECK(block_threads <= kMaxThreadsPerBlock, "threads_per_block (",
                threads_per_block, ") needs a block of ", block_threads,
                " threads at G=", args.G, ", D=", args.D,
                ", channels_per_thread=", channels_per_thread,
                ", above the hardware maximum of ", kMaxThreadsPerBlock);
    const int64_t thread_tier =
        block_threads <= kForwardThreadTier ? kForwardThreadTier : kMaxThreadsPerBlock;

    void (*kernel)(ForwardArgs<scalar_t>) = nullptr;
    // Bound to a variable first: the commas in the template argument list would
    // otherwise be read as further macro arguments by TORCH_CHECK.
    bool dispatched = false;
    deform_utils::dispatch_constant<kForwardThreadTier, kMaxThreadsPerBlock>(
        thread_tier, [&]<int64_t MaxThreads>() {
            dispatched = deform_utils::dispatch_constant<1, 2, 3, 4, 5>(
                L, [&]<int64_t Levels>() {
                    kernel =
                        forward_kernel<scalar_t, channels_per_thread, channel_vector_t,
                                       Levels, K, apply_softmax, MaxThreads>;
                });
        });
    TORCH_CHECK(dispatched, kLevelBoundMessage, L);
    kernel<<<geometry.grid, geometry.block, 0, stream>>>(args);
    check_launch("msda_split::forward_kernel");
}

template <typename scalar_t, int64_t K, int apply_softmax>
void forward_launch_inner(cudaStream_t stream, const ForwardArgs<scalar_t> &args,
                          const int64_t B, const int64_t L,
                          const int64_t channels_per_thread,
                          const int64_t threads_per_block) {
    // One dispatch over the runtime channel count; `channel_vector_t_for` picks
    // the widest matching vector type per precision (see backward_launch_inner).
    const auto launch = [&]<int64_t N>() {
        _forward_launch<scalar_t, deform_utils::channel_vector_t_for<scalar_t, N>, K, N,
                        apply_softmax>(stream, args, B, L, threads_per_block);
    };
    // `forward_thread_layout` in `core/tuning.py` always emits 8, and
    // `test_dispatch_packs.py` checks that this pack matches. Bound first for
    // TORCH_CHECK.
    const bool dispatched =
        deform_utils::dispatch_constant<8>(channels_per_thread, launch);
    TORCH_CHECK(dispatched, "unsupported channels_per_thread: ", channels_per_thread);
}

template <typename scalar_t>
void forward_launch(cudaStream_t stream, const ForwardArgs<scalar_t> &args,
                    const int64_t B, const int64_t L, const int64_t K,
                    const int64_t channels_per_thread, const int64_t threads_per_block,
                    int64_t apply_softmax) {
    // Bound first for TORCH_CHECK. An unsupported K would launch nothing and
    // return the allocation's zeros.
    const bool dispatched =
        deform_utils::dispatch_constant<4, 8>(K, [&]<int64_t Points>() {
            if (apply_softmax == 1)
                forward_launch_inner<scalar_t, Points, 1>(
                    stream, args, B, L, channels_per_thread, threads_per_block);
            else
                forward_launch_inner<scalar_t, Points, 0>(
                    stream, args, B, L, channels_per_thread, threads_per_block);
        });
    TORCH_CHECK(dispatched, "unsupported points per level: ", K);
}

} // namespace msda_split
