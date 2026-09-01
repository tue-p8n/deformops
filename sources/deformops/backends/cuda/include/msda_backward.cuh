// Backward pass of the split-layout MSDA kernel. Its traits choose the value
// gradient precision, the softmax reduction and the atomic width; see
// docs/kernels.md.
#pragma once

#include "msda_args.cuh"
#include "msda_shared_memory.cuh"

namespace msda_split {

// One kernel, two reductions, chosen by ``check_backward_warp``: warp shuffles
// when a head's lanes fit one warp, otherwise shared-memory tiles behind
// uniform barriers.
template <typename scalar_t, int64_t channels_per_thread, typename channel_vector_t,
          int64_t L, int64_t K, int apply_softmax, bool use_warp_reduction,
          typename grad_value_t, bool softmax_uses_warp, bool allow_half2_atomics>
// The hardware maximum, not tiers: a tighter bound does not speed up the
// backward, and a second tier would double the larger kernel's instantiations.
__global__ void __launch_bounds__(1024)
    backward_kernel(const BackwardArgs<scalar_t, grad_value_t> args) {
    const scalar_t *const __restrict__ value = args.value;
    const int64_t *const __restrict__ spatial_shapes = args.spatial_shapes;
    const int64_t *const __restrict__ level_start_index = args.level_start_index;
    const accum_t<scalar_t> *const __restrict__ location = args.location;
    const scalar_t *const __restrict__ attention_weight = args.attention_weight;
    const scalar_t *const __restrict__ grad_output = args.grad_output;
    grad_value_t *const __restrict__ grad_value = args.grad_value;
    accum_t<scalar_t> *const __restrict__ grad_location_out = args.grad_location;
    accum_t<scalar_t> *const __restrict__ grad_attention_weight_out =
        args.grad_attention_weight;
    const int64_t N = args.N;
    const int64_t G = args.G;
    const int64_t D = args.D;
    const int64_t Q = args.Q;
    const int64_t queries_per_block = args.queries_per_block;

    extern __shared__ char _s[];
    const int64_t batch_query_index = blockIdx.x * queries_per_block + threadIdx.z;
    const int64_t qi = batch_query_index % Q;
    const int64_t bi = batch_query_index / Q;

    // 32-bit per-point index math (bilinear, shared and output tiles); base pointers
    // and query bookkeeping are 64-bit. The host TORCH_CHECK enforces the bound.
    const int32_t channel_offset = threadIdx.x * channels_per_thread;
    const int32_t gi = threadIdx.y;
    const int32_t lanes_per_head = D / channels_per_thread;
    const int32_t head_slot = threadIdx.z * G + gi;

    // Warp-shuffle bookkeeping is valid (and used) only when a head fits in one
    // warp, which is the warp-reduction path.
    unsigned lane_mask = 0;
    int64_t head_index_in_warp = 0;
    if constexpr (use_warp_reduction) {
        const int64_t heads_per_warp =
            blockDim.x <= kWarpSize ? kWarpSize / blockDim.x : 1;
        head_index_in_warp = (threadIdx.z * G + threadIdx.y) % heads_per_warp;
        lane_mask = deform_utils::get_lane_mask(blockDim.x, head_index_in_warp);
    }

    // Shared memory: the attention-gradient tile always, plus the per-lane gradient
    // tile and per-head location slot on the shared-reduction path.
    accum_t<scalar_t> *shared_grad_attention_weight_before_softmax =
        (accum_t<scalar_t> *)(_s);
    accum_t<scalar_t> *tile_cursor =
        shared_grad_attention_weight_before_softmax +
        BackwardSharedExtents<channels_per_thread>::attention_tile(queries_per_block, G,
                                                                   L, K);
    accum_t<scalar_t> *shared_grad_sampling_parameters = nullptr;
    accum_t<scalar_t> *shared_location = nullptr;
    if constexpr (!use_warp_reduction) {
        shared_grad_sampling_parameters = tile_cursor;
        // Both sides read the partition from one description, so the offsets
        // the kernel walks and the size the host allocates cannot drift apart.
        tile_cursor += BackwardSharedExtents<channels_per_thread>::parameters(
            queries_per_block, G, D);
        shared_location = tile_cursor;
        tile_cursor +=
            BackwardSharedExtents<channels_per_thread>::locations(queries_per_block, G);
    }
    accum_t<scalar_t> *const shared_attention_weight_tile =
        tile_cursor + head_slot * L * K;

    const accum_t<scalar_t> *location_pointer =
        location + (((bi * Q + qi) * G + gi) * L) * K * 2;
    const scalar_t *attention_weight_pointer =
        attention_weight + (((bi * Q + qi) * G + gi) * L) * K;
    const scalar_t *grad_output_pointer =
        grad_output + ((bi * Q + qi) * G + gi) * D + channel_offset;

    const int64_t num_iterations = (L * K) / lanes_per_head;
    const int64_t remainder = (L * K) - num_iterations * lanes_per_head;
    for (int i = 0; i < num_iterations; i++)
        shared_attention_weight_tile[lanes_per_head * i + threadIdx.x] =
            (accum_t<scalar_t>)
                attention_weight_pointer[lanes_per_head * i + threadIdx.x];
    if (remainder > 0 && threadIdx.x < remainder)
        shared_attention_weight_tile[lanes_per_head * num_iterations + threadIdx.x] =
            (accum_t<scalar_t>)
                attention_weight_pointer[lanes_per_head * num_iterations + threadIdx.x];
    __syncthreads();

    if (apply_softmax == 1) {
        if constexpr (softmax_uses_warp && use_warp_reduction) {
            deform_utils::softmax_forward_warp(shared_attention_weight_tile, L * K,
                                               lane_mask);
            // Every lane reads entries a sibling normalized, so the lanes meet first.
            __syncthreads();
        } else {
            if (threadIdx.x == 0)
                deform_utils::softmax_forward_thread0(shared_attention_weight_tile,
                                                      L * K);
            __syncthreads();
        }
    }

    if (apply_softmax == 1 && threadIdx.x == 0)
        for (int bsk = 0; bsk < L * K; bsk++)
            shared_grad_attention_weight_before_softmax[head_slot * L * K + bsk] = 0;

    int32_t location_index = 0;
    int32_t attention_index = 0;
    const int32_t w_stride = G * D;
    const int32_t base_offset = gi * D + channel_offset;
    for (int li = 0; li < L; li++) {
        const int32_t spatial_h = spatial_shapes[li * 2];
        const int32_t spatial_w = spatial_shapes[li * 2 + 1];
        const int64_t level_start = level_start_index[li];
        const scalar_t *value_pointer = value + (bi * N + level_start) * G * D;
        grad_value_t *grad_value_pointer = grad_value + (bi * N + level_start) * G * D;
        const int32_t parameter_slot = (head_slot * blockDim.x + threadIdx.x) * 3;
        accum_t<scalar_t> grad_sampling_parameters[3] = {0., 0., 0.};
        for (int ki = 0; ki < K; ki++) {
            // Broadcast the sampled location across the head.
            // Zero on the lanes a shuffle overwrites, so none reads garbage.
            accum_t<scalar_t> location_w = 0, location_h = 0;
            if constexpr (use_warp_reduction) {
                if (threadIdx.x == 0) {
                    location_w = (accum_t<scalar_t>)location_pointer[location_index];
                    location_h =
                        (accum_t<scalar_t>)location_pointer[location_index + 1];
                }
                location_w =
                    __shfl_sync(lane_mask, location_w, head_index_in_warp * blockDim.x);
                location_h =
                    __shfl_sync(lane_mask, location_h, head_index_in_warp * blockDim.x);
            } else {
                if (threadIdx.x == 0) {
                    shared_location[head_slot * 2] =
                        (accum_t<scalar_t>)location_pointer[location_index];
                    shared_location[head_slot * 2 + 1] =
                        (accum_t<scalar_t>)location_pointer[location_index + 1];
                }
                __syncthreads();
                location_w = shared_location[head_slot * 2];
                location_h = shared_location[head_slot * 2 + 1];
            }
            const accum_t<scalar_t> point_weight =
                shared_attention_weight_tile[attention_index];
            const accum_t<scalar_t> pixel_h = location_h * spatial_h - 0.5;
            const accum_t<scalar_t> pixel_w = location_w * spatial_w - 0.5;
            const bool in_bounds = (pixel_h >= -1 && pixel_w >= -1 &&
                                    pixel_h < spatial_h && pixel_w < spatial_w);
            const int64_t location_out_base =
                ((bi * Q + qi) * G + gi) * L * K * 2 + li * K * 2 + ki * 2;
            const int64_t attention_out_index =
                ((bi * Q + qi) * G + gi) * L * K + li * K + ki;
            const int64_t before_softmax_index = (head_slot * L + li) * K + ki;

            if constexpr (use_warp_reduction) {
                grad_sampling_parameters[0] = 0;
                grad_sampling_parameters[1] = 0;
                grad_sampling_parameters[2] = 0;
                if (in_bounds) {
                    deform_interpolate::col2im_bilinear<
                        scalar_t, channel_vector_t, channels_per_thread, grad_value_t,
                        allow_half2_atomics>(value_pointer, spatial_h, spatial_w,
                                             pixel_h, pixel_w, point_weight, w_stride,
                                             base_offset, (accum_t<scalar_t>)spatial_h,
                                             (accum_t<scalar_t>)spatial_w,
                                             grad_output_pointer, grad_value_pointer,
                                             grad_sampling_parameters);
                }
                // Every lane, out-of-bounds ones carrying zero, calls the collective
                // shuffle, never behind a divergent branch.
                deform_utils::warp_reduce_grad_sampling_parameters(
                    grad_sampling_parameters, lane_mask);
                if (threadIdx.x == 0 && in_bounds) {
                    grad_location_out[location_out_base] = grad_sampling_parameters[0];
                    grad_location_out[location_out_base + 1] =
                        grad_sampling_parameters[1];
                    if (apply_softmax == 1)
                        shared_grad_attention_weight_before_softmax
                            [before_softmax_index] = grad_sampling_parameters[2];
                    else
                        grad_attention_weight_out[attention_out_index] =
                            grad_sampling_parameters[2];
                }
            } else {
                // Shared-memory reduction: every thread reaches each barrier; the
                // bounds test skips only the work and the final write.
                shared_grad_sampling_parameters[parameter_slot] = 0;
                shared_grad_sampling_parameters[parameter_slot + 1] = 0;
                shared_grad_sampling_parameters[parameter_slot + 2] = 0;
                if (in_bounds) {
                    deform_interpolate::col2im_bilinear<
                        scalar_t, channel_vector_t, channels_per_thread, grad_value_t,
                        allow_half2_atomics>(
                        value_pointer, spatial_h, spatial_w, pixel_h, pixel_w,
                        point_weight, w_stride, base_offset,
                        (accum_t<scalar_t>)spatial_h, (accum_t<scalar_t>)spatial_w,
                        grad_output_pointer, grad_value_pointer,
                        shared_grad_sampling_parameters + parameter_slot);
                }
                __syncthreads();
                if (threadIdx.x == 0 && in_bounds) {
                    accum_t<scalar_t> reduced[3];
                    deform_utils::block_reduce_grad_sampling_parameters(
                        shared_grad_sampling_parameters, head_slot, reduced);
                    grad_location_out[location_out_base] = reduced[0];
                    grad_location_out[location_out_base + 1] = reduced[1];
                    if (apply_softmax == 1)
                        shared_grad_attention_weight_before_softmax
                            [before_softmax_index] = reduced[2];
                    else
                        grad_attention_weight_out[attention_out_index] = reduced[2];
                }
                __syncthreads();
            }
            location_index += 2;
            attention_index += 1;
        }
    }
    if (apply_softmax == 1 && threadIdx.x == 0) {
        const accum_t<scalar_t> *head_grad_attention_weight =
            shared_grad_attention_weight_before_softmax + head_slot * L * K;
        deform_utils::softmax_backward_thread0(
            &grad_attention_weight_out[((bi * Q + qi) * G + gi) * L * K],
            shared_attention_weight_tile, head_grad_attention_weight, L * K);
    }
}
template <typename scalar_t, typename channel_vector_t, int64_t K,
          int64_t channels_per_thread, int apply_softmax, typename grad_value_t,
          bool softmax_uses_warp, bool allow_half2_atomics>
void _backward_launch(cudaStream_t stream, BackwardArgs<scalar_t, grad_value_t> args,
                      const int64_t B, const int64_t L,
                      const int64_t threads_per_block) {
    CHECK_DIVISIBLE(args.D, channels_per_thread);
    const auto geometry = deform_utils::compute_block_geometry(
        B, args.Q, args.G, args.D, channels_per_thread, threads_per_block);
    args.queries_per_block = geometry.queries_per_block;
    // A zero or non-dividing block size would silently drop the tail queries.
    TORCH_CHECK(geometry.queries_per_block >= 1, "threads_per_block (",
                threads_per_block, ") is too small for one query at G=", args.G,
                ", D=", args.D, ", channels_per_thread=", channels_per_thread);
    CHECK_DIVISIBLE((B * args.Q), geometry.queries_per_block);
    const int64_t block_threads = block_thread_count(geometry.block);
    TORCH_CHECK(block_threads <= kMaxThreadsPerBlock, "threads_per_block (",
                threads_per_block, ") needs a block of ", block_threads,
                " threads at G=", args.G, ", D=", args.D,
                ", channels_per_thread=", channels_per_thread,
                ", above the hardware maximum of ", kMaxThreadsPerBlock);

    // The reduction split picks the specialization and the shared-memory budget.
    const bool use_warp_reduction = check_backward_warp(channels_per_thread, args.D);
    const int64_t shm_size =
        BackwardSharedExtents<channels_per_thread>::template bytes<scalar_t>(
            geometry.queries_per_block, args.G, args.D, L, K, use_warp_reduction);

    void (*kernel)(BackwardArgs<scalar_t, grad_value_t>) = nullptr;
    // Bound to a variable first: the commas in the template argument list would
    // otherwise be read as further macro arguments by TORCH_CHECK.
    const bool dispatched =
        deform_utils::dispatch_constant<1, 2, 3, 4, 5>(L, [&]<int64_t Levels>() {
            if (use_warp_reduction)
                kernel =
                    backward_kernel<scalar_t, channels_per_thread, channel_vector_t,
                                    Levels, K, apply_softmax, true, grad_value_t,
                                    softmax_uses_warp, allow_half2_atomics>;
            else
                kernel =
                    backward_kernel<scalar_t, channels_per_thread, channel_vector_t,
                                    Levels, K, apply_softmax, false, grad_value_t,
                                    softmax_uses_warp, allow_half2_atomics>;
        });
    TORCH_CHECK(dispatched, kLevelBoundMessage, L);
    TORCH_CHECK(cudaFuncSetAttribute(kernel,
                                     cudaFuncAttributeMaxDynamicSharedMemorySize,
                                     shm_size) == cudaSuccess,
                "msda_split::backward_kernel cannot be given ", shm_size,
                " bytes of dynamic shared memory");
    kernel<<<geometry.grid, geometry.block, shm_size, stream>>>(args);
    check_launch("msda_split::backward_kernel");
}

template <typename scalar_t, int64_t K, int apply_softmax, typename grad_value_t,
          bool softmax_uses_warp, bool allow_half2_atomics>
void backward_launch_inner(cudaStream_t stream,
                           const BackwardArgs<scalar_t, grad_value_t> &args,
                           const int64_t B, const int64_t L,
                           const int64_t channels_per_thread,
                           const int64_t threads_per_block) {
    // One dispatch over the channel count; fp16, bf16 and fp32 share these rungs.
    const auto launch = [&]<int64_t N>() {
        _backward_launch<scalar_t, deform_utils::channel_vector_t_for<scalar_t, N>, K,
                         N, apply_softmax, grad_value_t, softmax_uses_warp,
                         allow_half2_atomics>(stream, args, B, L, threads_per_block);
    };
    // `backward_thread_layout` emits only 1 or 2 (`2 if C >= 64 else 1`), and
    // `test_dispatch_packs.py` checks that this pack matches. Bound first for
    // TORCH_CHECK.
    const bool dispatched =
        deform_utils::dispatch_constant<1, 2>(channels_per_thread, launch);
    TORCH_CHECK(dispatched, "unsupported channels_per_thread: ", channels_per_thread);
}

template <typename scalar_t, typename grad_value_t, bool softmax_uses_warp,
          bool allow_half2_atomics>
void backward_launch(cudaStream_t stream,
                     const BackwardArgs<scalar_t, grad_value_t> &args, const int64_t B,
                     const int64_t L, const int64_t K,
                     const int64_t channels_per_thread, const int64_t threads_per_block,
                     int64_t apply_softmax) {
    // Bound first for TORCH_CHECK. An unsupported K would launch nothing and
    // return the allocation's zeros.
    const bool dispatched =
        deform_utils::dispatch_constant<4, 8>(K, [&]<int64_t Points>() {
            if (apply_softmax == 1)
                backward_launch_inner<scalar_t, Points, 1, grad_value_t,
                                      softmax_uses_warp, allow_half2_atomics>(
                    stream, args, B, L, channels_per_thread, threads_per_block);
            else
                backward_launch_inner<scalar_t, Points, 0, grad_value_t,
                                      softmax_uses_warp, allow_half2_atomics>(
                    stream, args, B, L, channels_per_thread, threads_per_block);
        });
    TORCH_CHECK(dispatched, "unsupported points per level: ", K);
}

} // namespace msda_split
