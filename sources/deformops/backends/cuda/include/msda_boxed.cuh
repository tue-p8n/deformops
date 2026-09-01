// Boxed stable-ABI entry points for the split-layout MSDA operators.
//
// Shared by every configuration, which differ only in backward traits and op names.
// ``torch::stable::detail::{to,from}``, not the global ``to``/``from``: those are
// deprecated shims that warn on every use.
#pragma once

#include "msda_launch.cuh"

namespace msda_split {
namespace boxed {

// Trait-free: configurations differ only in whether the in-kernel softmax runs.
template <int64_t apply_softmax>
void forward(StableIValue *stack, uint64_t, uint64_t) {
    auto value = torch::stable::detail::to<Tensor>(stack[0]);
    auto spatial_shapes = torch::stable::detail::to<Tensor>(stack[1]);
    auto level_start_index = torch::stable::detail::to<Tensor>(stack[2]);
    auto location = torch::stable::detail::to<Tensor>(stack[3]);
    auto attention_weight = torch::stable::detail::to<Tensor>(stack[4]);
    auto chunk_step = torch::stable::detail::to<int64_t>(stack[5]);
    auto K = torch::stable::detail::to<int64_t>(stack[6]);
    auto channels_per_thread = torch::stable::detail::to<int64_t>(stack[7]);
    auto threads_per_block = torch::stable::detail::to<int64_t>(stack[8]);
    stack[0] = torch::stable::detail::from(impl::forward_cuda(
        value, spatial_shapes, level_start_index, location, attention_weight,
        chunk_step, K, channels_per_thread, threads_per_block, apply_softmax));
}

// ``allow_half2_atomics`` matters only with ``native_grad_value`` and a 16-bit
// dtype; the fp32-accumulator path uses ``atomicAdd`` either way.
template <bool native_grad_value, bool softmax_uses_warp, bool allow_half2_atomics,
          int64_t apply_softmax>
void backward(StableIValue *stack, uint64_t, uint64_t) {
    auto value = torch::stable::detail::to<Tensor>(stack[0]);
    auto spatial_shapes = torch::stable::detail::to<Tensor>(stack[1]);
    auto level_start_index = torch::stable::detail::to<Tensor>(stack[2]);
    auto location = torch::stable::detail::to<Tensor>(stack[3]);
    auto attention_weight = torch::stable::detail::to<Tensor>(stack[4]);
    auto grad_output = torch::stable::detail::to<Tensor>(stack[5]);
    auto chunk_step = torch::stable::detail::to<int64_t>(stack[6]);
    auto K = torch::stable::detail::to<int64_t>(stack[7]);
    auto channels_per_thread = torch::stable::detail::to<int64_t>(stack[8]);
    auto threads_per_block = torch::stable::detail::to<int64_t>(stack[9]);
    auto grads =
        impl::backward_cuda<native_grad_value, softmax_uses_warp, allow_half2_atomics>(
            value, spatial_shapes, level_start_index, location, attention_weight,
            grad_output, chunk_step, K, channels_per_thread, threads_per_block,
            apply_softmax);
    stack[0] = torch::stable::detail::from(grads[0]);
    stack[1] = torch::stable::detail::from(grads[1]);
    stack[2] = torch::stable::detail::from(grads[2]);
}

} // namespace boxed
} // namespace msda_split
