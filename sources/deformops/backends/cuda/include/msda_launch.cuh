// Host entry points: batch chunking, output and gradient allocation, and the
// scalar-type dispatch into the forward and backward kernels.
#pragma once

#include <utility>

#include "msda_args.cuh"
#include "msda_backward.cuh"
#include "msda_forward.cuh"

namespace msda_split {

// Takes ``apply_softmax`` at runtime; each registered op fixes it to 0 or 1.
namespace impl {

// The largest divisor of ``batch`` that is at most ``chunk_step``.
inline int64_t batch_chunk(int64_t batch, int64_t chunk_step) {
    int64_t step = std::max<int64_t>(1, std::min(batch, chunk_step));
    while (batch % step != 0)
        --step;
    return step;
}

// Forward geometry: eight channels per thread, and the most queries per block
// (up to 64, dividing ``batch * queries``) within 512 threads.
inline std::pair<int64_t, int64_t>
forward_thread_layout(int64_t batch, int64_t queries, int64_t heads, int64_t channels) {
    const int64_t channels_per_thread = 8;
    const int64_t total = batch * queries;
    int64_t queries_per_block = 1;
    for (int64_t m = 1; m <= 64 && m <= total; ++m)
        if (total % m == 0 && m * heads * channels / channels_per_thread <= 512)
            queries_per_block = m;
    return {channels_per_thread,
            queries_per_block * heads * channels / channels_per_thread};
}

// The backward counterpart: one or two channels per thread (two from 64
// channels per head) and a 256-thread block preference.
inline std::pair<int64_t, int64_t> backward_thread_layout(int64_t batch,
                                                          int64_t queries,
                                                          int64_t heads,
                                                          int64_t channels) {
    const int64_t channels_per_thread = channels >= 64 ? 2 : 1;
    const int64_t total = batch * queries;
    int64_t queries_per_block = 1;
    for (int64_t m = 1; m <= 64 && m <= total; ++m)
        if (total % m == 0 && m * heads * channels / channels_per_thread <= 256)
            queries_per_block = m;
    return {channels_per_thread,
            queries_per_block * heads * channels / channels_per_thread};
}

// Locations must be float32, or float64 alongside a float64 value tensor.
inline void check_location_dtype(const Tensor &value, const Tensor &location) {
    const bool double_input = value.scalar_type() == ScalarType::Double;
    const auto expected = double_input ? ScalarType::Double : ScalarType::Float;
    TORCH_CHECK(location.scalar_type() == expected, "sampling locations must be ",
                double_input ? "float64" : "float32",
                " for a value tensor of this dtype");
}

inline Tensor forward_cuda(const Tensor &value, const Tensor &spatial_shapes,
                           const Tensor &level_start_index, const Tensor &location,
                           const Tensor &attention_weight, const int64_t chunk_step,
                           const int64_t K, const int64_t channels_per_thread,
                           const int64_t threads_per_block, int64_t apply_softmax) {
    const int64_t batch = value.size(0);
    const int64_t spatial_size = value.size(1);
    const int64_t num_heads = value.size(2);
    const int64_t num_channels = value.size(3);
    const int64_t num_levels = spatial_shapes.size(0);
    const int64_t num_query = location.size(1);
    TORCH_CHECK(batch > 0, "batch must be positive, got ", batch);
    TORCH_CHECK(chunk_step > 0, "chunk_step must be positive, got ", chunk_step);
    check_location_dtype(value, location);
    // The host loop strides the batch in chunks that must divide it.
    const int64_t chunk = batch_chunk(batch, chunk_step);
    const auto [channels_per_thread_, threads_per_block_] =
        channels_per_thread > 0 && threads_per_block > 0
            ? std::pair{channels_per_thread, threads_per_block}
            : forward_thread_layout(chunk, location.size(1), value.size(2),
                                    value.size(3));

    // Within-level indexing is 32-bit (see interpolate.cuh): reject S*G*D past
    // INT32_MAX instead of wrapping.
    TORCH_CHECK(spatial_size * num_heads * num_channels <= INT32_MAX,
                "value tensor too large for 32-bit indexing: S*G*D=",
                spatial_size * num_heads * num_channels, " exceeds INT32_MAX");

    // Allocate with final merged shape {B, Q, G*D} to avoid a .view() call.
    int64_t out_shape[] = {batch, num_query, num_heads * num_channels};
    AtenTensorHandle out_h;
    aoti_torch_aten_new_zeros(value.get(), out_shape, 3, nullptr, nullptr, nullptr, -1,
                              nullptr, &out_h);
    Tensor output(out_h);

    const auto stream = stable_compat::current_stream_for(value);

    const auto per_value_size = spatial_size * num_heads * num_channels;
    const auto per_loc_size = num_query * num_heads * num_levels * K * 2;
    const auto per_attn_size = num_query * num_heads * num_levels * K;
    const auto per_out_size = num_query * num_heads * num_channels;

    auto stype = value.scalar_type();
    auto dispatch = [&]<typename scalar_t>() {
        for (int n = 0; n < batch / chunk; ++n) {
            const ForwardArgs<scalar_t> args{
                .value = static_cast<const scalar_t *>(value.data_ptr()) +
                         n * chunk * per_value_size,
                .spatial_shapes =
                    static_cast<const int64_t *>(spatial_shapes.data_ptr()),
                .level_start_index =
                    static_cast<const int64_t *>(level_start_index.data_ptr()),
                .location =
                    static_cast<const accum_t<scalar_t> *>(location.data_ptr()) +
                    n * chunk * per_loc_size,
                .attention_weight =
                    static_cast<const scalar_t *>(attention_weight.data_ptr()) +
                    n * chunk * per_attn_size,
                .output = static_cast<scalar_t *>(output.data_ptr()) +
                          n * chunk * per_out_size,
                .N = spatial_size,
                .G = num_heads,
                .D = num_channels,
                .Q = num_query,
                .queries_per_block = 0, // set inside _forward_launch
            };
            forward_launch(stream, args, chunk, num_levels, K, channels_per_thread_,
                           threads_per_block_, apply_softmax);
        }
    };
    if (!stable_compat::dispatch_scalar_type(stype, dispatch))
        throw std::runtime_error(stable_compat::unsupported_dtype_message(
            "msda_split::forward_cuda", stype));

    return output;
}

// See msda_boxed.cuh for ``allow_half2_atomics``. Defaults to packing.
template <bool native_grad_value, bool softmax_uses_warp,
          bool allow_half2_atomics = true>
std::vector<Tensor> backward_cuda(
    const Tensor &value, const Tensor &spatial_shapes, const Tensor &level_start_index,
    const Tensor &location, const Tensor &attention_weight, const Tensor &grad_output,
    const int64_t chunk_step, const int64_t K, const int64_t channels_per_thread,
    const int64_t threads_per_block, int64_t apply_softmax) {
    const int64_t batch = value.size(0);
    const int64_t spatial_size = value.size(1);
    const int64_t num_heads = value.size(2);
    const int64_t num_channels = value.size(3);
    const int64_t num_levels = spatial_shapes.size(0);
    const int64_t num_query = location.size(1);
    TORCH_CHECK(batch > 0, "batch must be positive, got ", batch);
    TORCH_CHECK(chunk_step > 0, "chunk_step must be positive, got ", chunk_step);
    check_location_dtype(value, location);
    // The host loop strides the batch in chunks that must divide it.
    const int64_t chunk = batch_chunk(batch, chunk_step);
    const auto [channels_per_thread_, threads_per_block_] =
        channels_per_thread > 0 && threads_per_block > 0
            ? std::pair{channels_per_thread, threads_per_block}
            : backward_thread_layout(chunk, location.size(1), value.size(2),
                                     value.size(3));

    // 32-bit within-level indexing (see forward_cuda); reject overflowing inputs.
    TORCH_CHECK(spatial_size * num_heads * num_channels <= INT32_MAX,
                "value tensor too large for 32-bit indexing: S*G*D=",
                spatial_size * num_heads * num_channels, " exceeds INT32_MAX");

    // The value gradient is scalar_t; location and weight gradients match the
    // kernel's accum_t<scalar_t> write width and the inputs' flat or expanded shapes.
    int64_t val_shape[] = {batch, spatial_size, num_heads, num_channels};
    int64_t loc_ndim = location.dim();
    TORCH_CHECK(loc_ndim <= 8, "location has ", loc_ndim, " dims; expected <= 8");
    int64_t loc_shape[8];
    for (int64_t i = 0; i < loc_ndim; i++)
        loc_shape[i] = location.size(i);
    int64_t attn_ndim = attention_weight.dim();
    TORCH_CHECK(attn_ndim <= 8, "attention_weight has ", attn_ndim,
                " dims; expected <= 8");
    int64_t attn_shape[8];
    for (int64_t i = 0; i < attn_ndim; i++)
        attn_shape[i] = attention_weight.size(i);
    int32_t f32_dtype = aoti_torch_dtype_float32();
    int32_t f64_dtype = aoti_torch_dtype_float64();
    const bool double_input = (value.scalar_type() == ScalarType::Double);
    int32_t *opmath_dtype = double_input ? &f64_dtype : &f32_dtype;
    // native_grad_value=false accumulates grad_value in an fp32 opmath scratch
    // tensor (cast back at the tail); true keeps the native scalar_t tensor.
    int32_t *grad_input_dtype = native_grad_value ? nullptr : opmath_dtype;
    AtenTensorHandle gi_h, gl_h, ga_h;
    aoti_torch_aten_new_zeros(value.get(), val_shape, 4, grad_input_dtype, nullptr,
                              nullptr, -1, nullptr, &gi_h);
    aoti_torch_aten_new_zeros(location.get(), loc_shape, loc_ndim, opmath_dtype,
                              nullptr, nullptr, -1, nullptr, &gl_h);
    aoti_torch_aten_new_zeros(attention_weight.get(), attn_shape, attn_ndim,
                              opmath_dtype, nullptr, nullptr, -1, nullptr, &ga_h);
    Tensor grad_input(gi_h), grad_location(gl_h), grad_attention_weight(ga_h);

    const auto stream = stable_compat::current_stream_for(value);

    const auto per_value_size = spatial_size * num_heads * num_channels;
    const auto per_loc_size = num_query * num_heads * num_levels * K * 2;
    const auto per_attn_size = num_query * num_heads * num_levels * K;
    const auto per_out_size = num_query * num_heads * num_channels;

    auto stype = value.scalar_type();
    auto dispatch = [&]<typename scalar_t>() {
        using grad_value_t =
            std::conditional_t<native_grad_value, scalar_t, accum_t<scalar_t>>;
        for (int n = 0; n < batch / chunk; ++n) {
            const BackwardArgs<scalar_t, grad_value_t> args{
                .value = static_cast<const scalar_t *>(value.data_ptr()) +
                         n * chunk * per_value_size,
                .spatial_shapes =
                    static_cast<const int64_t *>(spatial_shapes.data_ptr()),
                .level_start_index =
                    static_cast<const int64_t *>(level_start_index.data_ptr()),
                .location =
                    static_cast<const accum_t<scalar_t> *>(location.data_ptr()) +
                    n * chunk * per_loc_size,
                .attention_weight =
                    static_cast<const scalar_t *>(attention_weight.data_ptr()) +
                    n * chunk * per_attn_size,
                .grad_output = static_cast<const scalar_t *>(grad_output.data_ptr()) +
                               n * chunk * per_out_size,
                .grad_value = static_cast<grad_value_t *>(grad_input.data_ptr()) +
                              n * chunk * per_value_size,
                .grad_location =
                    static_cast<accum_t<scalar_t> *>(grad_location.data_ptr()) +
                    n * chunk * per_loc_size,
                .grad_attention_weight =
                    static_cast<accum_t<scalar_t> *>(grad_attention_weight.data_ptr()) +
                    n * chunk * per_attn_size,
                .N = spatial_size,
                .G = num_heads,
                .D = num_channels,
                .Q = num_query,
                .queries_per_block = 0, // set inside _backward_launch
            };
            backward_launch<scalar_t, grad_value_t, softmax_uses_warp,
                            allow_half2_atomics>(stream, args, chunk, num_levels, K,
                                                 channels_per_thread_,
                                                 threads_per_block_, apply_softmax);
        }
    };
    if (!stable_compat::dispatch_scalar_type(stype, dispatch))
        throw std::runtime_error(stable_compat::unsupported_dtype_message(
            "msda_split::backward_cuda", stype));

    // Cast float32 gradients back to the location and weight dtypes.
    if (location.scalar_type() != ScalarType::Float) {
        AtenTensorHandle hl_h, ha_h;
        aoti_torch_aten_new_empty(location.get(), loc_shape, loc_ndim, nullptr, nullptr,
                                  nullptr, -1, nullptr, &hl_h);
        aoti_torch_aten_new_empty(attention_weight.get(), attn_shape, attn_ndim,
                                  nullptr, nullptr, nullptr, -1, nullptr, &ha_h);
        aoti_torch_copy_(hl_h, grad_location.get(), 0);
        aoti_torch_copy_(ha_h, grad_attention_weight.get(), 0);
        grad_location = Tensor(hl_h);
        grad_attention_weight = Tensor(ha_h);
    }
    // fp32-scratch grad_value (native_grad_value=false): cast back to the value
    // dtype; the native path leaves grad_input at scalar_t already.
    if constexpr (!native_grad_value) {
        if (value.scalar_type() != ScalarType::Float) {
            AtenTensorHandle hi_h;
            aoti_torch_aten_new_empty(value.get(), val_shape, 4, nullptr, nullptr,
                                      nullptr, -1, nullptr, &hi_h);
            aoti_torch_copy_(hi_h, grad_input.get(), 0);
            grad_input = Tensor(hi_h);
        }
    }
    return {grad_input, grad_location, grad_attention_weight};
}

} // namespace impl

} // namespace msda_split
