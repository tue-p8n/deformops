#pragma once

#include <algorithm>
#include <cstdio>
#include <cstring>

#include <type_traits>

#include "utils.cuh"

namespace deform_interpolate {

// Bilinear sampling (aka 'im2col') for input feature map.
template <typename scalar_t, typename channel_vector_t, int64_t channels_per_thread>
__device__ void im2col_bilinear(accum_t<scalar_t> output_accumulator[],
                                const scalar_t *value, int32_t height, int32_t width,
                                accum_t<scalar_t> pixel_h, accum_t<scalar_t> pixel_w,
                                accum_t<scalar_t> attention_weight, int32_t w_stride,
                                int32_t base_offset) {
    // 32-bit offsets: the host rejects levels whose S*G*D could exceed INT32_MAX,
    // and the 64-bit level base pointer is added at the call site.
    using acc_t = accum_t<scalar_t>;

    // Top-left corner of the sample's cell, and the fraction past it on each axis.
    const int32_t row_origin = floor(pixel_h);
    const int32_t col_origin = floor(pixel_w);
    const acc_t row_frac = pixel_h - row_origin;
    const acc_t col_frac = pixel_w - col_origin;

    // Per-axis weights, near corner first; a corner's weight is their product.
    const acc_t row_weight[2] = {acc_t(1) - row_frac, row_frac};
    const acc_t col_weight[2] = {acc_t(1) - col_frac, col_frac};

    // The caller guarantees -1 <= pixel < extent, so one test per corner suffices.
    const bool row_inside[2] = {row_origin >= 0, row_origin + 1 < height};
    const bool col_inside[2] = {col_origin >= 0, col_origin + 1 < width};

    const int32_t row_stride = width * w_stride;
    const int32_t origin_offset = row_origin * row_stride + col_origin * w_stride;

    acc_t sample[channels_per_thread];
#pragma unroll
    for (int c = 0; c < channels_per_thread; c++) {
        sample[c] = acc_t(0);
    }

    // Near-to-far order fixes the summation order; corners outside the level
    // contribute zero, the reference's zero padding.
#pragma unroll
    for (int dr = 0; dr < 2; dr++) {
#pragma unroll
        for (int dc = 0; dc < 2; dc++) {
            if (!(row_inside[dr] && col_inside[dc]))
                continue;

            alignas(channel_vector_t) scalar_t corner[channels_per_thread];
            deform_utils::vectorized_copy<scalar_t, channel_vector_t,
                                          channels_per_thread>(
                corner,
                value + origin_offset + dr * row_stride + dc * w_stride + base_offset);

            const acc_t weight = row_weight[dr] * col_weight[dc];
#pragma unroll
            for (int c = 0; c < channels_per_thread; c++)
                sample[c] += weight * static_cast<acc_t>(corner[c]);
        }
    }

#pragma unroll
    for (int c = 0; c < channels_per_thread; c++)
        output_accumulator[c] += attention_weight * sample[c];
}

// Backward of the bilinear sample: scatters into ``grad_value`` and writes the
// (location_x, location_y, attention_weight) gradients. ``grad_value``'s element
// type selects the accumulation: ``accum_t<scalar_t>`` adds per element; a native
// 16-bit ``scalar_t`` packs half2 atomics. For fp32 both are plain ``atomicAdd``.
template <typename scalar_t, typename channel_vector_t, int64_t channels_per_thread,
          typename grad_value_t, bool allow_half2_atomics = true>
__device__ void col2im_bilinear(
    const scalar_t *value, int32_t height, int32_t width, accum_t<scalar_t> pixel_h,
    accum_t<scalar_t> pixel_w, accum_t<scalar_t> attention_weight, int32_t w_stride,
    int32_t base_offset, const accum_t<scalar_t> location_scale_h,
    const accum_t<scalar_t> location_scale_w, const scalar_t *grad_output_pointer,
    grad_value_t *grad_value, accum_t<scalar_t> *grad_sampling_parameters) {
    // Native 16-bit gradients pack two channels per atomic unless unpacked atomics
    // are requested.
    constexpr bool native_16bit = deform_utils::is_16bit<scalar_t>::value &&
                                  std::is_same_v<grad_value_t, scalar_t>;
    constexpr bool use_half2_atomics = native_16bit && allow_half2_atomics;

    // 32-bit index arithmetic (see im2col_bilinear); scatter addresses are
    // formed as `grad_value + int`, keeping the 64-bit base pointer.
    const int32_t h_low = floor(pixel_h);
    const int32_t w_low = floor(pixel_w);
    const int32_t h_high = h_low + 1;
    const int32_t w_high = w_low + 1;
    const accum_t<scalar_t> lh = pixel_h - h_low;
    const accum_t<scalar_t> lw = pixel_w - w_low;
    const accum_t<scalar_t> hh = 1 - lh;
    const accum_t<scalar_t> hw = 1 - lw;

    const accum_t<scalar_t> w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;
    const accum_t<scalar_t> weights[4] = {w1, w2, w3, w4};

    alignas(channel_vector_t) scalar_t grad_output_values[channels_per_thread];
    deform_utils::vectorized_copy<scalar_t, channel_vector_t, channels_per_thread>(
        grad_output_values, grad_output_pointer);
    accum_t<scalar_t> grad_output_opmath[channels_per_thread];
#pragma unroll
    for (int i = 0; i < channels_per_thread; ++i)
        grad_output_opmath[i] = (accum_t<scalar_t>)grad_output_values[i];

    const int32_t h_stride = width * w_stride;
    const int32_t h_low_ptr_offset = h_low * h_stride;
    const int32_t h_high_ptr_offset = h_low_ptr_offset + h_stride;
    const int32_t w_low_ptr_offset = w_low * w_stride;
    const int32_t w_high_ptr_offset = w_low_ptr_offset + w_stride;

    int32_t idx[4];
    idx[0] = h_low_ptr_offset + w_low_ptr_offset + base_offset;
    idx[1] = h_low_ptr_offset + w_high_ptr_offset + base_offset;
    idx[2] = h_high_ptr_offset + w_low_ptr_offset + base_offset;
    idx[3] = h_high_ptr_offset + w_high_ptr_offset + base_offset;

    bool cond[4];
    cond[0] = (h_low >= 0 && w_low >= 0);
    cond[1] = (h_low >= 0 && w_high < width);
    cond[2] = (h_high < height && w_low >= 0);
    cond[3] = (h_high < height && w_high < width);

    // Load the four bilinear neighbors (zeroed where out of bounds, so the
    // weight-gradient accumulation below can run unconditionally).
    alignas(channel_vector_t) scalar_t v_array[4][channels_per_thread];
#pragma unroll
    for (int j = 0; j < 4; j++) {
#pragma unroll
        for (int i = 0; i < channels_per_thread; i++)
            v_array[j][i] = (scalar_t)0.;
        if (cond[j])
            deform_utils::vectorized_copy<scalar_t, channel_vector_t,
                                          channels_per_thread>(v_array[j],
                                                               value + idx[j]);
    }

    // Scatter the value gradient; the strategies differ only here.
#pragma unroll
    for (int j = 0; j < 4; j++) {
        if (!cond[j])
            continue;
        grad_value_t *addr = grad_value + idx[j];
#pragma unroll
        for (int i = 0; i < channels_per_thread; i++) {
            const accum_t<scalar_t> g_val =
                grad_output_opmath[i] * attention_weight * weights[j];
            if constexpr (use_half2_atomics) {
                // half2 atomic when the pair is aligned; otherwise per-element.
                if ((channels_per_thread % 2 == 0) && (i + 1 < channels_per_thread) &&
                    ((reinterpret_cast<uintptr_t>(addr + i) & 3) == 0)) {
                    const accum_t<scalar_t> g_val2 =
                        grad_output_opmath[i + 1] * attention_weight * weights[j];
                    deform_utils::atomicAdd_half2(addr + i, (scalar_t)g_val,
                                                  (scalar_t)g_val2);
                    i++;
                } else {
                    deform_utils::atomicAdd_scalar(addr + i, (scalar_t)g_val);
                }
            } else if constexpr (native_16bit) {
                // One native atomic per element: atomicAdd has no 16-bit overload.
                deform_utils::atomicAdd_scalar(addr + i, (scalar_t)g_val);
            } else {
                atomicAdd(addr + i, (grad_value_t)g_val);
            }
        }
    }

    accum_t<scalar_t> grad_h_weight[channels_per_thread];
    accum_t<scalar_t> grad_w_weight[channels_per_thread];
#pragma unroll
    for (int i = 0; i < channels_per_thread; i++) {
        grad_h_weight[i] = 0;
        grad_w_weight[i] = 0;
    }
#pragma unroll
    for (int i = 0; i < channels_per_thread; i++) {
        grad_h_weight[i] -= hw * (accum_t<scalar_t>)v_array[0][i];
        grad_w_weight[i] -= hh * (accum_t<scalar_t>)v_array[0][i];
        grad_h_weight[i] -= lw * (accum_t<scalar_t>)v_array[1][i];
        grad_w_weight[i] += hh * (accum_t<scalar_t>)v_array[1][i];
        grad_h_weight[i] += hw * (accum_t<scalar_t>)v_array[2][i];
        grad_w_weight[i] -= lh * (accum_t<scalar_t>)v_array[2][i];
        grad_h_weight[i] += lw * (accum_t<scalar_t>)v_array[3][i];
        grad_w_weight[i] += lh * (accum_t<scalar_t>)v_array[3][i];
    }

    accum_t<scalar_t> grad_location_x = 0;
    accum_t<scalar_t> grad_location_y = 0;
#pragma unroll
    for (int i = 0; i < channels_per_thread; ++i) {
        grad_location_x += grad_w_weight[i] * grad_output_opmath[i];
        grad_location_y += grad_h_weight[i] * grad_output_opmath[i];
    }
    grad_location_x *= (location_scale_w * attention_weight);
    grad_location_y *= (location_scale_h * attention_weight);
    *grad_sampling_parameters = grad_location_x;
    *(grad_sampling_parameters + 1) = grad_location_y;

    accum_t<scalar_t> grad_attention_weight = 0;
#pragma unroll
    for (int i = 0; i < channels_per_thread; i++) {
        const accum_t<scalar_t> current_val = w1 * (accum_t<scalar_t>)v_array[0][i] +
                                              w2 * (accum_t<scalar_t>)v_array[1][i] +
                                              w3 * (accum_t<scalar_t>)v_array[2][i] +
                                              w4 * (accum_t<scalar_t>)v_array[3][i];
        grad_attention_weight += current_val * grad_output_opmath[i];
    }
    *(grad_sampling_parameters + 2) = grad_attention_weight;
}

} // namespace deform_interpolate
