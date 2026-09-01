// Argument structs and launch checks for the split-layout MSDA kernels, which take
// locations and attention weights as separate tensors.
// Include preamble: ``stable_compat.cuh`` redefines ``TORCH_CHECK``, so the other
// kernel headers include CUDA and torch only through this file.
#pragma once

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/csrc/inductor/aoti_torch/c/shim.h>
#include <torch/csrc/inductor/aoti_torch/generated/c_shim_aten.h>
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/tensor.h>

#include <algorithm>
#include <stdexcept>
#include <type_traits>

#include "interpolate.cuh"
#include "stable_compat.cuh"
#include "utils.cuh"

namespace msda_split {

// Namespaced to keep the aliases out of every including translation unit.
using Tensor = torch::stable::Tensor;
using ScalarType = torch::headeronly::ScalarType;

template <typename scalar_t>
struct ForwardArgs {
    const scalar_t *value;
    const int64_t *spatial_shapes;
    const int64_t *level_start_index;
    // Accumulation width: a 16-bit pixel coordinate can fall in the next cell.
    const accum_t<scalar_t> *location;
    const scalar_t *attention_weight;
    scalar_t *output;
    int64_t N;
    int64_t G;
    int64_t D;
    int64_t Q;
    int64_t queries_per_block;
};

template <typename scalar_t, typename grad_value_t>
struct BackwardArgs {
    const scalar_t *value;
    const int64_t *spatial_shapes;
    const int64_t *level_start_index;
    const accum_t<scalar_t> *location;
    const scalar_t *attention_weight;
    const scalar_t *grad_output;
    grad_value_t *grad_value;
    accum_t<scalar_t> *grad_location;
    accum_t<scalar_t> *grad_attention_weight;
    int64_t N;
    int64_t G;
    int64_t D;
    int64_t Q;
    int64_t queries_per_block;
};

// `L` is a template parameter so level loops unroll; the
// `dispatch_constant<1, 2, 3, 4, 5>` packs list the instantiated depths.
inline constexpr const char *kLevelBoundMessage =
    "multi-scale deformable attention supports 1 to 5 levels; `L` is a template "
    "parameter so the level loop unrolls, and an unsupported depth has no "
    "instantiation to dispatch to. Got L=";

// CUDA's ceiling on threads per block. One query can exceed the tuning preference,
// so both launchers check this to report a clear error instead of the driver's.
inline constexpr int64_t kMaxThreadsPerBlock = 1024;

// Threads the driver launches after rounding down to whole queries.
__host__ inline int64_t block_thread_count(const dim3 &block) {
    return static_cast<int64_t>(block.x) * block.y * block.z;
}

// A rejected launch leaves outputs at zero, so read the error here, not later.
inline void check_launch(const char *name) {
    const cudaError_t launch_error = cudaGetLastError();
    TORCH_CHECK(launch_error == cudaSuccess, name,
                " launch failed: ", cudaGetErrorString(launch_error));
}

} // namespace msda_split
