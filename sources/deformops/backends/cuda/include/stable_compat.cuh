// Stable-ABI replacements for TORCH_CHECK and CHECK_*: throwing
// std::runtime_error keeps the .so from importing c10's torchCheckFail, which
// would bind it to one PyTorch C++ ABI. Device code has no C++ ABI imports.

#pragma once

#include <cuda_runtime.h>

#include <torch/csrc/inductor/aoti_torch/c/shim.h>
#include <torch/csrc/inductor/aoti_torch/generated/c_shim_aten.h>
#include <torch/csrc/stable/tensor.h>

#include <sstream>
#include <stdexcept>
#include <string>

#include "utils.cuh"

#undef TORCH_CHECK
#undef CHECK_DIVISIBLE

// Fold-expression helper: concatenates all args into a single string.
namespace stable_compat {
template <typename... Args>
inline std::string concat(Args &&...args) {
    std::ostringstream oss;
    (oss << ... << args);
    return oss.str();
}
} // namespace stable_compat

// do/while(0) makes each expansion one statement; the variadic message is kept.
#define TORCH_CHECK(cond, ...)                                                         \
    do {                                                                               \
        if (!(cond)) {                                                                 \
            throw std::runtime_error(stable_compat::concat(__VA_ARGS__));              \
        }                                                                              \
    } while (0)

#define CHECK_DIVISIBLE(x, y)                                                          \
    do {                                                                               \
        if ((x) % (y) != 0) {                                                          \
            throw std::runtime_error(stable_compat::concat(                            \
                #x " (", (x), ") must be divisible by " #y " (", (y), ")"));           \
        }                                                                              \
    } while (0)

namespace stable_compat {

// Call `fn` (`[&]<typename scalar_t>() { ... }`) with the type `stype` names and
// return whether it is supported. This is where the supported dtypes are listed.
// `Double` (needed by gradcheck) is compiled unless `DEFORMOPS_NO_FP64`, which only
// the ahead-of-time `setup.py` builds define.
template <typename Fn>
bool dispatch_scalar_type(torch::headeronly::ScalarType stype, Fn &&fn) {
    using torch::headeronly::ScalarType;
    switch (stype) {
    case ScalarType::Float:
        fn.template operator()<float>();
        return true;
    case ScalarType::Half:
        fn.template operator()<c10::Half>();
        return true;
    case ScalarType::BFloat16:
        fn.template operator()<c10::BFloat16>();
        return true;
#ifndef DEFORMOPS_NO_FP64
    case ScalarType::Double:
        fn.template operator()<double>();
        return true;
#endif
    default:
        return false;
    }
}

// Message for a dtype `dispatch_scalar_type` rejected; for `Double` it names the
// build that restores double precision.
inline std::string unsupported_dtype_message(const char *op,
                                             torch::headeronly::ScalarType stype) {
    if (stype == torch::headeronly::ScalarType::Double)
        return concat(op, ": this library is built without double precision; set "
                          "DEFORMOPS_COMPILE=always to compile one that includes "
                          "it (needs a CUDA toolkit and deformops[jit])");
    return concat(op, ": unsupported dtype");
}

// The caller's current CUDA stream, through the stable C shim.
inline cudaStream_t current_stream_for(const torch::stable::Tensor &t) {
    int32_t device = -1;
    void *stream = nullptr;
    // Off a CUDA device the shim leaves the out-parameters indeterminate.
    if (aoti_torch_get_device_index(t.get(), &device) != AOTI_TORCH_SUCCESS ||
        aoti_torch_get_current_cuda_stream(device, &stream) != AOTI_TORCH_SUCCESS) {
        throw std::runtime_error("cannot resolve the current CUDA stream; the tensor "
                                 "is not on a CUDA device");
    }
    return reinterpret_cast<cudaStream_t>(stream);
}

} // namespace stable_compat
