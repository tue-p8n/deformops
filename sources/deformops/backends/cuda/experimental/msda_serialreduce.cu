// Multi-scale deformable attention, split layout:
// fp32 accumulation, serial softmax reduction.
// The kernels are shared in include/msda_boxed.cuh; this unit picks the backward
// traits and operator names. See docs/kernels.md.

#include "msda_boxed.cuh"

namespace {
namespace boxed = msda_split::boxed;

constexpr bool kNativeGradValue = false;
constexpr bool kSoftmaxUsesWarp = false;
constexpr bool kAllowHalf2Atomics = true;

template <int64_t apply_softmax>
constexpr auto kBackward = &boxed::backward<kNativeGradValue, kSoftmaxUsesWarp,
                                            kAllowHalf2Atomics, apply_softmax>;
} // anonymous namespace

STABLE_TORCH_LIBRARY_IMPL(deformops_cuda, CUDA, m) {
    m.impl("msda_serialreduce_forward", &boxed::forward<0>);
    m.impl("msda_serialreduce_backward", kBackward<0>);
    m.impl("msda_serialreduce_softmax_forward", &boxed::forward<1>);
    m.impl("msda_serialreduce_softmax_backward", kBackward<1>);
}
