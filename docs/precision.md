# Numerical precision

Accumulation widths and operator dtype rules.

## Atomic accumulation and the SM version

The SM (streaming multiprocessor) version is what PyTorch reports as the CUDA compute capability:
SM 8.0 for A100, SM 9.0 for H100.

The backward kernels scatter gradients into shared feature-map positions with atomic adds.
Before SM 9.0 there is no native bfloat16 atomic add,
and the emulating compare-and-swap loop biases the result under contention.
So below SM 9.0, half-precision training of multi-scale and single-scale attention
uses the Triton float32 accumulator (`fp32acc`).
See [Backend selection](../README.md#backend-selection).

## Float32 accumulation per operator

- **Multi-scale attention, CUDA.**
  The host wrapper casts the inputs and upstream gradient to float32,
  the kernel accumulates with native float32 atomics,
  and the gradients are cast back.
- **Single-scale sampling, Triton.**
  With `fp32acc`, the input gradient accumulates into a float32 buffer and is cast back.
  Without it, the final cast is a no-op.
- **Deformable convolution, Triton.**
  Every accumulator is always float32:
  the forward accumulator, the three per-point backward accumulators and all four gradient buffers.
  Half precision would multiply the forward and offset-gradient error,
  and below SM 9.0 every gradient would hit the biased loop;
  the extra workspace is small.

## Native 16-bit atomics in CUDA

CUDA configurations that accumulate in 16 bits
compile their atomic add only from SM 7.0 (float16) and SM 8.0 (bfloat16).
The overloads are always declared so nvcc's host pass finds them.
Below that architecture the body is empty and the value gradient is zeros,
so the bridge refuses the launch and points to the float32 default CUDA backend.

## Gradient bias diagnostic

Reduced-precision error should be rounding noise, not an offset along the float64 gradient.
Launches reorder, so atomic adds differ per launch and the bias is averaged over several runs.
A non-zero cosine between the bias and the float64 gradient
means the error grows linearly in training, like learning-rate drift.
Noise grows with the square root of the step count.

- A backend is training-safe below an absolute cosine of 0.05,
  which allows jitter but not a systematic offset.
- The diagnostic exposes the SM 8.0 bfloat16 atomic discontinuity,
  the bias of the emulated 16-bit atomic add.
  It scales with the workload:
  1024 queries with 4 points is the smallest shape that shows it on the value gradient.
- It runs the default CUDA backend on SM 8.0 and SM 9.0.
  Below SM 9.0, train half precision with the query-tiled Triton configuration and `fp32acc`.

## Half-precision sampling locations

Every backend maps half-precision locations to pixel coordinates in float32.
The location gradient jumps at a cell boundary,
so a coordinate computed at the operand width
can move a sample into the neighboring cell,
which changes its gradient by the full jump.
Tests use the CUDA kernel, which widens first, as the reference,
bounded by the gradient's storage resolution
(bfloat16 keeps about two significant digits at the tested magnitudes).

## Mixed operand dtypes

- **Multi-scale attention.**
  `value` and `attention_weights` must share a dtype:
  the compiled kernels template on the value dtype and read the weights at that width,
  so a mismatch misreads memory.
  `sampling_locations` may differ.
  Under autocast, float32 reference points make the locations float32 while values and weights are half,
  as in Deformable DETR and Mask2Former.
  Locations only have to be floating point (the Triton kernels type their loads per pointer);
  a backend that cannot take them says so.
- **Deformable convolution.**
  Autocast leaves offsets in float32 while the value and weight are half,
  so the offset dtype only has to be floating point.

## Float64

- Backend selection sends float64 calls to eager on every operator.
- Triton has no float64 type and its loop accumulators are float32,
  so a float64 load fails to compile instead of running slowly.
  The multi-scale Triton bridge casts float64 to float32 and back for gradcheck.
  The single-scale and convolution Triton backends do not support float64.
- Triton gradchecks run at float32 with relaxed tolerances and `nondet_tol`,
  because `tl.atomic_add` is order-dependent.
- The eager single-scale backend builds its coordinate constants at the input dtype.
  Float64 gradcheck perturbs by `eps=1e-6`, below the float32 quantization floor,
  so a fixed float32 cast would turn the estimate into noise.
- The compiled CUDA kernels have a float64 branch unless `DEFORMOPS_NO_FP64` is defined.
  The JIT build does not define it,
  so locally compiled libraries (`DEFORMOPS_COMPILE=always`) run float64;
  only the ahead-of-time artifacts that `setup.py` assembles drop it.
  The macro must reach both the C++ and the CUDA translation unit,
  or they disagree on whether the branch exists.

## TF32

- The Triton convolution kernel never uses TF32 in its contraction, with no user option.
  Its many small dot products would round far more often than one fused matrix multiply,
  for a modest speedup.
  Half-precision operands are the better trade, since the kernel accumulates in float32.
- The test configuration's "high" float32 matmul precision puts cuBLAS into TF32 (about ten mantissa bits).
  torchvision contracts through cuBLAS while deformops runs full float32.
  Parity tests disable TF32 and use a float64 reference to show which result is closer.
  They also disable cuDNN, which adds another difference
  and whose runtime sublibrary fails to load in some environments.

## Precision in tests

- The Triton single-scale kernel accumulates in float32 and the reference at the input dtype.
  So half-precision tests compare against a single-precision reference on the same operands.
  Otherwise a lost wide accumulator would show only as slow training drift.
- The CPU eager backend accumulates at the input dtype, beyond the float32-accumulator tolerances,
  so half-precision parity runs only on CUDA.
