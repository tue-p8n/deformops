# Operator conventions

Layout, ordering, boundary and validation rules shared by every implementation of an operator.

## Reference, eager and device implementations

- `backends/reference` holds one pure PyTorch reference per operator,
  stating the math in its simplest form.
  Every other backend is tested against it.
- Nothing registers a reference and `Backend` has no member for one,
  so only an explicit import reaches it.
  Only `eager`, `triton` and `cuda` are dispatchable, and `eager` may diverge to optimize.
- The single-scale sampling and convolution references, and the eager convolution,
  trace fully under `torch.compile`:
  no `.item()`, no `.any()`, no branch on a tensor value.
- The multi-scale reference uses `grid_sample` and is slow by design.
  Its per-level loop reads `.item()` for slice bounds,
  so `torch.compile` breaks the graph once per level.
  The eager backend (`backends/eager/msda_pytorch.py`) is traceable.

## Kernel-point order

Mixing the two grid conventions gives plausible output from the wrong offsets.

| | Single-scale sampling | Classic convolution |
|---|---|---|
| Grid walk | width-outer, height-inner | row-major, `k = kh * kW + kw` |
| Offset pair | `(dx, dy)`: `offset[..., 0]` is x | `(dy, dx)`: the first displaces along the row axis |
| Packing | `(dx, dy, weight)` per point | separate `offset` and `mask` |
| Follows | the in-repo reference | torchvision |

## Multi-scale deformable attention

- Pixel coordinates match `grid_sample(v, 2 * sampling_locations - 1, align_corners=False)`:
  `pix = sampling_locations * size - 0.5`.
  The eager backend zero-pads out-of-bounds samples with boolean masks
  and clamps indices to keep the gather in range.
- `level_start_index` is the exclusive prefix sum of `height * width` over the levels.
  Each level must start where the previous one ended,
  and the levels together must cover the value tensor.
  Wrong calls most often break this rule.
- Layer reference points are normalized to `[0, 1]`,
  one per query and level, shaped `(B, Q, L, 2)` or `(B, Q, L, 4)`.
  - For `(x, y)` points, predicted offsets are in pixels of each level
    and are divided by the level's `(width, height)`
    (`spatial_shapes` with its columns swapped) before they are added.
  - For `(cx, cy, w, h)` boxes:
    `location = (cx, cy) + offset / num_points * (w, h) / 2`.

### Batch chunking with `chunk_step`

- Only the CUDA backend reads the chunk size.
  It chunks at the largest divisor of the batch that is at most `chunk_step`
  (128 by default).
  The chunk size:
  - keeps flat indices below `2**31`,
    since spatial indexing and bilinear weight math use 32-bit integers;
  - lets `find_factors` tile the grid
    without oversized grid dimensions or exceeding hardware block limits;
  - bounds the scratch space and shared memory of the backward.
- The layer and the model integrations pass `chunk_step` only to an explicit backend.
  The public operator behind `backend=None` has no chunk size,
  so a selected CUDA kernel uses its default.

## Single-scale deformable sampling

- `offset_mask` packs `(dx, dy, weight)` per point, grouped as `(groups, K, 3)`,
  in its first `groups * K * 3` channels, shaped `[B, Q, G, K, 3]` after slicing.
  Later channels are ignored, so the tensor may be wider than the operator uses.
  `DeformAttn2d` rounds its projection width up to a multiple of 8
  (see the stride hazard in [kernels](kernels.md)).
- Offsets are in input pixels.
  `spread` scales each displaced grid point about the kernel center.
  It enters the centering term of the base position as well as the offset,
  so it also rescales coordinate gradients.
- Kernel points are numbered width-outer, height-inner.
  `drop_center` drops the point at `kernel // 2` on each axis, not `(kernel - 1) // 2`;
  for an even kernel that is one cell past the midpoint.
  The point index advances for every non-center grid point,
  following the loop position compacted for the dropped center,
  not the number of points in bounds.
  A kernel that breaks any of these rules
  reads the wrong offset triple for every point after the center,
  with plausible, well-formed output.
  The grid-order function is public so tests can check it directly.
- The kernel builds normalized locations from the convolution geometry;
  multi-scale attention receives them.

### Boundaries

- A location's four bilinear corners span `[floor(loc), floor(loc) + 1]`.
  A sample is taken when the location lies in the open interval `(-1, extent)`;
  outside it, no corner is inside.
- Corners outside `[0, H) x [0, W)` contribute zero instead of being clamped to the edge.
  Gradcheck sees this asymmetry.

### Softmax

- The softmax method normalizes the modulation over the `K` points of each group.
  The Triton path applies it before the kernel runs:
  the reference normalizes and then accumulates linearly,
  so the result is the same and autograd handles the Jacobian.
- For why the reference skips `valid_loc.any()` and builds constants in the input dtype,
  see [export](export.md) and [precision](precision.md).

## Classic deformable convolution

The operator torchvision calls `deform_conv2d` computes

```text
y[b, c_o, p] = sum_{c_i} sum_k w[c_o, c_i, k] * m_k * x(b, c_i, p + p_k + Δp_k) + bias[c_o]
```

- v1 is v2 with the modulation `m_k` fixed at one,
  so one implementation with an optional mask covers both:
  `mask=None` equals `mask=1`.
- The contraction against a `(C_out, C_in / groups, kH, kW)` weight
  makes this a convolution, unlike the sampling operator.
- Layout follows torchvision.
  Samples outside the input contribute zero; corners are not clamped.
- `groups` partition input and output channels together.
  `offset_groups` partition only the input channels, independently:
  channel `c` uses the offsets of group `c // (C_in // offset_groups)`.

### Parity with torchvision

- The forward and every gradient are compared with torchvision across all arguments.
  The in-repo reference is checked against torchvision too,
  and replaces it where torchvision is absent.
- The backward specializes on which gradients autograd requests.
  Those gradients share intermediates.
  The interpolated value feeds the weight and mask gradients,
  and the gradient at each sample feeds the input, offset and mask gradients.
  Skipping a block based on one output that reads it, instead of all of them,
  gives wrong numbers without an error,
  so the tests compare every gradient subset with torchvision.
- The bias gradient is a host-side reduction, so a bias-only backward launches no kernel.
  Skipping the weight gradient matters for fine-tuning.

## Linear and softmax method

- Every backend adapter and the context builder, for both attention operators,
  pass `method` and `softmax` to `resolve_method`, so the default is defined once.
  An explicit `method` wins; otherwise the boolean `softmax` decides,
  and its default makes `LINEAR` the functional default.
- `LINEAR` is the default because the operator takes attention weights as given.
  Deformable DETR and mmcv normalize before the call.
  `MultiScaleDeformAttn2d` holds raw logits and requests `SOFTMAX`,
  which for multi-scale attention normalizes jointly over the `L * K` points of each head.

## Training detection

PyTorch builds a backward graph only when grad mode is enabled
and at least one differentiable input requires grad.
`is_training_from` checks both;
ORing `torch.is_grad_enabled()` with the per-tensor checks
would classify inference as training, even with `requires_grad=True` inputs.

## Operand validation

- Constructor-time checks, such as `check_msda_params`,
  validate integers when a module is built and always run.
  Call-time checks (`check_msda_call`, `check_attn_call`, `check_conv_call`)
  validate the tensors passed to a functional entry point.
- The kernels index the value tensor from `spatial_shapes` and `level_start_index`
  without re-deriving either.
  An inconsistent set is an out-of-bounds device read, not an exception:
  corrupted output, an unrelated CUDA fault later, or no visible effect.
- Structural checks read only rank, size, dtype and device,
  so they work under `torch.compile`, fake tensors and CUDA-graph capture,
  and run on every call.
- Value checks read `spatial_shapes` and `level_start_index` back to the host,
  which synchronizes and breaks graph capture.
  They run only when `DEFORMOPS_VALIDATE` is set,
  once when connecting a new caller, not in training.
  They skip traced calls and fake tensors, including wrapped ones.

### Operator-specific checks

- Device kernels compute a per-image pointer base as `batch_index * per_image_stride`
  (see [kernels](kernels.md)),
  so a per-image size check alone misses a large batch of small images.
  Convolution checks both input and output extents,
  since either can be larger depending on channels and groups.
- For single-scale sampling and convolution,
  the reference derives the output grid from the input extents
  and the device kernel reads it from `offset_mask` or `offset`.
  A disagreeing operand raises in one
  and returns a differently shaped tensor from the other, without an error.
- `conv_output_size` is the single definition of the convolution output size,
  shared by the reference, the device kernel and validation.
- Convolution requires `out_channels` divisible by `groups`.
  The kernel splits output channels evenly into an uninitialized buffer,
  so the last `out_channels % groups` channels would hold uninitialized memory.
- The convolution kernel reads `bias` at stride one per output channel
  without a length check, so a short bias reads past its end on the device.

## torchvision drop-ins

`deformops.integrations.torchvision` keeps the argument names, order and NCHW layout of `torchvision.ops`
(see the [README](../README.md#torchvision-drop-in)).
Numerics need not match bit for bit,
so a model trained with one may need retraining to reproduce with the other.
