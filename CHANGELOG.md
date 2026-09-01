# Changelog

## v3.0.1

_September 1, 2026_

- Added the MSDA Triton kernels.
- Added the float32 accumulator (`fp32acc`).
- Moved to the tue-p8n organization.
- Accompanies the TMLR paper
  [Efficient Multi-Scale Deformable Attention on GPUs](https://openreview.net/forum?id=Q4jZ7zKNKx).

## v2.0.0

_March 22, 2024_

- Added the `DeformAttn2d` and `DeformConv2d` Triton kernels.
- Added the softmax method to the MSDA kernel.

## v1.0.0 (internal release)

_October 2, 2023_

- MSDA CUDA kernel with vectorized loads.
