# DeformOps

`deformops` is a PyTorch library of three deformable sampling operators:
single-scale deformable sampling,
classic deformable convolution,
and multi-scale deformable attention (MSDA).
Its CUDA and Triton kernels are tuned for the encoder-scale workload
that dominates Mask2Former-style training:
5k to 22k queries per call in the pixel decoder.
[Efficient Multi-Scale Deformable Attention on GPUs](https://openreview.net/forum?id=Q4jZ7zKNKx)
(TMLR 2026)
describes the kernel design and its measurements.

- **Zero-buffer kernels.**
  The multi-scale attention backward writes into the value gradient
  with no intermediate sampling buffer.
  At encoder scale, the paper measures forward and backward peak memory
  71% below the reference for the CUDA kernel
  and 88% below for the native-bfloat16 Triton kernel.
- **Triton kernels** for all three operators,
  with multi-scale attention measured in the paper on Ampere and Hopper GPUs.
- **Automatic backend selection** by device, GPU architecture and training precision;
  see [Backend selection](#backend-selection).
- **Half precision** (bfloat16 and float16) throughout;
  bfloat16 is the recommended training dtype.
- **Parallel-safe kernel builds.**
  When several processes compile the same CUDA kernel at once,
  such as the ranks of a multi-GPU job,
  a file lock runs one build at a time,
  and a build killed mid-compile leaves no stale lock behind.

## Installation

```bash
pip install deformops
```

`deformops` supports Python 3.12 and 3.13
and needs PyTorch 2.11 or newer (2.12 or newer for CUDA 13.x).
A bare install needs only `torch`
and gives the eager backends on any platform.
The GPU kernels need Linux x86-64 and a CUDA-capable GPU.

The published package is pure Python.
With `backend=None`, a GPU call runs a Triton kernel compiled at runtime,
so it needs no CUDA toolkit.
The CUDA backend needs a compiled kernel library;
see [Kernel loading](#kernel-loading).

Installing from the source distribution with `--no-build-isolation`
compiles the non-experimental CUDA kernels into the package,
against the installed CUDA build of torch that later loads them.
This needs setuptools, a CUDA toolkit with torch's CUDA major version,
and a visible GPU or `TORCH_CUDA_ARCH_LIST`:

```bash
pip install --no-binary deformops --no-build-isolation deformops
DEFORMOPS_EXTENSIONS=none pip install --no-binary deformops deformops  # without compiled kernels
```

Without a CUDA toolkit, the install skips the compile without a message.
It skips the kernels and prints why in four other cases:
the build is isolated,
the toolkit's CUDA major version differs from torch's,
no GPU is visible and `TORCH_CUDA_ARCH_LIST` is unset,
or the compile fails.
The CUDA backend repeats that reason when it finds no library.
An explicit `DEFORMOPS_EXTENSIONS` or `FORCE_CUDA=1` fails the install instead.
pip and uv reuse a wheel built from the same source distribution,
so after changing these settings install with `--no-cache-dir` (pip) or `--no-cache` (uv).
A host compiler newer than PyTorch's range for the CUDA toolkit fails torch's version check;
`TORCH_DONT_CHECK_COMPILER_ABI=1` skips it.

| Install                      | What it adds                                                                  |
| :--------------------------- | :---------------------------------------------------------------------------- |
| `deformops[cpu]`, `deformops[cu126]`, `deformops[cu128]`, `deformops[cu129]`, `deformops[cu130]`, `deformops[cu132]` | In a uv-synced checkout, routes `torch` to the CPU or matching CUDA wheel index (`cu130` and `cu132` need torch 2.12 or newer). |
| `deformops[triton]`          | Installs the Triton compiler for a CPU torch.                                 |
| `deformops[jit]`             | Installs `ninja`, so torch can compile a kernel locally.                      |
| `deformops[mask2former]`     | Installs `transformers`, and `torchmatch` on Python 3.13, for the pre-made Mask2Former graph. |
| `deformops[deformable-detr]` | Installs `transformers` and `timm` for the pre-made Deformable DETR graph.    |
| `deformops[mmcv]`            | Installs `mmcv-lite` for the mmcv integration.                                |

The `cpu` and `cuXXX` extras are mutually exclusive.
Their index mappings live in `[tool.uv.sources]`, not in published metadata,
so under pip, choose the build through the PyTorch index:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128  # CUDA 12.8, or /whl/cpu
pip install deformops
uv pip install deformops --torch-backend=cu128                        # the same, with uv
```

A CUDA torch already depends on Triton;
the Triton backends detect it at runtime either way.
Compiling needs a CUDA toolkit besides `ninja`, and pip does not provide one.
If either is missing, a compile that `DEFORMOPS_COMPILE` requests raises `KernelBuildError` naming what is missing.
Extras do not change operator speed.

### Kernel loading

The CUDA backend takes its compiled kernel library from the first source that has one:
the install itself;
an earlier compile on this machine for the same sources and environment;
a prebuilt index (`DEFORMOPS_DOWNLOAD`, off by default);
or a local compile (`DEFORMOPS_COMPILE`, off by default).
An enabled download that fails raises `KernelDownloadError`,
which names the variable that skips it.
With both disabled, the CUDA backend is unavailable:
`backend=None` moves on with a warning,
and `backend=Backend.CUDA` raises `KernelNotProvidedError`.

Ahead-of-time libraries (prebuilt or from a source-distribution install)
have no float64 kernels,
so a float64 call on the CUDA backend raises
unless the library is a just-in-time build (`DEFORMOPS_COMPILE=always`).

The library loads on the first call,
or earlier through `deformops.registry.msda.load`.
Run it before loading an exported program that calls the CUDA operators
(`torch.ops.deformops_cuda.*`):

```python
from deformops.registry import Backend, msda

msda.load(Backend.CUDA)  # loads the CUDA kernel library
```

## Multi-scale deformable attention

The layer is `deformops.nn.MultiScaleDeformAttn2d`
and the functional form `deformops.registry.msda`.
Shapes below use `B` batch, `G` heads, `D` channels per head,
`Q` queries, `L` levels and `K` points per level.

```python
import torch
from deformops.nn import MultiScaleDeformAttn2d

layer = MultiScaleDeformAttn2d(
    dim=256,
    num_heads=8,
    num_levels=4,
    num_points=4,
).cuda()

query = torch.randn(1, 300, 256).cuda()
reference = torch.rand(1, 300, 4, 2).cuda()  # normalized [0, 1]
value = torch.randn(1, 5440, 256).cuda()  # flattened feature maps
shapes = torch.tensor([[64, 64], [32, 32], [16, 16], [8, 8]]).cuda()
starts = torch.tensor([0, 4096, 5120, 5376]).cuda()  # prefix sum of H * W

out = layer(query, reference, value, shapes, starts)
```

The functional form takes the sampled operands without the projections;
its five tensor operands are positional-only.

By default, every backend uses the attention weights as given.
`method="softmax"` normalizes raw logits jointly over levels and points
inside the kernel.
The layer requests softmax itself, so this default does not affect it.

```python
from deformops.registry import msda

value = torch.randn(1, 5440, 8, 32).cuda()  # (B, sum H*W, G, D)
locations = torch.rand(1, 300, 8, 4, 4, 2).cuda()  # (B, Q, G, L, K, 2)
weights = torch.rand(1, 300, 8, 4 * 4).cuda()  # (B, Q, G, L * K), raw logits

out = msda(value, shapes, starts, locations, weights, method="softmax")
```

## Single-scale deformable sampling

The layers are `deformops.nn.DeformAttn2d` and `deformops.nn.DeformSample2d`;
the functional form is `deformops.registry.attn`.
The operator samples and aggregates without the spatial weight of deformable convolution.
`DeformAttn2d` takes a separate query and value,
predicts offsets and aggregation weights from the query,
and projects the value and the output.
`DeformSample2d` derives the query from its input with a depthwise convolution
and runs `DeformAttn2d` on it.

The device kernel is Triton,
so it needs no CUDA toolkit, per-CUDA build or downloaded library.
Off CUDA and for float64, `backend=None` runs the eager implementation.

```python
from deformops.nn import DeformSample2d

layer = DeformSample2d(dim=64, kernel_size=3, groups=4).cuda()
x = torch.randn(1, 64, 32, 32).cuda()  # NCHW, as nn.Conv2d takes
y = layer(x)  # -> (1, 64, 32, 32)
```

## Deformable convolution

The layer is `deformops.nn.DeformConv2d`;
the functional form is `deformops.registry.conv`.
The operator computes classic deformable convolution as `torchvision.ops.deform_conv2d` does:
a `(C_out, C_in / groups, kH, kW)` weight contracted against the samples,
v2 with a mask and v1 without.

```python
from deformops.nn import DeformConv2d

layer = DeformConv2d(64, 64, kernel_size=3, padding=1).cuda()
offset = torch.randn(1, 2 * 9, 32, 32).cuda()  # (dy, dx) per kernel point
mask = torch.rand(1, 9, 32, 32).cuda()  # omit for v1
y = layer(x, offset, mask)  # -> (1, 64, 32, 32)
```

One kernel gathers and contracts,
so the im2col column buffer `(B, H_out * W_out, C_in, kH * kW)` never exists.
Accumulation is float32 for every operand dtype.
TensorFloat-32 (TF32) is off inside the contraction,
since its rounding error would reach the offset gradients.
`conv` runs the same kernel over channels-last operands with `deformops` argument names.

## torchvision drop-in

`deformops.integrations.torchvision` matches `torchvision.ops.deform_conv2d`'s
argument order and NCHW layout.

```python
from deformops.integrations.torchvision import deform_conv2d

y = deform_conv2d(x, offset, layer.weight, layer.bias, padding=layer.padding, mask=mask)
```

## Backend selection

Every operator dispatches through one of three backends,
named by `deformops.registry.Backend`:

```python
from deformops.registry import Backend

assert set(Backend) == {Backend.CUDA, Backend.TRITON, Backend.EAGER}
```

Pass one to a layer, or leave `backend=None` for the selection policy:

```python
from deformops.nn import MultiScaleDeformAttn2d

layer = MultiScaleDeformAttn2d(
    dim=256,
    num_heads=8,
    num_levels=4,
    backend=Backend.CUDA,
)
```

| Backend          | What it is                                                                 |
| :--------------- | :------------------------------------------------------------------------- |
| `Backend.CUDA`   | The compiled CUDA kernel, for multi-scale attention.                       |
| `Backend.TRITON` | The Triton kernels, compiled at runtime.                                   |
| `Backend.EAGER`  | The eager PyTorch backend; device-agnostic and traceable.                  |

Only the CUDA backend has these limits:
a per-head channel count divisible by 8,
1 to 5 levels,
4 or 8 points per level,
and at most 1024 threads per block.
A head layout uses `G * D / 8` threads in the forward
and, when training, `G * D` in the backward, or `G * D / 2` when `D >= 64`.

With `backend=None`, CPU tensors run eager.
CUDA tensors run a GPU kernel or raise:
MSDA tries Triton, then CUDA;
single-scale sampling and deformable convolution use Triton.
Eager runs on a CUDA tensor only with `backend=Backend.EAGER` or for float64.
A call that skips its first choice warns with `BackendFallbackWarning`,
naming the skipped backends and why;
`DEFORMOPS_FALLBACK_WARNINGS=0` silences it.
When no backend can run, `NoBackendError` lists every candidate and why it was rejected.

GPUs below SM 9.0 (streaming-multiprocessor version, the CUDA compute capability)
have no native bfloat16 atomic add,
the instruction that lets parallel GPU threads add into one gradient entry.
The kernels emulate it with a compare-and-swap retry loop,
which biases the value gradient under contention,
so half-precision training there runs Triton with the float32 accumulator.
MSDA and single-scale sampling share this rule,
and their other calls use native half-precision atomics.
The choice is memoized per call context,
so it follows a model to a new device or dtype.
An explicit `backend=Backend.TRITON` gets the float32 accumulator,
which is correct on every architecture;
`fp32acc=False` uses native atomics on SM 9.0 and later.

For MSDA, `select_backend` implements this policy:

```python
from deformops.registry.msda import select_backend
```

Each operator's `resolve` reports the backend a call would run
and why it skipped earlier candidates:

```python
import torch
from deformops.registry import msda

value = torch.randn(1, 80, 4, 8)
shapes = torch.tensor([[8, 8], [4, 4]])
starts = torch.tensor([0, 64])
locations = torch.rand(1, 10, 4, 2, 4, 2)
weights = torch.rand(1, 10, 4, 2 * 4)

print(msda.resolve(value, shapes, starts, locations, weights))
```

`deformops.registry.msda`, `.conv` and `.attn` hold their operator's backends
and are callable:

```python
out = msda(value, shapes, starts, locations, weights, method="softmax", backend=Backend.EAGER)
```

A `backend=None` call runs a public `torch.library` operator
(`torch.ops.deformops.msda`, `msda_softmax`, `attn`, `attn_softmax` or `conv`).
Its one kernel for every device applies the selection policy
and runs the chosen backend's private operator
under `torch.ops.deformops_triton` or `torch.ops.deformops_cuda`;
[operator registration and export](https://github.com/tue-p8n/deformops/blob/main/docs/export.md) explains why.
Calling a public operator directly selects the same way:

```python
out = torch.ops.deformops.msda_softmax(value, shapes, starts, locations, weights)
```

A `backend=None` call with a backend-specific keyword such as `fp32acc`
selects its backend in Python, since the public operators take none,
and a traced graph records the private operator.

The reference implementations are `msda.reference`, `attn.reference` and `conv.reference`
in the submodules of `deformops.backends.reference`.
`Backend` has no member for them, so dispatch never selects them;
the test suite checks every other backend against them.

## Pre-made models

Each model package loads the upstream graph and overrides its deformable attention;
Mask2Former also overrides the `einsum` mask predictor, which obstructs export,
and the scipy-backed Hungarian matcher.
Architecture, weights and post-processing are upstream.

| Model           | Full graph              | deformops override                                     | Extra               |
| :-------------- | :---------------------- | :----------------------------------------------------- | :------------------ |
| Mask2Former     | `models.Mask2Former`    | MSDA + mask predictor + torchmatch matcher             | `[mask2former]`     |
| Deformable DETR | `models.DeformableDETR` | Hugging Face MSDA core                                 | `[deformable-detr]` |

```bash
pip install "deformops[mask2former]"      # transformers + torchmatch
pip install "deformops[deformable-detr]"  # transformers + timm
```

Mask2Former needs Python 3.13 and torch 2.11 or newer, through `torchmatch`,
which publishes no Python 3.12 wheel.

The patched graphs compile and export on every backend;
see [Compile and export](#compile-and-export).

The overrides are importable without the full graph:

```python
from deformops.models.mask2former import (
    PatchedMask2FormerHungarianMatcher,
    PatchedMask2FormerLoss,
    PatchedMask2FormerMaskPredictor,
    PatchedMask2FormerPixelDecoderEncoderMultiscaleDeformableAttention,
)
from deformops.models.deformable_detr import PatchedMultiScaleDeformableAttention

# Or reuse the library integration directly:
from deformops.integrations.transformers import MultiScaleDeformableAttention
from deformops.integrations.mmcv import MultiScaleDeformableAttention as MMCVAttention
from deformops.integrations.transformers import patch as patch_transformers
```

The integration `patch(...)` helpers replace matching upstream modules in place
and raise `RuntimeError` when nothing matches,
usually because the upstream version renamed its modules.
`strict=False` accepts a model with nothing to replace.

A full Mask2Former loads from a Hugging Face checkpoint (network access)
and takes a batch of RGB images:

```python
import torch
from deformops.models.mask2former import Mask2Former

model = Mask2Former.from_pretrained("facebook/mask2former-swin-base-coco-instance").eval()
pixel_values = torch.randn(1, 3, 384, 384)  # (B, 3, H, W), normalized RGB

with torch.no_grad():
    outputs = model(pixel_values)  # class_queries_logits, masks_queries_logits
```

`deformops.models.mask2former.DEFAULT_CHECKPOINT` names the base COCO instance checkpoint;
the other Swin sizes (tiny, small, large) and the COCO panoptic and ADE20K semantic checkpoints load the same way.

Models built on mmcv 2.x's own `MultiScaleDeformableAttention`,
such as the mmdet 3.x Deformable DETR and DINO,
run on deformops through `deformops.integrations.mmcv.patch(model)`,
with the `deformops[mmcv]` extra.

## Compile and export

Every operator is a registered `torch.library` operator with autograd,
so models using them work with `torch.compile`, `torch.export` and AOTInductor.
The test suite runs each recipe below on both pre-made models,
on every backend and with `backend=None`.

`torch.compile(model, fullgraph=True)` compiles inference and the backward pass.
With `backend=None`, Dynamo records the public operator,
and the compiled graph holds the private operator selected for the traced call.
In training mode, both upstream models call `isfinite(...).all()` on their hidden states,
which breaks the graph in upstream code;
compile them for training without `fullgraph`.

`torch.export.export` keeps each public operator (`torch.ops.deformops.*`)
and each private operator an explicit backend selects
(`torch.ops.deformops_triton.*` and `torch.ops.deformops_cuda.*`)
as a single node.
The eager backend exports as standard operators.
`ExportedProgram.run_decompositions()` replaces each public operator
with the private operator selected on the host that runs the decomposition.
A saved program names its operators,
so `torch.export.load` needs deformops installed and `import deformops.registry` run first.
Mask2Former exports with a dynamic batch size.
Deformable DETR resizes its pixel mask with the batch in the channel position,
which fixes the batch size of its export on CUDA.

AOTInductor runs those decompositions before compiling,
so a package holds the kernel selected at build time,
with the Triton kernels compiled in.
A package built on the Triton or eager backend,
or with `backend=None` selecting Triton,
loads without deformops installed.
A package that calls the CUDA operators needs their kernel library loaded first
in the process that loads it:

```python
import torch
from deformops.models import Mask2Former
from deformops.registry import Backend, msda

model = Mask2Former.from_pretrained(backend=Backend.CUDA).cuda().eval()
image = torch.randn(2, 3, 384, 384, device="cuda")
batch = torch.export.Dim("batch", min=1, max=16)

with torch.no_grad():
    program = torch.export.export(model, (image,), dynamic_shapes={"pixel_values": {0: batch}})
torch._inductor.aoti_compile_and_package(program, package_path="mask2former.pt2")

# In the process that serves the package:
msda.load(Backend.CUDA)
served = torch._inductor.aoti_load_package("mask2former.pt2")
```

AOTInductor needs `openssl` on `PATH` to hash headers.

## Environment variables

| Variable                       | Effect                                                                                                                                                               |
| :----------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `DEFORMOPS_DOWNLOAD`           | `0` (default) skips the download. `1` downloads a prebuilt CUDA kernel library from `DEFORMOPS_OPS_INDEX_URL` when none is installed or compiled; failure raises. |
| `DEFORMOPS_COMPILE`            | `0` (default) never compiles. `1` compiles when no library is available. `always` uses a library compiled on this machine; `rebuild` recompiles it from scratch.       |
| `DEFORMOPS_FALLBACK_WARNINGS`  | `0` silences `BackendFallbackWarning`.                                                                                                                                |
| `DEFORMOPS_OPS_INDEX_URL`      | Base URL of the prebuilt operator index, for a private mirror.                                                                                                       |
| `DEFORMOPS_OPS_CACHE_DIR`      | The cache directory for downloaded libraries. Defaults to `~/.cache/deformops/ops`.                                                                                         |
| `DEFORMOPS_OPS_ALLOW_INSECURE` | Permits a plaintext `http://` index. Off by default, since the manifest holds the library checksums and plaintext defeats that check. |
| `DEFORMOPS_OPT_LEVEL`          | Optimization level, `0` to `3`, for a runtime JIT build. Unset, the build uses the level the extension declares.                                                     |
| `TORCH_EXTENSIONS_DIR`         | Where JIT builds are cached. Unset, torch's default build root, one directory per package and environment.                                          |
| `DEFORMOPS_VALIDATE`           | Enables operand checks that read the level table back to the host. Off by default because they synchronize.        |
| `DEFORMOPS_EXTENSIONS`         | Build-time only. `all` compiles the non-experimental CUDA kernels, `everything` adds the experimental ones, `none` compiles nothing, and a comma-separated list names kernels. See [Installation](#installation). |

## Examples

The [tests](https://github.com/tue-p8n/deformops/tree/main/tests/deformops)
hold runnable examples, one file per operator,
with the pre-made models under [tests/deformops/models](https://github.com/tue-p8n/deformops/tree/main/tests/deformops/models).

## Support

Report bugs and propose changes through GitHub issues and pull requests;
for questions about the method, see the [paper](https://openreview.net/forum?id=Q4jZ7zKNKx).

## Contributing

[CONTRIBUTING.md](https://github.com/tue-p8n/deformops/blob/main/CONTRIBUTING.md)
describes the layout, conventions, commands and CI.

## Citation

Cite the paper that introduced each operator you use.

### Multi-scale deformable attention

This covers `MultiScaleDeformAttn2d`, `registry.msda`,
and the Mask2Former, Deformable DETR and mmcv integrations.
Cite Deformable DETR for the operator and Efficient MSDA for the kernels.

```bibtex
@inproceedings{zhu2021deformable,
  title     = {Deformable {DETR}: Deformable Transformers for End-to-End Object Detection},
  author    = {Zhu, Xizhou and Su, Weijie and Lu, Lewei and Li, Bin and Wang, Xiaogang and Dai, Jifeng},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2021},
  url       = {https://openreview.net/forum?id=gZ9hCDWe6ke}
}

@article{stolle2026efficient,
  title   = {Efficient Multi-Scale Deformable Attention on GPUs},
  author  = {Stolle, Kurt H. W.},
  journal = {Transactions on Machine Learning Research},
  year    = {2026},
  url     = {https://openreview.net/forum?id=Q4jZ7zKNKx}
}
```

### Deformable convolution

This covers `DeformConv2d` and `integrations.torchvision.deform_conv2d`.
Cite v1, and v2 as well if you pass a modulation mask.

```bibtex
@inproceedings{dai2017deformable,
  title     = {Deformable Convolutional Networks},
  author    = {Dai, Jifeng and Qi, Haozhi and Xiong, Yuwen and Li, Yi and Zhang, Guodong and Hu, Han and Wei, Yichen},
  booktitle = {Proceedings of the IEEE International Conference on Computer Vision (ICCV)},
  year      = {2017},
  pages     = {764--773},
  doi       = {10.1109/ICCV.2017.89}
}

@inproceedings{zhu2019deformable,
  title     = {Deformable {ConvNets} v2: More Deformable, Better Results},
  author    = {Zhu, Xizhou and Hu, Han and Lin, Stephen and Dai, Jifeng},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2019},
  pages     = {9300--9308},
  doi       = {10.1109/CVPR.2019.00953}
}
```

### Single-scale deformable sampling

This covers `DeformAttn2d` and `DeformSample2d`,
which implement the deformable convolution v3 (DCNv3) operator introduced in InternImage.

```bibtex
@inproceedings{wang2023internimage,
  title     = {{InternImage}: Exploring Large-Scale Vision Foundation Models with Deformable Convolutions},
  author    = {Wang, Wenhai and Dai, Jifeng and Chen, Zhe and Huang, Zhenhang and Li, Zhiqi and Zhu, Xizhou and Hu, Xiaowei and Lu, Tong and Lu, Lewei and Li, Hongsheng and Wang, Xiaogang and Qiao, Yu},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2023},
  pages     = {14408--14419},
  doi       = {10.1109/CVPR52729.2023.01385}
}
```

## Resources

- [PyTorch Blog on Triton kernel optimization](https://pytorch.org/blog/accelerating-triton/)

## License

MIT. See [LICENSE](https://github.com/tue-p8n/deformops/blob/main/LICENSE).
