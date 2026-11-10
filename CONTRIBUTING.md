# Contributor guide

This file describes the project's layout, conventions, commands and CI.
Where it disagrees with the code, the code wins,
and the fix belongs in this file.
Design rationale lives in `docs/`:
[operator conventions](docs/conventions.md),
[numerical precision](docs/precision.md),
[backend dispatch](docs/dispatch.md),
[kernel loading and builds](docs/kernel-loading.md),
[operator registration and export](docs/export.md),
[kernel design](docs/kernels.md)
and [test design](docs/testing.md).

## Project overview

DeformOps is a PyTorch library of custom operators
for deformable sampling with bilinear interpolation.
It has three operators:
classic deformable convolution (Deformable ConvNets, Dai et al., ICCV 2017),
multi-scale deformable attention (Deformable DETR, Zhu et al., ICLR 2021),
and single-scale deformable sampling (InternImage, Wang et al., CVPR 2023).
[Efficient Multi-Scale Deformable Attention on GPUs](https://openreview.net/forum?id=Q4jZ7zKNKx)
(TMLR 2026)
describes the kernels.

Version 3.1.0 (from `VERSION`), Python 3.12 or 3.13, MIT license.

## Performance operating point

Performance work optimizes for multi-scale deformable attention (MSDA)
in the Mask2Former pixel decoder,
which dominates training time in Mask2Former-style segmentation.

- `Q` (queries) is the sum of spatial positions over the 1/8, 1/16 and 1/32 levels,
  typically 5k–22k and always encoder-scale (`Q ≫ 1024`).
- Each call uses `G = 8` heads, `D = 32` channels per head (dim 256), `L = 3` levels and `K = 4` points per level.
- The pixel decoder runs MSDA 6× per forward.

Approximate `Q` per dataset, common training crops:

| Setup                    | Crop        | Pixel-decoder `Q` (1/8 + 1/16 + 1/32) |
| ------------------------ | ----------- | ------------------------------------- |
| COCO instance / panoptic | 1024²       | ≈ 21 500                              |
| Cityscapes               | 512 × 1024  | ≈ 10 750                              |
| ADE20K (semantic)        | 640²        | ≈ 8 400                               |
| ADE20K (semantic)        | 512²        | ≈ 5 400                               |
| YouTube-VIS, per frame   | ≈ 480 × 853 | ≈ 8 400                               |

"Encoder scale" in a benchmark or performance note means this regime.
Decoder-scale workloads (`Q ≤ ~1024`, e.g. Deformable DETR object queries)
are kept correct but not tuned.

## Repository structure

Each backend package has one registry adapter per operator
(`msda.py`, `attn.py`, `conv.py`),
`registry` has one dispatch module per operator,
and `core` holds what the operators share.

```
sources/deformops/
  __init__.py                             # Lazy re-export of backends + nn + registry; no torch at import
  registry/                               # Per-operator dispatch contract + the Registry instances
    _dispatch.py                          # Backend StrEnum + generic Registry[T] (context, policy, supports)
    _public_ops.py                        # deformops:: public operators whose kernel selects a backend
    _vmap.py                              # The batch fold every operator's vmap rule uses
    msda.py                               # MSDA: Context + Backend protocol + helpers + paper pick + msda
    attn.py                               # Single-scale sampling: Context + Backend protocol + attn
    conv.py                               # Classic deformable convolution: Context + Backend protocol + conv
  version.py                              # Version and environment detection
  backends/                               # Compute backends; each registers into the registry
    cuda/                                 # C++/CUDA multi-scale kernels
      __init__.py                         # Lazy submodules
      _schema.py                          # Python torch.library schemas (resolve without a .so)
      _msda_bridge.py                     # Shared autograd / meta / reference code
      msda_fp32acc.{cpp,cu,py}            # The CUDA backend: fp32 accumulator, warp reduce
      msda.py                             # Registry adapter for the CUDA kernels
      attn.py / conv.py                   # Backend.CUDA aliases of the Triton kernels
      include/                            # Shared headers every CUDA extension includes
        msda_args.cuh                     # Argument structs, launch checks, include order
        msda_shared_memory.cuh            # Backward shared-memory extents + reduction pick
        msda_forward.cuh                  # Forward kernel + dispatch ladder
        msda_backward.cuh                 # Backward kernel + dispatch ladder
        msda_launch.cuh                   # Host entry points
        msda_boxed.cuh                    # Stable-ABI boxed entry points; what each .cu includes
        utils.cuh / interpolate.cuh / stable_compat.cuh
      experimental/                       # Measurement configurations, not dispatched
        msda_serialreduce.{cpp,cu,py}     # fp32 accumulator, serial reduce
        msda_half2atomic{,_serialreduce}.*  # Native dtype, packed half2 atomics
        msda_scalaratomic{,_serialreduce}.* # Native dtype, unpacked atomics
        msda_morton.py / msda_hilbert.py  # Host-side query reordering, no kernel
        _query_order.py                   # Space-filling-curve reordering helpers
    triton/                               # Triton kernels; requires Triton to import
      __init__.py                         # Lazy submodules
      _kernels/                           # Every @triton.jit kernel, nothing else
        msda_query_tiled.py               # Query-tiled online-softmax sampling
        attn_query_tiled.py               # Single-scale sampling
        conv_deform2d.py                  # Classic deformable convolution
      _msda_bridge.py                     # MSDA: 4 Triton operators + autograd + vmap
      _attn_bridge.py                     # The same for single-scale sampling
      _launch.py                          # Forward tile shapes
      _vmap.py                            # Re-exports registry/_vmap.py for the three bridges
      msda_query_tiled{,_fp32acc}.py      # The Triton MSDA configurations
      attn_query_tiled{,_fp32acc}.py      # Single-scale sampling
      conv_deform2d.py                    # Classic deformable convolution
      msda.py / attn.py / conv.py         # Registry adapters
      experimental/                       # Measurement configurations, not dispatched
        _kernels/                         # Their kernels
        msda_query_tiled_cached{,_fp32acc}.py
        msda_query_tiled_autotuned{,_fp32acc}.py
        msda_point_parallel.py
        msda_atomic_wide.py
        msda_{block,warp}_reduce{,_wide}.py
        msda_sort_segscan{,_wide}.py
    eager/                                # The eager PyTorch backend
      msda_pytorch.py                     # MSDA, torch.compile-traceable
      attn_pytorch.py                     # Single-scale deformable sampling
      conv_pytorch.py                     # Classic deformable convolution
      msda.py / attn.py / conv.py         # Registry adapters
    reference/                            # Reference implementations; not dispatchable
      msda.py / attn.py / conv.py         # Parity targets every backend is measured against
  nn/                                     # Neural network building blocks
    msda.py                               # MultiScaleDeformAttn2d
    attn.py                               # DeformAttn2d / DeformSample2d
    conv.py                               # DeformConv2d
  models/                                 # Pre-built model graphs + their deformops parts
    __init__.py                           # Lazy re-export of the full graphs only
    mask2former/                          # HF graph + MSDA / mask / torchmatch-loss overrides
    deformable_detr/                      # HF Deformable DETR MSDA core -> deformops
  core/                                   # Shared machinery, private
    build.py                              # Bundled, cached, downloaded or compiled kernel libraries
    configs.py                            # EXTENSION_CONFIGS, EXT_PACKAGE, shared compiler flags
    capabilities.py                       # Triton / JIT-toolchain probes
    errors.py                             # Kernel-loading and dispatch errors and warnings
    runtime.py                            # Method enum and shared call-context helpers
    sanitize.py                           # Parameter and operand validation
    tuning.py                             # Stride/thread heuristic + find_factors
  integrations/                           # Third-party compatibility shims
    __init__.py                           # docstring only; imports nothing
    _ops.py                               # run_msda: the aggregation bridge
    _patch.py                             # generic recursive module replacement
    transformers.py                       # HF MSDA core + Mask2Former attention
    mmcv.py                               # mmcv MultiScaleDeformableAttention
    torchvision.py                        # deform_conv2d, DeformConv2d: drop-in
                                          #   for torchvision.ops

tests/deformops/                          # Operator and integration tests
  _internal/                              # Shared build/runtime machinery
  integrations/                           # Third-party shims
  models/                                 # Pre-made models and their components

nix/                                      # Nix shells, ops.nix and wheel.nix per CUDA version
```

`sources/` is the only tree published to PyPI.
`tests/deformops/test_distribution.py` checks that package discovery finds every package
and that the package data covers every kernel source and header,
since a reorganization can drop either without an import failing.

## Development commands

Nix owns the environment and every command;
there is no justfile and no uv invocation outside a shell.

```bash
# ruff, clang-format, shell (shfmt, shellcheck); deadnix removes dead
# nix bindings, not formatting
nix fmt

# Every check: formatting, hooks, ruff, pyright, the CPU suite
nix flake check

# One check on its own
nix build .#checks.x86_64-linux.tests --no-link

# The fast local loop
nix develop .#cpu
#   pytest tests/deformops/test_attn.py -q
#   pyright
#   ruff check

# GPU work
nix develop .#cu128          # also cu126, cu129, cu130, cu132
nix run .#test-gpu           # the full GPU suite
nix run .#test-gpu-quick     # only the gpu-marked subset
nix run .#test-gpu-fp64      # the double-precision tests, through a JIT build
nix run .#check-gpu          # driver, nvcc, and a real extension build

# Wheels
nix build .#wheel            # generic, pure Python, what goes to PyPI
nix build .#wheel-cu128      # per-CUDA, bundling prebuilt kernels

# clangd include paths (compile_flags.txt)
nix run .#clangd-setup
```

CI runs `nix flake check` on the same flake and tool versions,
so a green local check means the CI checks pass.
CI also installs the built distribution into a clean environment
on both supported Python versions, which `nix flake check` does not.

### Environments

- **GPU work: `nix develop .#cu128`**
  (also `cu126`, `cu129`, `cu130`, `cu132`, `cu134`)
  supplies nvcc, the profiler, the host CUDA driver on the loader path
  and a virtual environment with a working CUDA torch.
- **CPU work: `nix develop .#cpu`**
  has the CPU torch, the development dependency group, pyright and the clang tools.
  The dev group includes the model dependencies that import on a CPU torch
  (transformers, torchmatch, timm, mmcv-lite), so their tests run here.

Each shell has its own virtual environment,
so the CPU and CUDA torch builds never share a directory.
CUDA toolchain setup comes from the shared
[`tue-p8n/nix`](https://github.com/tue-p8n/nix) uv2nix builder (`lib.uv.mkUv2nix`);
toolchain changes belong there.

## Code quality

- **Linter and formatter:** ruff (line length 88, target py313),
  with `check` and `format --check` enforced in CI.
  Rule set:
  F, E, B, C, I, ARG, C4, EM, ICN, G, PGH, PIE, PL, PT, PTH, RET, SIM, T20, UP, NPY, FURB, PYI.
  PLR2004 (magic values) applies to `sources/` and is waived for tests and CI scripts.
- **Type checking:** pyright strict on `sources/` (Python 3.13, Linux),
  enforced by the `typecheck` check at zero errors.
  Suppressions are per line and name the stub gap they work around.
  `ty` is configured over `sources` and `tests` but not enforced.
- **Coverage:** `fail_under = 77` on the CPU CI run,
  a few points below a development machine's figure,
  because the runners have no CUDA toolchain.
  The Triton backend modules are omitted from the count.
  Raise the threshold as more paths become CPU-testable.
- **Tests:** pytest with hypothesis, pytest-benchmark and pytest-xdist.
  Markers are `gpu`, `integration`, `slow`, `unit`, `notebooks` and `fp64`;
  CI excludes `gpu` and `integration`.

## CI

`.github/workflows/ci.yml` runs on push and pull request to `main`, on Nix
(`DeterminateSystems/determinate-nix-action`,
cached by `DeterminateSystems/magic-nix-cache-action`).

- `checks` runs `nix flake check`, which covers
  `lint` (ruff),
  `treefmt` (every other language, including clang-format on the CUDA sources),
  `typecheck` (pyright strict against a CPU environment with Triton),
  `tests` (the CPU suite and the coverage threshold)
  and `pre-commit`.
- `package` builds the wheel through the flake,
  then an sdist and wheel the way `release.yml` does.
  It installs the wheel with `pip`/`uv` into a clean environment on Python 3.12 and 3.13
  and runs `.github/scripts/wheel_smoke.py` from outside the checkout.
  No other job tests an installed package.

The test suite runs on the interpreter the uv2nix builder supplies, currently 3.13.
Nothing in CI needs a GPU or a CUDA toolchain.

Publishing uses two workflows.

- `release.yml` runs on `release: published`
  and requires the tag to be `v` followed by `VERSION`.
  It builds the pure-Python wheel and sdist (`uv build`, `py3-none-any`)
  and smoke-tests the wheel against a CPU torch.
  It then publishes to PyPI through trusted publishing
  and attaches the files to the GitHub release.
- `pages.yml` runs only on `workflow_dispatch`,
  since a per-CUDA build takes about an hour.
  Per CUDA version (`cu126`/`cu128`/`cu129`/`cu130`/`cu132`/`cu134`)
  it builds the stable-ABI operator libraries (`nix build .#ops-<cuXXX>`)
  and a wheel bundling them (`nix build .#wheel-<cuXXX>`),
  attaches the wheels to the matching GitHub release when present,
  and deploys both with a PEP 503 index to GitHub Pages.

## Releasing

1. Run `nix run .#test-gpu` and `nix run .#test-gpu-fp64` on a CUDA machine,
   since CI cannot verify a CUDA or Triton path.
   A first run on a cold Triton cache spends most of its time
   autotuning the deformable convolution kernels.
2. Set `VERSION` and merge with a green CI run.
3. Publish a GitHub release tagged `v` followed by that version.
   `release.yml` rejects a tag that does not match `VERSION`,
   smoke-tests the wheel and publishes it to PyPI;
   a re-run skips files PyPI already has.
4. To refresh the per-CUDA libraries and wheels,
   dispatch `pages.yml`.

Because CI has no GPU,
several CPU tests compare declarations across files:
the extension path across four build systems
and the CUDA versions across three.

## Key conventions

- **Tooling:** uv drives dependencies through the Nix flake;
  the normal loop has no direct `uv sync`, `uv run` or `uvx`.
  The source layout is `sources/`, not `src/`.
- **Pre-build hook:** `tests/deformops/conftest.py` loads the CUDA backend
  through `msda.load(Backend.CUDA)` once, before xdist workers start,
  so no compile runs inside a timed test.
  The test session compiles locally and never downloads.
- **Every operator with a kernel has a reference and a traceable eager backend.**
  References in `backends/reference/` are not dispatchable,
  so only an explicit import reaches them.
  Nothing in `backends/eager/` may read a tensor value
  (no `.item()`, no `.any()`, no branch on data),
  so `torch.compile` traces it without a graph break.
  `tests/deformops/test_attn_eager.py` and `tests/deformops/test_conv_eager.py`
  check for breaks with `torch._dynamo.explain`,
  since `torch.compile(fn)` does not report one.
  See [operator conventions](docs/conventions.md).
- **Backends register into the registry.**
  `registry/_dispatch.py` holds the `Backend` StrEnum (`CUDA`, `TRITON`, `EAGER`) and the generic `Registry`;
  each operator module holds its `Context`, `Backend` protocol and `Registry` instance.
  An adapter registers at import when its capability is present,
  declares `supports(context)` and a `load` hook,
  and otherwise records why with `mark_unavailable`.
  The registry rejects raw strings:
  pass a `Backend` member or a concrete backend object.
  See [backend dispatch](docs/dispatch.md) for the selection rules and errors.
- **Public operators select a kernel.**
  `deformops::msda` and its siblings register one `CompositeImplicitAutograd` kernel
  through `registry/_public_ops.py::define_public_op`.
  It calls `Registry.run_selected`,
  and `Registry.route_selected_calls` sends `backend=None` registry calls through it.
  The selected backend's private operator has the autograd, fake and device kernels.
  A public kernel's context must hash under fake tensors and dynamic shapes.
  Read the extents that decide support as `int`,
  and keep batch, query and spatial sizes out of the memo key.
  See [operator registration and export](docs/export.md).
- **The functional MSDA default method is linear**,
  set only in `core/runtime.py::resolve_method`.
  `MultiScaleDeformAttn2d` asks for softmax explicitly.
- **The MSDA selection policy never runs eager on a GPU, except for float64.**
  `registry/msda.py::select_backend` is a pure function of the `MSDAContext`,
  so `tests/deformops/test_registry_policy.py` tests it without a GPU.
  See [numerical precision](docs/precision.md) for the float32 accumulator below SM 9.0.
- **Kernel configurations share bridges.**
  The CUDA backend and the five experimental CUDA configurations
  include `backends/cuda/include/msda_boxed.cuh`
  and share `backends/cuda/_msda_bridge.py`.
  Each Triton configuration is one module over `_msda_bridge.py` or `_attn_bridge.py`,
  with its kernels under `_kernels/`,
  and states only its operator prefix, backend name and two launches.
  Call an operator through `torch.ops.<namespace>.<name>`,
  which Dynamo traces, not through the registered function object.
  See [kernel design](docs/kernels.md).
- **Operand validation** lives in `core/sanitize.py`,
  which all three functional entry points call before dispatch.
  Checks that read the level table back to the host run only with `DEFORMOPS_VALIDATE`.
  Add new checks there, not at a call site.
- **Public surface:** `deformops.nn` (layers), `deformops.backends` (implementations)
  and `deformops.registry` (dispatch); shared machinery is private under `core`.
- **Models are upstream graphs plus integrations plus minimal overrides.**
  `deformops.integrations.<library>` has the replacement `nn.Module`
  and a `patch(model)` helper, which raises `RuntimeError` when nothing matched unless `strict=False`.
  `integrations/_patch.py` and `integrations/_ops.py` hold the shared replacement and the `run_msda` bridge.
  Each `deformops.models.<name>` package keeps the graph in `model.py`
  and its overrides in `model_patches.py`, both re-exported lazily,
  so `import deformops` imports no torch.
  `tests/deformops/models/test_export.py` holds the compile and export recipes.
  Extras: `[mask2former]` (transformers, plus torchmatch on Python 3.13),
  `[deformable-detr]` (transformers and timm) and `[mmcv]` (mmcv-lite).
- **Style:** type annotations throughout
  (`TYPE_CHECKING`, `Final`, `Literal`, `override`);
  NumPy docstrings;
  logging instead of print (`T20`);
  pathlib instead of os.path (`PTH`, `setup.py` exempt).
- **Half precision:** the kernels support float16 and bfloat16;
  bfloat16 is the recommended training dtype.

## Architecture notes

- The CUDA backend loads its library from a bundled per-CUDA wheel,
  an earlier local compile,
  a download (`DEFORMOPS_DOWNLOAD`, off by default)
  or a compile (`DEFORMOPS_COMPILE`, off by default).
  Only `configs.EXT_PACKAGE` declares where libraries live,
  and `tests/deformops/_internal/test_ext_path.py` fails when the build systems disagree.
  See [kernel loading and builds](docs/kernel-loading.md).
- Every CUDA extension compiles with `--use_fast_math`
  (`SHARED_CUDA_CFLAGS` in `core/configs.py`),
  so results can differ from strict IEEE evaluation in the last bits.
- `setup.py::get_extensions` compiles what `configs.select_extensions` picks from `DEFORMOPS_EXTENSIONS`.
  Unset, only a `--no-build-isolation` install from the sdist compiles the non-experimental kernels,
  and a checkout compiles nothing.
  The release and CI workflows build with `DEFORMOPS_EXTENSIONS=none`,
  so the PyPI wheel is pure Python.
- Operator names follow the defining file.
  The public operators are `deformops::msda`, `msda_softmax`, `attn`, `attn_softmax` and `conv`.
  A private operator in `backends/<backend>/<stem>.*`
  registers as `deformops_<backend>::<stem>_forward` and `<stem>_backward`,
  plus `_softmax_forward` and `_softmax_backward` for an MSDA configuration.
  For a CUDA kernel the stem is also the extension name and `EXTENSION_CONFIGS` key.
  Backend names (`"cuda"`, `"cuda_half2atomic"`, `"triton_query_tiled"`) are user-facing and do not follow this rule.
- Each registry imports its three backend adapters on first dispatch,
  so importing the package imports no compiled or Triton module.
