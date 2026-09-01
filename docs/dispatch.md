# Backend dispatch

How a call chooses its backend, and how integrations patch models.
The README summarizes this under
[Backend selection](../README.md#backend-selection).

## Registry layout

- Each operator has one module
  (`registry.msda`, `registry.attn`, `registry.conv`)
  holding its call context, backend protocol, normalization helpers,
  selection policy and registry instance.
  `deformops.registry` is the public package.
- The `Backend` enum has `CUDA`, `TRITON` and `EAGER`.
  The reference implementations have no member;
  only an import of `deformops.backends.reference` reaches them.
- The registry copies the operator's signature and docstring onto its instance,
  so `help()`, IDEs and `inspect.signature` show the operator.

## The registry

- A registry maps `Backend` members to functions.
  The operator supplies a hook that turns a call into a context
  and a policy that orders the candidates.
  The dispatch module imports no torch.
- Backend adapters are imported once per operator,
  from `deformops.backends.<backend>.<operator>`.
  An adapter that cannot run on the host (no Triton, no CUDA device)
  marks its backend unavailable,
  as does one whose import raises, with a warning.
- The loaded flag is set under a lock after every import finishes,
  so a concurrent first call never reads a partial registry.

## Selection with `backend=None`

- The registry runs the first candidate that is registered,
  supports the call,
  and has its kernel available.
  A candidate's load hook runs first
  and raises `KernelNotProvidedError`
  when the kernel needs a download or compile the environment disallows.
- Each skipped candidate emits `BackendFallbackWarning`
  (`DEFORMOPS_FALLBACK_WARNINGS=0` silences it).
  When no candidate can run,
  `NoBackendError` lists each one with its reason.
- The outcome is memoized per call context.
  The chosen backend receives the policy's options;
  explicit keyword arguments override them.
- `resolve()` takes the call's arguments.
  With `backend=None` it runs the selection, including kernel loading.
  With an explicit backend it checks that the backend can run the call
  and loads its kernel library,
  so the report matches what the call would do.
- `load(backend)` (for example `msda.load(Backend.CUDA)`)
  makes a backend ready before the first call and returns a `LoadedBackend`
  with the library path, if the backend has one.

### Routing through the public operators

An operator with a public `torch.library` operator
routes `backend=None` calls through it,
so tracing records the public operator.
The operator's kernel runs the selection and the backend loading,
so Dynamo traces a first call without entering the registry's load lock.

## Selection policies

| Call                        | Multi-scale attention                         | Sampling and convolution                 |
| --------------------------- | --------------------------------------------- | ---------------------------------------- |
| CPU tensor                  | eager                                         | eager                                    |
| float64, any device         | eager                                         | eager                                    |
| CUDA tensor                 | Triton, then CUDA                             | Triton only                              |
| Eager on a CUDA tensor      | only when `Backend.EAGER` is requested        | only when `Backend.EAGER` is requested   |
| `Backend.CUDA`              | the CUDA kernel                               | the Triton kernel (no C++/CUDA kernel)   |

A GPU call runs a device kernel or raises.
For multi-scale attention,
half-precision training before SM 9.0 uses Triton's float32 accumulator
and every other call the native one;
see [precision](precision.md).

## Call contexts

A context is frozen and hashable,
so it is also the memoization key.

- **Multi-scale attention.**
  `sm_version`, `is_training` and `dtype` decide the production pick;
  `device`, `num_levels`, `num_heads`, `num_points` and `head_dim`
  decide which kernels can run.
  `num_queries` is excluded from equality and hashing,
  so varying input resolutions do not grow the memo.
  The other extents are read as Python integers,
  so a compiled graph specializes on them.
  A symbolic query count is recorded as 0.
  The count is then dynamic, and a memoized context holds no trace state.
  The builder validates operands before reading any extent,
  so a malformed call raises the same `ValueError` on every backend,
  never an `IndexError`.
- **Single-scale sampling.**
  `sm_version`, `is_training` and `dtype` decide the Triton accumulator;
  `device` decides which backends can run.
  The context reads no tensor extent, so a compiled graph specializes on no size.
- **Convolution.**
  The context holds only `device` and `dtype`,
  so a compiled graph specializes on no extent.

## Backend keywords

### Multi-scale attention

- An explicit `Backend.CUDA` call outside the CUDA kernel templates raises,
  and `backend=None` never selects CUDA for it.
  The templates cover a head dimension divisible by 8,
  1 to 5 levels,
  4 or 8 points per level,
  and at most 1024 threads per block,
  counting the `G * D / 8` forward threads and the backward's own stride.
- Triton reads `fp32acc`, which the policy sets per architecture.
  CUDA reads `points`, `channels_per_thread` and `threads_per_block`,
  derived from the operands when not given.
  A keyword from this set that the selected backend does not read is ignored;
  any other keyword raises `TypeError`.
- The public operators take none of these keywords.
  A `backend=None` call that passes one runs its backend in Python,
  and tracing records the backend's operator.
- `chunk_step` sizes only the CUDA backend's batch chunks.
  It does not reach the backend on the public-operator path,
  where CUDA uses its default chunk size;
  see [conventions](conventions.md).
- `nn.MultiScaleDeformAttn2d` given a backend instance
  calls its `forward` directly, without the functional operand checks.

### Single-scale sampling

- Every backend builds its call through `resolve_attn_call`,
  so spellings, defaults and checks match across backends.
- Geometry is an int,
  an `(h, w)` pair,
  or paired `kernel_h`/`kernel_w`, `stride_h`/`stride_w`, `pad_h`/`pad_w`
  and `dilation_h`/`dilation_w` keywords, which take precedence.
- The method defaults to linear (`core.runtime.resolve_method`).
  Structural operand checks run on every call.
- `fp32acc=False` accumulates the input gradient in the input dtype;
  an explicit Triton or CUDA pick defaults to `True`.
- The public operators take separate geometry integers and no `fp32acc`.
  The route normalizes and validates the geometry before the operator runs.
  A `backend=None` call passing `fp32acc` runs its backend in Python.

### Convolution

- Every backend builds its call through `resolve_conv_call`,
  so spellings, defaults and checks match across backends.
- `deformops::conv` puts the mask before the weight, as the kernels do;
  its kernel reorders the registry's functional order.
- Operand checks run inside the operator.

## Model patching

- Integrations supply classes and factories;
  `integrations/_patch.py` does the traversal.
- One walk covers all classes and visits the children of a replacement.
  It finds a class nested in another,
  such as a Mask2Former attention inside its encoder layer,
  and walks no subtree twice.
  The first matching class wins.
- A patch that matches nothing usually means upstream renamed or moved the class.
  `strict=True`, the default, raises `RuntimeError`;
  `strict=False` patches the classes that match.

### Hugging Face Hub kernels

Upstream tags its multi-scale attention class for replacement by a Hub kernel
when the `kernels` package is installed and a model is kernelized.
A replacement inheriting that tag would lose the deformops forward without a warning.
The replacements set `kernel_layer_name = None`,
so kernelizing keeps the deformops forward and warns.

### MMCV

- mmcv-lite from PyPI has no compiled extension,
  so `MultiScaleDeformableAttention` runs `multi_scale_deformable_attn_pytorch`.
- `mmcv.ops` loads the extension at import,
  so an empty namespace stands in for the absent `mmcv._ext`.
- The patch replaces every attention instance and that module-level function.
