# Operator registration and export

How operators register with `torch.library`
to serve `torch.compile`, `torch.export` and AOTInductor.
User recipes are in [Compile and export](../README.md#compile-and-export).

## Public and private operators

PyTorch dispatches by device type,
but the right CUDA kernel depends on GPU architecture and whether the call trains.
Each public operator (`deformops::msda`, `msda_softmax`, `attn`, `attn_softmax` and `conv`)
registers one `CompositeImplicitAutograd` kernel.
It runs the registry's selection and calls the selected adapter,
which calls a backend-private operator (plain PyTorch for eager)
that holds the autograd formula, fake kernel and device kernel.

- Eager autograd records the private operator,
  so the public kernel needs no autograd formula but must use differentiable operators.
  It runs under `no_grad`, `inference_mode` and `FakeTensorMode`.
- Dynamo records the public operator;
  AOTAutograd and Inductor decompose it,
  so compiled graphs hold the kernel selected for the traced call.
- `torch.export.export` keeps the public operator as one node;
  `ExportedProgram.run_decompositions` replaces it
  with the private operator selected on the exporting host.
- The public operator has no `chunk_step` argument.
- Registration imports only torch, so it works without Triton or CUDA.
- `define_public_op` is idempotent.
- The `Library` fragment lives for the process, since a collected `Library` removes its operators.

## CUDA operator schemas

The CUDA schemas are defined in Python (`backends/cuda/_schema.py`),
not with `m.def(...)` in C++.
The `torch.ops.deformops_cuda.*` handles and `register_autograd` targets
then exist before any library loads.
Configuration modules can register autograd at import and compile on first call.

- Each `.cu` file registers its kernels against the Python schema of the same name,
  so `define_schemas()` must run before the library loads;
  `install_msda_family` calls it first.
- `deformops.backends.cuda` does not import `_schema`,
  which would import torch on every otherwise torch-free `import deformops`.
- One forward and one backward signature serve every configuration;
  configurations differ in backward traits,
  and linear and softmax only in the in-kernel reduction.
- `core.configs` derives `EXTENSION_CONFIGS` from `OP_PREFIXES` (one prefix per translation unit),
  so every configuration with a schema has a build.
  `msda_fp32acc` is the production configuration;
  the others are named for their traits.
  Each prefix gives four operators: a linear and a softmax forward and backward pair.
- `define_schemas` checks each operator against `torch.ops.deformops_cuda`, not a module flag,
  so repeated calls, calls from other modules and calls after `importlib.reload`
  skip redefinition instead of raising.
  It never compiles or loads a library;
  importing a configuration module registers schemas, handles, meta kernels and autograd only.

## Triton operator families

Every Triton operator is a `torch.library` Triton operator.
`torch.export` keeps it as one node,
and `torch.compile` and AOTInductor compile its kernels into the graph.
Its body is also the fake kernel.

### Multi-scale deformable attention

A Triton MSDA configuration supplies the operator prefix, backend name, forward launch and backward launch;
`install_msda_family` builds the rest.

- Linear and softmax aggregation are separate operators, as in CUDA,
  because a mode integer would hide two operations behind one schema.
- The softmax backward is the linear backward kernel
  plus a host-side Jacobian correction from the normalized-weight gradient to the logit gradient.
- Every operator has a vmap rule (backward operators receive vmapped gradients)
  and autograd where it applies.
- Triton launches on the current device, not the operands' device,
  so a call on a non-current GPU would address foreign memory;
  the functional entry point switches to the operands' device.

### Single-scale deformable sampling

The attn family does the same for the output-pixel-tiled kernel.
Its configurations differ only in where the input gradient accumulates.
`install_attn_family` builds the operator pair, fake kernels, autograd and vmap rule.

- Each accumulator has its own operator pair, since Dynamo would trace a flag as a constant.
  Separate names also let a profiler tell apart two operators that share a device kernel.
- Softmax runs before the operator.
  The reference normalizes over the kernel points and then accumulates linearly,
  so the result is the same,
  and autograd handles the softmax Jacobian.
- `supports()` rejects non-CUDA devices and float64, so dispatch falls back early.

## vmap

Every operator indexes its per-sample tensors by the batch index.
The vmap rule merges the vmapped axis into the batch,
calls the operator once over `size * B` samples, and splits the results.
Nothing is summed across the merged axis,
so the fold also works on backward operators.

- Arguments shared by the whole batch
  (the level table, a convolution's weight and bias)
  raise when batched.
  Folding them would give a wrong answer without an error,
  and a loop would cost one launch per element.
  Pass them with `in_dims=None`.
- The fold copies unvmapped arguments,
  so vmap over query sets against one feature map copies the feature map.
- A zero-element placeholder for an unrequested gradient has no batch to split,
  so it becomes an empty `(size, 0)` tensor per element
  instead of an ambiguous `-1` extent.
- Positions not listed as shared are folded when they hold a tensor and passed through otherwise,
  which covers integer geometry and optional tensors.

### Function transforms

`torch.library.register_autograd` builds an `autograd.Function` without `setup_context`,
which `Function.apply` requires under functorch transforms,
so on torch 2.11 `torch.func.grad`, `jacrev` and `vmap(grad(...))` raise for these operators.
Use vmap over the forward followed by ordinary autograd,
which differentiates through the folded launch and gives exact gradients.

## Tracing and graph breaks

- The eager backends trace without graph breaks.
  MSDA uses one tensor-indexed bilinear gather that Inductor can lower.
  Attn and conv have no `.item()`, `.any()` or branch on a tensor value.
  The eager attn backend imports nothing that compiles.
- The eager attn loop does not short-circuit on `valid_loc.any()`:
  reading it would synchronize and break the graph,
  and multiplying by `valid_loc` already zeroes samples with nothing in bounds.
- The registry filters options with a membership scan,
  since Dynamo cannot trace `options.keys() & ...` on every supported PyTorch.
- The reference implementations read tensor values and are not traceable;
  the MSDA reference calls `.item()` and breaks once per level.
- A backend's first dispatch imports its module, which breaks the graph,
  so tests that count graph breaks warm the backend with an eager call first.
- Model integrations share the aggregation step in `integrations/_ops.py`
  and accept a concrete, traceable backend object for `torch.export`.

## Model export recipes

`tests/deformops/models/test_export.py` runs small random-weight Mask2Former and Deformable DETR models
on eager, Triton, CUDA and `backend=None`,
and compares each recipe with the uncompiled model.

| Recipe | Checks |
| --- | --- |
| `torch.compile(fullgraph=True)`, inference | outputs match |
| `torch.compile(fullgraph=True)`, forward and backward | loss and parameter gradients match |
| `torch.compile` in training mode | no graph break happens inside deformops |
| `torch.export`, static batch size | outputs match |
| `torch.export` with `backend=None`, then `run_decompositions` | the export holds the public operator; the decomposed program does not, and outputs match |
| `torch.export`, dynamic batch size (Mask2Former only) | outputs match on a different batch size |
| AOTInductor package loaded in a fresh process | outputs match |
| Torch-TensorRT, when installed | outputs match |

- AOTInductor packaging compiles in the private operator that `run_decompositions` selects on the host.
- The fullgraph compile runs in eval mode, disabling dropout and the models' `isfinite` check.
  In training both upstream models branch on `isfinite(...).all()`,
  which breaks the graph on every backend,
  so the training test only asserts no break inside deformops.
- Deformable DETR's export fixes the batch size:
  its pixel-mask resize puts the batch in the channel position, on which the CUDA nearest-neighbor resize specializes.
- A fresh-process AOTInductor package runs without importing deformops
  for Triton, eager and `backend=None` programs (`backend=None` selects Triton on a GPU).
  A program that calls the CUDA operators needs their library loaded first.
- `tests/deformops/models/test_mask2former.py` exports the patched Mask2Former through the eager backend.

## Mask2Former

`deformops.models.Mask2Former` loads an official `Mask2FormerForUniversalSegmentation` checkpoint
with patched modules, so the pixel decoder's deformable attention runs on deformops
and architecture, weights and post-processing are unchanged.
`from_pretrained` fetches and patches, `forward` delegates, and `.model` is the wrapped graph.

- The patches remove what breaks `torch.export`:
  data-dependent Python control flow, `einsum` and in-place tensor writes.
  The deformable attention is one registered operator.
- The default checkpoint is `facebook/mask2former-swin-base-coco-instance`;
  `facebook/mask2former-swin-{tiny,small,base,large}-*`
  and the `-coco-panoptic` and `-ade20k-semantic` variants load the same way.
- The base install has neither transformers nor torchmatch;
  without them the module raises an error naming `deformops[mask2former]`.

Overrides in `models/mask2former/model_patches.py`:

- `PatchedMask2FormerMaskPredictor` is a drop-in for `Mask2FormerMaskPredictor`
  that replaces `einsum("bqc, bchw -> bqhw")` with `matmul`.
- `PatchedMask2FormerHungarianMatcher` replaces `scipy.optimize.linear_sum_assignment`
  with `torchmatch.assignment.solve` on the upstream cost matrix, moved to CPU float32 as in the scipy path.
- `PatchedMask2FormerLoss` skips `super().__init__`,
  because upstream builds a scipy matcher and asserts scipy is installed (`requires_backends(self, ["scipy"])`).
  It keeps every weight and buffer, swaps in the patched matcher,
  and inherits the loss math.
- The loss patch installs at import and is idempotent,
  so a model built before wrapping also avoids scipy.

## Operator checks

### opcheck

`tests/deformops/test_opcheck.py` runs `torch.library.opcheck`, which verifies that:

- an operator's schema matches its aliasing and mutation behavior;
- its fake (meta) implementation agrees with the real kernel;
- autograd is registered against the right dispatch key;
- the operator survives AOTAutograd tracing under static and dynamic shapes.

How the module is built:

- `SAMPLE_INPUTS` maps `"namespace::op_name"` to a builder returning `(args, kwargs)`.
  `test_sample_registry_is_complete` imports every kernel module
  and fails by name for any operator in `torch.ops.deformops`, `deformops_cuda` or `deformops_triton`
  with neither a builder nor a stand-in whose schema is identical.
- Each of the five subtests has its own parametrized test,
  because `raise_exception=True` stops at the first failure.
  Each is a hard assertion except strict xfails recorded in `_XFAIL`.
  A test checks that `_XFAIL` and `_BACKWARD_CUSTOM_OPS` name real operators and subtests,
  since a stale key would stop attaching the mark without an error.
- Each Triton MSDA configuration registers four operators under `msda_<name>_` with the same arguments,
  so one pair of builders covers all of them.
  Caching configurations return online-softmax statistics (`(B, Q, G)` float32) from the softmax forward.
  Their softmax backward takes them back, with eight arguments instead of six.
  A non-caching softmax backward takes the linear one's six.
- Shape tensors are int64 for the CUDA operators and int32 for Triton;
  the `msda_query_tiled` wrapper casts before dispatch.
- The six CUDA configurations share one schema and factory.
  Checking the production configuration plus one other
  covers both registration paths of `install_msda_family`
  without building every extension;
  the rest differ only in backward traits, which the parity tests cover.
- `deformops::msda` and `msda_softmax` select per call,
  so opcheck checks the private operator selected on the host.
  The attn and conv public operators reuse their Triton forwards' builders.
- Shapes are minimal because several operators compile on first call.
- Expected failure:
  the raw Triton `*_backward` helpers are `torch.library.custom_op`s without `register_autograd`,
  since they run only inside their forward's autograd bridge with grad tracking off.
  The AOT subtests use an input with `requires_grad=True`,
  so AOTAutograd traces a backward through `custom_op`'s Autograd-key fallback, which raises.
  The schema-only `deformops_cuda` backward operators (`torch.library.define`)
  fall through instead, a path PyTorch marks deprecated.
- The conv backward schema returns a zero-element placeholder per unrequested gradient,
  since it has no optional-tensor return and a shared placeholder would alias outputs.
  `test_schema` checks aliasing; the AOT subtests are excluded as for the xfail.

### Public operators

`tests/deformops/test_public_ops.py` covers the public operators.

- CPU tests check:
  - the exact schema;
  - parity with an explicit eager call, forward and backward;
  - `inference_mode` and fake tensors;
  - the public operator in the Dynamo graph but not in the AOTAutograd graphs;
  - dynamic shapes;
  - `torch.export` round trips;
  - opcheck;
  - a vmap rule that batches instead of looping.
- GPU tests check which kernel a CUDA call selects per architecture and training precision.
  Half-precision training before SM 9.0 selects the float32 accumulator,
  other calls the native one, and eager-mode calls select the same kernel.
  GPU tests also check parity with the device backends at the Triton float32 tolerance.
- Each operator section defines an operand factory and appends `PublicOpCase` entries to `CASES`;
  generic tests follow, and a new operator adds a section above them.
- The test keeps Triton operators whole in AOTAutograd graphs, so they name the selected kernel
  instead of an unnamed launch.
- The test passes `chunk_step` as the layer does
  and applies selection options and overrides in its own selection,
  since the public kernel ignores them.
- The vmap batching fallback warns from C++, outside Python warnings,
  so the test counts kernel runs: once for a vmap rule, once per sample for the fallback.
- Test geometries:
  - conv uses a 3x3 kernel with per-axis padding and dilation that keep an S by S input at S by S,
    so an axis swap changes values without changing shapes;
  - attn uses kernel 3x2, stride `(1, 2)`, padding `(1, 0)` and dilation `(2, 1)`;
  - the CUDA layout uses 4 points per level and a per-head channel count divisible by 8,
    as the CUDA kernel templates require;
  - the attn keywords mix pairs with separate `dilation_h` and `dilation_w`,
    all normalized by the registry.
