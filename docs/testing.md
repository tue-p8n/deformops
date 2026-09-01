# Test design

Why the tests under `tests/deformops/` are built as they are.
[CONTRIBUTING.md](../CONTRIBUTING.md#development-commands) lists the commands that run them.

## Session setup

- No prebuilt index serves a source checkout,
  so `pytest_configure` turns downloads off and kernels compile on demand;
  download tests re-enable it through `monkeypatch`.
  Some build variables are set only when unset,
  so a session can still request the `always` or `rebuild` strategy.
- At session start, `msda.load(Backend.CUDA)` loads only the production CUDA multi-scale kernel,
  since compiling every similar-size configuration would add minutes.
  Compiles across xdist workers already serialize on `build.serialized_build`,
  so this pass only saves time.
  A failure warns once, and each test then loads, fails or skips on its own.
  `DEFORMOPS_TEST_PREBUILD=0` skips the pass.
- xdist workers re-run module-level setup so coverage records it;
  only the main process reports coverage.
- `set_float32_matmul_precision` is deprecated on PyTorch 2.9 and later,
  but setting `fp32_precision` directly breaks Inductor's legacy `allow_tf32` probe,
  so the suite keeps the call and silences only the notice.
- The CUDA context and the lazily allocated cuBLAS handle pool
  are initialized on the main and autograd worker threads.
  Otherwise the first worker cuBLAS call in a `gradcheck` backward warns that no CUDA context is current.
- A CUDA-built torch without a CUDA runtime (a `cuXXX` environment on a CPU-only host)
  aborts with SIGABRT in `torch.backends.cudnn.version()`,
  which `deformops.version.environment` calls when cuDNN reports itself available.
  The session disables cuDNN, which sends the probe to the `-nocudnn` tag.
  A CPU-only torch has no cuDNN and is left alone.

## Devices and markers

- Tests taking the CUDA-only `device` fixture are marked `gpu` before `-m` deselection.
- A module whose reference and eager backend both run on the CPU
  overrides `device` with `any_device`,
  so the CPU suite runs the same comparison.
  Only the CUDA case is marked `gpu`.
- Benchmarks take `device`, so they run in the GPU suite and skip without CUDA;
  reference benchmarks still run on the CPU as the comparison point.

## References and parity

- The reference implementations define correct output.
  The multi-scale reference breaks the graph per level on `.item()`,
  so it serves only parity tests.
- The eager multi-scale tests check parity with the reference for both methods,
  dispatch through `MultiScaleDeformAttn2d` (`Backend.EAGER` and `backend=None` on the CPU),
  and `torch.compile(fullgraph=True)` tracing.
- The autotuned Triton configuration is compared against the cached one,
  the validated baseline, with gradient parity in float32 only.
  The bias test compares half precision against float64.
- Two backends can each match the reference with their own split
  yet disagree on the packed layout, which a cross-backend layout test catches.
  Only an unsupported-call refusal may skip a backend there,
  since a compile or launch failure is the defect under test.
- Library parity runs the bundled and JIT libraries in fresh processes,
  comparing full outputs and gradients with a tolerance for accumulation order only.
  The tests name the CUDA backend, since automatic selection could pick Triton.
- Hypothesis seeds tensor values, so a failing example replays and shrinks with its operands.
  Operands are generated in float64 and cast,
  because `torch.randn` gives different bits per dtype for one seed.
- CUDA and Triton accumulate sums in different orders,
  so 16-bit dtypes, float16 especially, diverge on large `L * K * G * D` workloads.
  Query-tiled parity tolerances catch gross mismatches only,
  and skip bfloat16 CUDA against input-dtype Triton gradients,
  which diverge on SM 8.0 (`test_msda_bias.py` covers that).
- Eager half-precision parity runs on the GPU;
  its gap is the sum of two independent rounding errors.

## Gradient checks

- Gradchecks for both attention operators cover both methods,
  since a softmax flag lost on the way into the backward
  gives plausible gradients that only finite differences expose.
- A zero forward passes gradcheck with zero gradients,
  so the attention projection gets small Gaussian noise
  and single-scale tests perturb zero-initialized projections.
- Bilinear sampling has kinks at integer coordinates and a discontinuity at the input edge,
  so gradcheck keeps samples off the boundary.
  Module-level tests draw reference points from `[0.25, 0.75]` and zero the offset bias,
  so each initial sample is its reference point.
  The feature maps are large enough that the default DETR-grid bias
  (offsets up to `num_points` pixels) also samples inside the image.
  Single-scale tests bias offsets a quarter cell inward, avoid padding and enlarge extents.
  A fixed seed keeps reduced-precision backends off the boundary and avoids flakes.
- Each gradcheck row covers one distinct kernel.
  `Backend.CUDA` names only the production configuration.
  Each Triton configuration uses its own backend object,
  imported lazily because a CPU host may lack Triton.
- Eager and production CUDA run in float64 with the tight tolerance,
  CUDA through the `Double` branch of `dispatch_scalar_type` in a JIT build.
  The CUDA extension receives the input tensor directly,
  and only its backward downcasts to float32, when asked,
  so a build that drops double breaks only these rows.
- Triton computes float64 operands in float32, where quantization sets the noise floor,
  so its rows use `eps=1e-3`, `atol=1e-2` and `nondet_tol` for `tl.atomic_add`'s order.
  `test_msda_triton.py` calls the kernel object, since dispatch refuses float64 for Triton.
- A (method, backend) pair with a broken backward goes in a strict xfail table,
  so a fix must remove its entry.
- The CUDA multi-scale test checks the attention gradient one step earlier,
  because the weights passed to the kernel are not a leaf.
  Both graphs run raw weights through softmax, the forward and a sum,
  one with the CUDA operator and one with the reference.

## Boundaries and robustness

- The eager boundary test maps locations to pixels as `pix = loc * 4 - 0.5`:

  | location | pixel |
  | -------- | ----- |
  | -0.125 | -1    |
  | 0      | -0.5  |
  | 0.125  | 0     |
  | 0.375  | 1     |
  | 0.5    | 1.5   |
  | 1.0    | 3.5   |
  | 1.25   | 4.5   |

  Pixels -1 and 4.5 are fully outside, and 3.5 is `W - 0.5`.
  It checks eager's two-sided zero-padding mask,
  which differs from the CUDA kernel's one-sided masking.
- Eager locations come from `[-0.1, 1.1]`, so some samples exercise zero padding.
- Samples at integer pixels, including -1 and the far edge,
  take eager's one-sided location gradient on every backend.
  Detection models initialize offsets to a grid, so their first step samples there.
- Non-finite locations, single-pixel levels and odd point counts are valid input.
  An infinite location adds nothing to the value gradient,
  so a non-finite value gradient means an out-of-range kernel write.
  Location and attention gradients of a query with infinite locations are undefined,
  as in the reference, so only other queries must be finite.
- Every device backend rejects structurally malformed operands with `ValueError` before launching.

## Validation and release tests

- A mismatched level table or geometry is an out-of-bounds device read
  (see [conventions](conventions.md#operand-validation)),
  so validation tests assert on messages that name the wrong operand.
- 2^30 elements per image fits in int32,
  but a batch of 4 overflows 2^31 - 1 in the kernel's pointer arithmetic.
  The tests build such operands with `expand`, at no memory cost.
- Under autocast, half-precision offsets added to float32 reference points
  give float32 locations with half value and weights,
  the ordinary Deformable DETR and Mask2Former call, which validation accepts.
- Selection policies depend only on the call context,
  so synthetic contexts test them on the CPU.
  A throwaway operator tests explicit choice, capability rejection,
  fallback with its warning, loader errors, memoization and `resolve`.
- The first multi-scale call on a CPU host imports every adapter and raises any adapter warning,
  so the warning test runs in a fresh interpreter.
- Release hardening tests check named errors for malformed calls on every backend,
  documented policy choices,
  actionable errors instead of broken libraries after build or cache failures,
  and that a delayed adapter registration never shows a second thread an empty registry.
- CUDA multi-scale batches larger than or indivisible by `chunk_step` match eager.
- Release regression tests check one-place method resolution,
  a linear functional default on every backend,
  the CUDA kernel running the operands' point count,
  and module patching that raises on no match and reaches nested replacements.
- Non-contiguous operands match contiguous copies in the forward and gradient writes.
  The tests slice locations and weights from packed tensors
  and transpose a value from `(B, G, S, D)` to `(B, S, G, D)`.
  A non-uniform upstream gradient keeps a stride mix-up from hiding.

### CUDA multi-scale adapters

- A factory builds each configuration's backend from one closure
  (name, supported contexts, both forward branches),
  which the tests run for every configuration.
- The adapter subclasses a protocol with abstract members,
  so a missing member raises `TypeError` at construction.
- The softmax branch passes raw weights for the kernel to normalize,
  and the linear branch passes them as given.
  Each is compared with the functional entry point called the same way.
- The CUDA forward's `points` selects the kernel template and defaults to 8.
  Tests pass the actual count (4 or 8), or the kernel reads past the locations and weights.

## Compile, export and fake tensors

- `torch.compile(fn)` runs through a graph break, so tests assert that none occurred.
  They use `backend="aot_eager"` (single-scale, convolution) or `backend="eager"` (eager multi-scale),
  because Inductor's C++ codegen needs a host toolchain CI may lack.
- Fake registrations under `deformops_triton` must match the real return arity, dtype and shape,
  or compiled multi-scale models fail at run time instead of trace time.
- Triton configurations register their operators when their module loads.
  The imports are wrapped for the Triton-less CPU CI; tests using the operators require CUDA and skip there.
- Only the softmax operator returns online-softmax statistics
  (a float32 `(B, Q, G)` pair), since only its backward reads them.
  `test_msda_triton.py` checks the linear operator's output shape.
- The cached wrappers drop `(m_run, z_run)`, so an order, dtype or shape regression
  would show only as wrong gradients, possibly passing gradcheck at small shapes.
  The tests check `exp(raw_attn - m_run) / z_run == softmax(raw_attn)` on the registered operator,
  catching swapped statistics, a wrong dtype and a missing kernel rescale.
- The fused-softmax operator, normalizing over the flattened `(L, K)` axis,
  is compared with a Python-side softmax fed to the linear operator.
- Cached softmax gradients must agree for 4D `(B, Q, G, L * K)` and 5D `(B, Q, G, L, K)` weights,
  catching a bridge `sum(dim=-1)` that spans only `K`.
- Zero queries give `(B, 0, ...)` outputs and a working backward;
  the bridge unpacks a 3-tuple there too,
  so a wrong early-return shape would break training that prunes every query.

## Optional dependencies

- After a torch mismatch, torchvision still imports,
  but `torchvision.ops` raises `RuntimeError` from its fake registration.
  Tests catch every exception and probe one valid operator
  (a 3x3 weight needs `2 * 9` offset channels).
- transformers reports that mismatch as an `ImportError` of the model class.

## README examples

- `test_readme.py` runs the README's `python` and `py` blocks in order, in one namespace,
  so a moved module or renamed function fails the suite.
- On the CPU, `.cuda()` calls are dropped;
  kernel-selection snippets construct modules without a forward.
- Registry `load` calls (`msda.load`, `attn.load`, `conv.load`), with a trailing comment,
  are bound against the real signature instead of run,
  with arguments still evaluated, so renamed names or keywords fail.
- Snippets that download weights or need a missing extra are compiled and reported as skips.
- Extras are detected by importing a model class,
  because transformers imports lazily and only a model class reaches torchvision.
