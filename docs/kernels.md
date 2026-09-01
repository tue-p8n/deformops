# Kernel design

How the Triton and CUDA kernels are laid out, launched and specialized,
and what each experimental configuration changes.
Precision rules, including the float32 accumulator, are in [precision](precision.md).

The eager multi-scale gather materializes a `[B, Q, G, L, K, D]` intermediate
whose memory grows too fast at encoder scale,
so eager is the fallback, not a production path.
It is pure PyTorch, so it runs and is tested on CPU.

## Triton backend

### Layout

- The Triton backend needs no CUDA toolchain and no downloaded library.
  Triton compiles the kernels on first use on a CUDA device,
  for float16, bfloat16 and float32.
- The `msda`, `attn` and `conv` adapters register on import,
  and only where Triton is importable and a CUDA device is available.
  The package imports them lazily.
  Accessing one without Triton raises `ImportError` naming the missing dependency.
- Each `msda_*` module is a multi-scale configuration built on `_msda_bridge`,
  imported by its full module path.
  It exports `forward` and a `Backend` class that `MultiScaleDeformAttn2d` accepts as `backend`.
  The `msda` adapter runs `msda_query_tiled_fp32acc` or `msda_query_tiled`,
  as the selection policy asks.
- `attn_query_tiled` and `attn_query_tiled_fp32acc` build on `_attn_bridge`.
  `conv_deform2d` holds deformable convolution.
- The `@triton.jit` kernels and their launch helpers live in `_kernels`.
  They register nothing and do not import `torch.library`,
  so a configuration module only names its launch.
- The float32 accumulator is a separate configuration module, not an option.
  The selection policy picks it per architecture.
  An explicit Triton call gets it, which is correct on every architecture,
  unless it passes `fp32acc=False` for native half-precision atomics.
- Triton operators use the vmap fold from `registry/_vmap.py`,
  so a private operator batches like the public operator that selects it.

### Launch limits and empty calls

- Kernels compute a sample's base pointer as `batch_index * stride`.
  Triton types a stride that fits in 32 bits as int32,
  so the product wraps once batch times per-sample elements passes 2^31.
- The batch is the launch grid's third axis, which holds at most 65535.
- vmap folds its axis into the batch, so both limits are reachable.
  The launch code for every operator, backward and vmap rule splits larger calls along the batch.
  A single sample too large for int32 addressing raises `ValueError`.
- A zero batch, query, level or point count returns zeros without launching.
  A launch would have a zero-sized grid,
  and the normalizing kernels would divide by an empty sum.
- Launch arguments after unpacked stride tuples are passed by keyword.
  Tools that cannot see the tuple lengths read a positional tail as an argument-count mismatch.

### Relaxed atomics

Default acquire-release semantics make every atomic accumulation a synchronization point.
Relaxed memory ordering lets the compiler elide the L1 coherence-sector traffic
those semantics require around each transaction.
Every atomic accumulation in the Triton kernels uses relaxed ordering,
except in the experimental point-parallel kernel.
This is sound because the accumulations commute,
write disjoint gradient storage,
and have no reader until the kernel finishes.

### Multi-scale query-tiled kernel

- One forward kernel serves every configuration, which vary only its launch grid.
  Each program processes a block of queries (32 by default),
  which gains the data reuse the point-parallel kernel gives up.
- A wider query block with more warps raises occupancy but enlarges the per-program working set.
  The payoff depends on the sampled shape, so it is a configuration, not a heuristic.
  The default launch leaves `num_warps` and `num_stages` to Triton.
- The calling operator fixes `aggregate`, so its branch folds away.
- The backward launch shape lives with the backward kernels.
  Two backward launches share a kernel.
  The default launch accumulates the value gradient in the input dtype;
  the float32 launch uses a float32 scratch buffer at twice that buffer's memory.
- The backward is the linear-attention backward over the weights it receives.
  The caller handles dtype casts and the final `flatten(2)`.

#### Online softmax

- The softmax is a Welford-style running (max, sum) over (level, point).
  A new maximum rescales the accumulator and the running sum.
  One sweep over the attention and location tensors builds the denominator
  and the value-weighted accumulator,
  consistent with the stable shifted exponent.
- Softmax masks out-of-range queries with `-inf`, so they add nothing to the denominator.
  The linear path loads the raw value with `other=0.0`.
- Both shifts check for `-inf - -inf`.
  Otherwise a masked logit seen while the running maximum is `-inf` turns the query into NaN.
  An unchanged maximum rescales by one, and a masked logit adds nothing.
- Under the query mask, the running sum is positive whenever any (level, point) was processed,
  so the final division is safe.

#### Softmax on the host

- The bridge computes softmax probabilities in float32 for every input dtype.
  They feed the kernel weights and the Jacobian correction,
  a difference of similar magnitudes that loses its leading digits in 16-bit.
- The bridge passes normalized weights to the kernel.
  The value and location gradients are then correct as returned,
  and the weight gradient is dL/dp.
  The Jacobian turns it into dL/da:
  dL/da_j = p_j (dL/dp_j − Σ_i p_i dL/dp_i).
- The Jacobian runs on the host.
  It reduces over `L * K` on a tensor far smaller than the value gradient,
  and in the kernel it would need all `L * K` values live at once.
- The backward reshapes `grad_output` with `reshape`, not `view`.
  Upstream operations can return a non-contiguous gradient that `view` rejects.

### Single-scale sampling kernel

- Warp counts follow from a fixed number of output-tile elements per warp,
  because the tile grows with the channel count
  and a fixed warp count suits only one channel count.
  The forward uses 512 elements per warp.
  The backward uses 64, since scattered accumulation into the input gradient limits it
  and many warps doing little arithmetic each serve it best.
- The host builds the grid displacement tables.
  The ordering function is then the only definition of the point order,
  and the kernel needs neither kernel extents nor dilation as compile-time constants.
- Tables are cached by geometry and device, their only inputs,
  and callers repeat the same geometry every step.
  A per-call table would be a pageable host-to-device copy on every forward and backward,
  which a CUDA-graph capture cannot record.
  The kernel only reads tables, so sharing them is safe.
- Bilinear interpolation is inlined, because Triton only admits calls to jitted functions.
- Value loads have no eviction hint.
  Adjacent output positions share sampling neighborhoods,
  but on Ampere keeping these loads in cache slows the kernel.
- The forward output is allocated with `empty`.
  The grid `cdiv(positions, BLOCK_SIZE_Q) × groups × batch`, with its query and channel masks,
  writes every element once.
- In the backward, only the input gradient scatters, because output positions read overlapping pixels.
  It accumulates atomically.
  Each offset and mask gradient belongs to one output position
  and is a direct store after a reduction over channels.
- The packed offsets are a view of `offset_mask` sliced to its first used columns.
  A wider `offset_mask`, for example padded to a multiple of 8, makes the view non-contiguous,
  with a per-position stride of the wider row.
  The kernel addresses the offsets and their gradient with the same stride,
  so the gradient must share that stride pattern.
  `torch.zeros_like` on the view would allocate a contiguous tensor that the kernel overruns,
  so the gradient is sliced from a zeroed `offset_mask` the same way.

### Deformable convolution kernel

#### Structure

- The kernel covers classic deformable convolution, v1 and v2, channels-last,
  with torchvision's conventions (see [conventions](conventions.md)).
- The kernel contracts sampled values against the weight directly,
  instead of deformable im2col followed by GEMM.
  No `(B, P, C_in, K)` buffer exists, so memory is the operands plus the output.
- The kernel blocks over output positions and output channels
  and reduces over `K * C_in / groups` with `tl.dot`.
- Convolution groups partition input and output channels together.
  Offset groups partition input channels only, choosing each channel's offsets.
  The two are independent, so the reduction walks their intersection.
- Launch grid axis 1 encodes the convolution group and the output-channel block within it,
  so a block never crosses a group boundary.
  The kernel derives the split, because the autotuned block width is unknown at launch.
- The host permutes the small weight to `(K, C_in / groups, C_out)`,
  so the contraction reads a `[BLOCK_K, BLOCK_N]` tile directly.

#### Autotuning

- The tile that fits depends on a device's per-program shared memory:
  roughly 100 KB on SM 8.6, 164 KB on an A100 and 228 KB on an H100.
  GEMM-preferred configurations do not fit the smaller budgets.
  Triton discards configurations that fail to build and times the rest,
  so one source picks a tile per budget.
- The configuration space is small, because autotuning recompiles all of it per distinct key.
  A large space makes a first call take minutes.
  `num_stages` is not varied: the point loop is unrolled and has nothing to pipeline.
- Keying on the position count `H_out * W_out` would re-time the space for every input size.
  Once the count exceeds a tile, the winner no longer depends on it,
  so the key rounds the count up to a power of two.
  Small maps, where the choice matters, keep separate entries.
  A map below the largest tile can get an overshooting tile,
  which costs less than re-timing because such a map cannot saturate the device.
- The tuning key holds the channel counts, which decide viable tile shapes,
  and the bucketed position count.
  Forward and backward share one key list, although each takes only some of its names.
  Triton's autotuner skips names missing from a call's arguments, so the union is safe.
- The compile-time flags are in the key,
  because each flag combination is a separate body with its own live-value and register footprint.
  A full-gradient backward keeps far more live across a tile than a frozen-weight one.
  Without the flags, the expensive specialization could reuse the cheaper one's configuration
  and exceed its register or shared-memory budget outside any exception handling.
- `cache_results` stores the winner in Triton's cache directory,
  keyed on kernel source, Triton version and device target.
  Without it every process repeats the minutes-long sweep.
- The backward needs `reset_to_zero`.
  Autotuning runs the kernel repeatedly, and every backward output accumulates into a zeroed buffer.
  Without the reset, the gradients would sum every timing run.

#### Backward

- Every backward output accumulates across programs with relaxed atomics.
  The input gradient does because output positions share pixels.
  The weight gradient reduces over all positions,
  and the offset and mask gradients reduce over output channels in different programs.
- The `NEEDS_*` flags are compile-time constants, so unrequested outputs cost nothing.
  Frozen weights are common in fine-tuning, and the weight gradient is the expensive output:
  a second `tl.dot` per channel tile plus an atomic over the whole weight.
  An absent mask cannot need a gradient,
  so `NEEDS_MASK_GRAD` implies `HAS_MASK` and the kernel tests it alone.
- The interpolated value feeds the weight and mask gradients.
  Each sample's incoming gradient feeds the input, offset and mask gradients.
  Each block runs only when an output that reads it is requested.
- The four corner values feed the interpolated value and its coordinate slopes (offset gradient).
  An input gradient alone needs neither and skips four loads per channel tile.
- The corners are bound before the branch as unannotated floats.
  Triton resolves an annotation inside a jit function as a global name,
  and a float instead of `None` keeps the arithmetic off optionals.
  Every use sits under the constant condition that loads the corners, so the placeholder is never read.
  They are four statements because Triton rejects multiple assignment.
- The host computes the bias gradient as a reduction over `grad_output`,
  and nothing launches when it is the only request.
  It is cast to the bias dtype,
  since autograd needs a gradient in its tensor's dtype
  and the bias width may differ from the weight's.
- The permuted weight's `.contiguous()` copy runs after the early return,
  so a frozen-weight call skips it.
- Placeholders for unused pointers, such as unrequested gradients or an absent mask,
  share one dtype so the kernel does not recompile per placeholder type.

#### Operator module

- Convolution is one module.
  Its seven specialization switches come from runtime state,
  four of them from the gradients autograd requests per call,
  so there is no configuration to name a file after.
- It registers on import as a `torch.library` Triton operator with autograd;
  the registry imports it on first dispatch.
  `torch.export` keeps it as one node,
  and `torch.compile` and AOTInductor compile its kernels into the graph.
- Under vmap, a batched weight or other operator-defining argument raises.
  Folding it into the batch would contract against the wrong filters without error,
  and a Python loop would cost a launch per element.
- Operator schemas have no `Tensor?` return type,
  so an unrequested output returns as a zero-element placeholder.
  The autograd bridge, which knows the request flags, restores `None`.
  Placeholders are separate allocations, because custom operator returns must not alias.
- `setup_context` reads the requested gradients from the named operands,
  independent of how `register_autograd` flattens them,
  not from `ctx.needs_input_grad` in the backward.
  A `None` bias dtype covers both no bias and a frozen bias.
- `supports()` rejects non-CUDA devices and float64, which Triton has no path for,
  so dispatch falls back to eager instead of failing late.

## CUDA backend

### Configurations

- Single-scale sampling and deformable convolution have no C++ or CUDA kernel.
  `Backend.CUDA` aliases their Triton kernels, so selecting CUDA still gets a device kernel.
  The alias registers only where Triton is importable and a CUDA device exists.
  Otherwise an explicit `Backend.CUDA` raises `NoBackendError` naming the missing dependency.
- Each multi-scale configuration has its own module, translation unit and extension.
  They share the split-layout kernels through `include/msda_boxed.cuh`.
  A translation unit only chooses the backward traits and the operator names.
  Each `.cpp` is a build stub, because the extension build pairs a `.cpp` with each `.cu`.
  Python registers the schemas.
- Configurations differ in their operator prefix, their backend name,
  and whether the backward upcasts its inputs to float32.
  `install_msda_family` builds schema registration, the deferred extension load,
  the functional entry point, the autograd and fake-tensor hooks, and the dispatch adapter.
- Every configuration compiles the same trait-free forward kernel,
  differing only in whether the in-kernel softmax runs.

| Configuration | Value-gradient accumulation | Softmax reduction |
| --- | --- | --- |
| `msda_fp32acc` (what `Backend.CUDA` runs) | Inputs upcast to float32; a float32 scratch gradient, cast back at the tail | Warp shuffles |
| `msda_serialreduce` | As `msda_fp32acc` | Serial |
| `msda_half2atomic` | Native dtype; 16-bit channels paired into one `half2` atomic per pair | Warp shuffles |
| `msda_half2atomic_serialreduce` | As `msda_half2atomic` | Serial |
| `msda_scalaratomic` | Native dtype; one atomic per element | Warp shuffles |
| `msda_scalaratomic_serialreduce` | As `msda_scalaratomic` | Serial |

- The slower scalar-atomic configurations separate the effect of `half2` packing
  from that of the native dtype, since the float32 accumulator changes both.
- Warp-shuffle reduction reduces the softmax across a head's warp lanes with shuffles.
  Serial reduction reduces it on thread 0 of each head.
  Only the schedule differs.
- Only the float32-accumulator configurations upcast the backward inputs
  (value, locations, attention weights and `grad_output`).
  The others keep their dtype, so the native 16-bit atomic paths are reachable.
  Upcasting everywhere would force `scalar_t` to `float`
  and make every accumulation trait bit-identical.
- The 16-bit atomic accumulators compile from SM 7.0 for float16 and SM 8.0 for bfloat16.
  Below that the accumulator body is empty and the value gradient would be zeros,
  so the non-upcasting configurations raise.

### Bridge

- `setup_context` saves a zero launch geometry.
  The backward then derives its layout from concrete sizes,
  and tracing never specializes on batch or query count.
  Each pass has its own layout heuristic in `core/tuning.py`.
- The backward runs with autocast disabled
  and casts each gradient to its saved tensor's dtype.
  The gradient tuple is padded with `None` to the forward operator's nine inputs,
  with value, locations and attention weights at positions 0, 3 and 4.
- The kernels index raw pointers with shape-derived arithmetic and never read strides,
  so non-contiguous operands would read the wrong elements in both passes.
  The entry point makes the operands contiguous,
  and the autograd hook saves what the operator receives.
- The launch reads level tables through an `int64_t` pointer,
  so int32 tables would read as wrong extents.
- The launch picks the batch chunk from concrete sizes,
  and the thread layout too when either geometry argument is zero.
- `supports_context` checks the launch constraints so dispatch falls back instead of raising.
  It requires a CUDA device,
  4 or 8 points and 1 to 5 levels (the instantiated templates),
  and per-head channels divisible by 8.
  The smallest block must fit the 1024-thread limit in the forward
  and, when training, in the backward.
- `points` selects the kernel template and is read from the location tensor.
  A mismatched template reads 8-point operands as 4-point ones without raising.

### Launch layout

- `forward_thread_layout` and `backward_thread_layout` in `core/tuning.py`
  pick `(channels_per_thread, threads_per_block)`,
  with the query count Q as the per-batch work count.
  The layout is a fixed function of the shapes,
  so the CUDA kernels need no runtime autotuning and pay no startup cost.
  The Triton backends autotune their own launch parameters.
- Forward: `channels_per_thread = 8`,
  for vectorized 128-bit memory accesses across 8 channels
  (for example `float4` or `half2` loads when D is 32 or 64) and high memory bandwidth.
  `threads_per_block` is at most 512,
  from the largest factor `queries_per_block <= 64` of `B * Q` that fits,
  trading streaming-multiprocessor occupancy against register usage.
- Backward: `channels_per_thread` is 2 when D >= 64 and 1 otherwise.
  The backward computes three gradients at once and uses far more registers,
  so a smaller channel slice avoids register spilling.
  `threads_per_block` is at most 256 for the same reason.
- Both caps are targets a shape may exceed.
  `queries_per_block` starts at one and only grows,
  so a shape whose `G * D / channels_per_thread` already exceeds the cap
  gets a larger `threads_per_block`.
  G = 8 and D = 128 yield 512 from the backward heuristic,
  and the launch bounds are sized for that.
- Sizes are converted to `int` before the memoized layout.
  A traced size is a `SymInt`, which does not hash.
  A launch needs a concrete layout anyway,
  so the traced graph specializes on it and any caller may pass a traced size.
- The block axes are the channel lane in x (`D / channels_per_thread` lanes per head),
  the head in y (`G`),
  and the query tile in z (`queries_per_block` packed (batch, query) units).
- Every launch site gets its geometry from `compute_block_geometry`, which checks every divisor.
  A zero head or channel count, or a `threads_per_block` too small for one query,
  yields `queries_per_block == 0`.
  Each launch site rejects that with a message instead of a process-killing division.
- The driver launches `block_thread_count` threads,
  `threads_per_block` rounded down to whole queries,
  and the launch bound must cover it.
- Both launchers check CUDA's ceiling `kMaxThreadsPerBlock = 1024`,
  so an oversized block reports a clear error
  instead of the driver's "invalid configuration argument".
- The grid size uses truncating division,
  so a zero or non-dividing `queries_per_block` would drop tail queries without error,
  leaving the allocation's zeros.
- A rejected launch leaves outputs and gradients at their allocated zeros,
  so the launch reads back the error before an unrelated CUDA call reports it.

### Headers and compile-time dispatch

- `msda_args.cuh` holds the shared kernel's include preamble.
  `stable_compat.cuh` redefines `TORCH_CHECK` and `CHECK_DIVISIBLE` to raise `std::runtime_error`,
  so the definition a use site gets depends on include order.
  Other kernel headers include CUDA, torch and utility headers only through this file,
  which fixes the order in one place.
- Type aliases live in the `msda_split` namespace, not the global namespace.
- `accum_t<scalar_t>` is float32 for half precision and the operand dtype otherwise.
  It is an alias template, so it obeys scope and works as a type argument.
- Locations are read at the accumulation width.
  A pixel coordinate rounded to 16 bits can fall into the neighboring bilinear cell.
- `dispatch_constant` turns a runtime value into a compile-time template argument,
  so the loops over levels, points per level and channels per thread unroll.
  It calls `fn` (`[&]<int64_t V>() { ... }`) with the matching `V` and returns whether one matched,
  so an unsupported value returns `false` instead of running another instantiation.
- Each supported level count L costs an instantiation.
  The `dispatch_constant<1, 2, 3, 4, 5>` packs define the supported depths.
- `TORCH_CHECK` arguments are bound to variables first,
  because commas in template argument lists would split the macro arguments.
  Without the K check, an unsupported K launches nothing
  and the caller reads the allocation's zeros as results.
- `channel_vector_t_for` picks the widest vector type per precision,
  so float16, bfloat16 and float32 share dispatch rungs.
- Each `channels_per_thread` pack must equal the values the tuning heuristic emits at that site.
  An unreachable value is dead code,
  and a missing reachable value makes a valid shape fail a `TORCH_CHECK`.
  `tests/deformops/_internal/test_dispatch_packs.py` compares
  the forward pack in `forward_launch_inner` and the backward pack in `backward_launch_inner`
  with each site's own reachable set, in both directions.
  A check against the union would miss a one-sided narrowing,
  such as the backward going from `<1, 2>` to `<2>`.
- Kernels copy `channels_per_thread` consecutive `scalar_t` elements in one load or store
  through a small vector type such as `uint4` or `ulonglong4`.
  This is valid only when `sizeof(channel_vector_t) == sizeof(scalar_t) * channels_per_thread`.
  `channel_vector_type` selects the widest vector type per byte width,
  covering 2-byte (float16, bfloat16), 4-byte (float32) and 8-byte (float64) channels.
  An uncovered width, such as float64 with an odd channel count, maps to `scalar_t`,
  and `vectorized_copy` falls back to an unrolled scalar copy
  instead of copying a partial range.

### Indexing and sampling

- Per-point offsets use 32-bit index arithmetic.
  The host rejects a value tensor whose within-level flat offset, bounded by `S * G * D`,
  could exceed `INT32_MAX`.
  The call site adds the per-level base pointer in 64-bit, since `value` already points at the level.
  The hot offset math then avoids emulated 64-bit operations,
  and scatter addresses are `grad_value + int`.
  Base-pointer and query bookkeeping (B, G, D, Q, batch and query indices, level starts) is 64-bit.
- Bilinear sampling relies on the caller's guarantee `-1 <= pixel < extent`.
  Each axis can then leave bounds on one side only,
  the near corner below zero or the far corner at the extent,
  which needs one comparison per corner.
- Corners are visited near to far on both axes, which fixes the summation order.
  A corner outside the level loads as zero, the padding the reference defines,
  so the weight-gradient accumulation runs unconditionally.

### Forward kernel

- The sampled location is the same for every lane of a head,
  so lane 0 reads it and broadcasts it with `__shfl`.
  This requires a head to tile a warp evenly.
  Otherwise `get_lane_mask` names lanes outside the head
  and shifts by the register width once a head needs more than a warp.
  `__shfl_sync` on a mask excluding the calling thread is undefined.
  Every lane then reads the location itself, as in the backward's block-wide reduction.
- The broadcast flag is uniform across the block,
  so the branch never splits a warp.
  Lanes a shuffle overwrites are zeroed first, so none reads garbage.
- The normalizing path keeps two scalars:
  the maximum logit and the sum of exponentials shifted by it.
  An `L * K` array would hold forty registers per thread at L = 5, K = 8.
- The maximum is taken over the whole block first.
  A running maximum inside the sampling loop would compute `-inf - -inf`
  while the prefix is fully masked, and NaN from there on.
- The weight enters the softmax sum before the bounds test,
  so a point outside its level still counts in the denominator.
- The launch bound has two tiers, sized to the block the launch needs,
  so the common case gets a tight register allocation and wider blocks still launch.
  The bound only has to be an upper bound, and each tier duplicates every instantiation.
  Typical forward launches use the tighter tier, which makes the float32 forward faster.
  Neither tier causes register spilling.

### Backward kernel

- The backward launch bound is the hardware maximum.
  A tighter bound does not speed up the backward,
  and a second tier would double the larger kernel's instantiations.
  A bound of 256 would reject G = 8, D = 128, which needs 512.
- `check_backward_warp` picks one of two reductions and the matching shared-memory budget.
- Warp path: a head's lane count `D / channels_per_thread` is at most the warp size of 32 and divides it.
  The location is broadcast and the three sampling-parameter gradients are reduced with warp shuffles.
  Every lane calls the shuffle collectively, never behind a divergent branch,
  including out-of-bounds lanes holding the pre-zeroed 0.
  This path drops the per-lane tile and keeps the two attention-weight tiles.
  Warp-shuffle bookkeeping is valid only on this path.
- Shared-reduction path: a head spans more than one warp or tiles it unevenly,
  for example D = 128 with 2 channels per thread gives 64 lanes.
  In this layout a wrong lane count crashes or corrupts the gradients.
  The location goes through a shared-memory slot,
  and the gradients through a shared-memory tile with block barriers all threads reach.
  The per-head bounds test skips only the work and the final write,
  because skipping a barrier would make `__syncthreads` divergent.
  Thread 0 of each head sums the per-lane partial gradients
  of (location x, location y, attention weight) from `shared_grad_sampling_parameters`.
  `head_slot = threadIdx.z * G + threadIdx.y` selects the head's stride-3 tile of `blockDim.x` entries.
- Each lane normalizes its own stride of the tile after the last reduction shuffle,
  before reading entries its siblings wrote.

#### Shared-memory partition

- The device offsets and the host total read the extents from one struct.
  The grouping and integer-division order of each extent must match on both sides.
- The partition holds two attention-shaped tiles, one tile apart:
  the pre-softmax weight gradients and the softmaxed weights the sampling loop reads.
  The shared-reduction path adds the per-lane gradients and the per-head location slot.
- `channels_per_thread` is a template parameter, so dividing by it avoids a 64-bit division.
- The softmax backward stores each point's pre-softmax attention-weight gradient
  in `shared_grad_attention_weight_before_softmax`,
  then runs the softmax-backward correction over all `L * K` slots.
  Only in-bounds points write their slots, so the kernels zero each head's slots first.
  An uninitialized slot would give nondeterministic NaN or garbage in the attention-weight gradient
  and, through the softmax, in the other gradients.
  Sanitizers do not catch uninitialized shared memory; see [testing](testing.md).

#### Value-gradient accumulation

The accumulation strategies share one body, selected by `grad_value`'s element type.

- The float32 accumulator `accum_t<scalar_t> *` uses per-element `atomicAdd`:
  higher precision for 16-bit inputs, at the cost of a scratch tensor and a cast back.
- Native `scalar_t *` with a 16-bit dtype uses packed `half2` atomics,
  which use less memory and run faster at native precision.
  It falls back per element where a pair is not aligned.
- The unpacked native form needs `atomicAdd_scalar`,
  since `atomicAdd` has no 16-bit pointer overload.
- For float32, all forms reduce to `atomicAdd`.
- `allow_half2_atomics` matters only with `native_grad_value` and a 16-bit dtype.

### Host entry

`msda_launch.cuh` chunks the batch, allocates outputs and gradients, and dispatches on scalar type.

- The implementation takes `apply_softmax` as a runtime int,
  set to 0 (linear) or 1 (softmax) by the registered operators.
- Batch chunks are the largest divisor of `batch` that is at most `chunk_step`.
- Locations must be float32, or float64 with a float64 value tensor.
- The value gradient is native `scalar_t`,
  or a float32 scratch tensor cast back at the tail when `native_grad_value=false`.
- The location and attention-weight gradients are shaped like the flat or expanded inputs.
  They are written in `accum_t<scalar_t>` (float32 for Float, Half and BFloat16, float64 for Double),
  then cast back to the input dtypes.

## Experimental configurations

`registry.Backend` cannot select the experimental configurations.
Pass a module's `Backend` instance to `MultiScaleDeformAttn2d`, or call its `forward`.
Each module exports `forward` and `Backend`; importing a Triton one needs Triton.
The production Triton configurations appear first, for comparison.

### Triton multi-scale configurations

- `msda_query_tiled` is the production Triton choice on Hopper and later,
  whose hardware has a native half-precision atomic add.
  The forward normalizes in-kernel with an online softmax when asked.
- The `*_fp32acc` variants (query-tiled, cached, autotuned, and single-scale `attn_query_tiled`)
  accumulate the value or input gradient in a float32 scratch buffer.
  Below SM 9.0, a 16-bit atomic add runs as a relaxed-ordering compare-and-swap loop
  that is slow and biases the gradient in reduced-precision training.
  These variants avoid it at twice that buffer's memory,
  and are the production choice on Ampere and Ada.
- The "wide" variants run the forward at four times the query block,
  with twice the warps and one more pipeline stage;
  kernel and algorithm are unchanged.
- `msda_atomic_wide` runs the query-tiled backward unmodified,
  one atomic per contribution, and lets the hardware serialize colliding atomics,
  with no per-block working set and no sort.
- `msda_point_parallel` launches one program per (query, point, head, batch):
  high occupancy, low bandwidth, no data reuse across points.
  Its forward atomic additions run in varying order,
  so repeated calls can differ in the least significant bits.
- `msda_query_tiled_cached` also writes the online softmax's running maximum and sum of exponentials.
  The softmax backward rebuilds the weights from them with one elementwise `exp(a - m) / z`,
  with no maximum scan or sum.
  That saving is the only reason to write them, so the backward must use that path.
  The statistics pass through the operator schema
  and cost two `(B, Q, G)` float32 tensors saved for the backward.
  They are valid only for `aggregate == 1`.
  The linear path writes no statistics but still passes pointers, since their strides are in the launch signature;
  it uses an uninitialized `(1, 1, 1)` dummy
  to avoid a `2 * B * Q * G * 4` byte allocation.
  The backward caller handles dtype casts and the `grad_output.view(B, Q, G, D)` reshape.
- `msda_query_tiled_autotuned` is the cached configuration
  with the query block size autotuned instead of fixed at 32.
  It targets Hopper, whose 228 KB of shared memory per streaming multiprocessor
  (against an A100's 164 KB) a fixed block may under-use.
  The first forward launch sweeps the configurations
  for each new query count, head dimension, level count and point count,
  and caches the result under `~/.triton/cache`.
  Value loads get an evict-last hint in both directions,
  because adjacent queries in raster order sample nearby positions.
  The launch grid is a lambda so the autotuner can inject `BLOCK_SIZE_Q`;
  `BLOCK_SIZE_D` is passed explicitly, since `next_power_of_2(D)` is its own specialization axis.
  Two backward launches share a kernel and differ in the value-gradient accumulator dtype.
- The `block_reduce`, `warp_reduce` and `sort_segscan` configurations, and their wide variants,
  change how the backward combines concurrent updates to one address;
  see [Reduce by key](#reduce-by-key).

### Point-parallel kernel

- Each query-point loops over the levels, usually only 4 to 5.
- The softmax path takes two passes:
  collect all `L * K` weights of a (query, head, batch), apply the softmax, then accumulate.
- The forward uses an atomic add because several points write the same output query-head.
- The backward computes the per-level bilinear weights and the four corner reads once
  for all three gradients (attention, value, location).

### Reduce by key

The kernels in `msda_reduce` compute the same gradient with fewer contended atomics
by reducing colliding addresses inside the program.
Each is slower than the scattered-atomic backward of `msda_query_tiled`.

- Block- and warp-scoped deduplication are the same kernel at two launch shapes,
  and the query block sets the scope.
  Threads colliding on an address merge only within one program,
  and for the warp form, within one warp.
  - `msda_block_reduce` reduces across a 64-query program with 4 warps:
    the widest scope, at the register cost of a 64-query working set.
  - `msda_warp_reduce` uses a single-warp 32-query program:
    a smaller scope with lower register pressure.
- For the value gradient, the scattered kernel issues four contended atomic stores per query.
  The reduction folds same-address contributions from other queries into one representative row,
  so each corner issues at most one atomic per unique address.
  Each corner materializes a `[Q, Q, D]` `tl.where` intermediate before `tl.sum` along axis 1,
  the main register cost at `BLOCK_SIZE_Q = 64`.
  A `tl.dot` of the redirect matrix with the contributions,
  or a per-source `tl.static_range` loop,
  would lower the peak to `[Q, D]`.
  The attention and location gradients are per-query writes without contention.
- `msda_sort_segscan` sorts (address, value) pairs, so each distinct address takes one atomic.
  Deduplication is complete within the program, at the cost of the sort.
  - The key packs the address into the upper 32 bits, so the sort orders by address first,
    and the block-local query index into the lower 32.
  - The sort runs over signed int64,
    so masked queries take a large positive sentinel and sort into a trailing, skipped segment.
  - The sentinel 2^30 − 1 must exceed the largest address offset
    (a cell index within one sample's feature pyramid) and keep the key non-negative.
    The bridge rejects samples with more than 2^31 elements, heads and channels included,
    so whenever `heads * channels >= 2` the pyramid has fewer than 2^30 cells.
  - Equal keys are contiguous after sorting.
    Each segment's first row holds the segment sum and issues its atomic.
- The reduce kernels keep sampled values in the source dtype, like the query-tiled kernel,
  and Triton promotes the later float32 and float16 multiplications.
  Following the query-tiled casting order,
  float16 and bfloat16 locations reach the same corner indices and weights as the reference backend.
  Upcasting before the scale and shift would change rounding and break bit-exact forward parity.

## Query ordering

The experimental CUDA configurations `msda_morton` and `msda_hilbert` reorder queries
to improve cache locality.
The permutation applies to the sampling parameters and is undone on the output,
so no per-query result changes.

- The reordering is host-side.
  These configurations have no translation unit and wrap the production kernel.
  `_query_order` is tensor arithmetic with no CUDA dependency, apart from its two dispatch adapters.
- The per-query key is the sampling location averaged over head, level and point.
  `Tensor.sort` sorts each batch row independently.
- The ordering function clamps locations to [0, 1] first.
  Sampling locations often fall outside [0, 1],
  and a negative coordinate becomes a garbage two's-complement index under bitwise operations.
  Clamping changes only the locality of out-of-bounds points.
- Morton (Z-order) indices are cheap but jump far at every power-of-two boundary.
  Hilbert indices, built by iterative quadrant rotation,
  keep consecutive indices on adjacent grid cells (Chebyshev distance 1)
  at the same asymptotic cost.
