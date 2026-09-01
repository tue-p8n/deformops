# Kernel loading and builds

How the CUDA backend obtains, builds, caches and verifies its kernel libraries.
User-facing summary:
[Kernel loading](../README.md#kernel-loading)
and [Environment variables](../README.md#environment-variables).

## Where a library comes from

`define_library` in `sources/deformops/core/build.py` uses the first source that has the library:

1. a library bundled in `deformops.core._ext`;
2. a library already compiled on this machine for the same sources and environment;
3. a download from the prebuilt index, if `DEFORMOPS_DOWNLOAD` enables it (off by default);
4. a local compile, if `DEFORMOPS_COMPILE` allows it (off by default).

| `DEFORMOPS_COMPILE` | Behavior |
|---|---|
| unset, `0`, `never` | Never compile; a library must be bundled, already compiled, or downloaded. |
| `1`, `fallback` | Compile when no bundled, compiled or downloaded library is available. |
| `always` | Skip the bundled and downloadable libraries and use a library compiled on this machine, compiling it if needed. |
| `rebuild` | Like `always`, after clearing this build's directory. |

A `KernelUnavailableError` subclass names the failed step (download, disabled compile, compile).

Only the multi-scale operator has CUDA extensions.
Sampling and convolution run Triton kernels,
which Triton compiles from the installed package on first use of each shape,
with no toolchain, per-CUDA build or library path.

### Lazy binding

Each CUDA configuration module (`msda_fp32acc.py`, `msda_half2atomic.py`, and so on)
wraps `define_library` in `lazy_extension`:
a memoized, thread-safe callable, exported as `load`,
that `forward` calls before its first launch.
Importing a module loads and compiles nothing.

- Schema definition, op-handle lookup (`load_op`)
  and meta and autograd registration (`register_op_pair`)
  need only the Python schema, so they run at import time.
- The first call's `download` and `compile` keywords reach `define_library`;
  later calls return the loaded library and ignore theirs.
- A failed load is not memoized, so a later call with other keywords can succeed.
  A successful load lasts for the process.

### Loading before the first call

`deformops.registry.msda.load` calls the same `load` closure as `forward`,
so an early load is never repeated.

- For CUDA multi-scale attention it obtains the library.
- For Triton and eager it registers the operators
  and checks that the backend can run on this host.
- `download` and `compile` apply to the first load only;
  `None` reads `DEFORMOPS_DOWNLOAD` and `DEFORMOPS_COMPILE`.
- The Morton and Hilbert query-reordering configurations
  wrap the production kernel and have no library of their own.

### Schema registration

`backends.cuda._schema.define_schemas()` registers every multi-scale schema
through `torch.library.define` before any library loads,
so the `torch.ops.deformops_cuda.*` handles exist for `register_autograd`.
The C++ translation units register implementations only.
Six configurations share the split-layout kernel,
each with four operators under its prefix:
a linear forward and backward pair and an in-kernel-softmax pair.

`load_op` handles one extension registering several operator pairs.
`register_op_pair` follows three rules.

1. `setup_context` unpacks the forward operator's positional inputs.
2. A backward disables autocast with `torch.autocast("cuda", enabled=False)`,
   so the kernel runs at the saved-tensor dtypes,
   and casts with `.type_as(saved_tensor)`.
3. The gradient tuple has one entry per positional input of the forward operator.

### Loading a prebuilt library file

A stable-ABI `.so` has no `PyInit_*` symbol, so `importlib.import_module` would fail.
The loader finds the file with `importlib.resources`
and loads it with `torch.ops.load_library`,
whose `StableTorchLibraryInit` static initializers populate `torch.ops.deformops_cuda.*`.
`used_prebuilt` reports whether a library came from outside this machine.

## Prebuilt downloads

The index defaults to `https://tue-p8n.github.io/deformops/ops`;
`DEFORMOPS_OPS_INDEX_URL` points to a private copy.
Each package version and CUDA tag has a directory with a manifest and the libraries it lists.

### Cache

- The cache root is `~/.cache/deformops/ops`, or `DEFORMOPS_OPS_CACHE_DIR`.
  When `Path.home()` raises (restricted containers, minimal runners),
  it falls back to the system temporary directory, scoped by user ID.
- The cache is keyed on package version and CUDA tag, not GPU architecture,
  because `nix/ops.nix` builds every published library for the toolkit's full compute-capability list.
- With the index unreachable, a cached copy loads offline; it was verified when written.
  A library missing from a reachable index is not loaded from the cache,
  since removal from the index withdraws it.

### Integrity and transport

- **HTTPS only.** A plaintext `http://` index needs `DEFORMOPS_OPS_ALLOW_INSECURE`.
  Over plaintext an attacker can replace both a library and its manifest SHA-256,
  so enable it only for an index on a trusted network.
- **Schemes.** Only `http` and `https` are fetched, never `file://` or another urllib scheme.
- **Size caps.** 1 MiB for the manifest (a real one is a few hundred bytes per extension),
  512 MiB for a library (real ones are tens of megabytes).
  The cap counts bytes read, not the server's `Content-Length`,
  so an under-reporting host cannot exhaust memory.
  The loader reads one byte past the cap,
  so an over-long body raises a size error, not a checksum mismatch.
- **File names.** Manifest file names must be bare names,
  refusing traversal and absolute paths.
  The name and digest come from the same manifest, so this is defense in depth.
- **Digests.** Digests are compared in lowercase.
  A checksum mismatch (corruption, a truncating proxy, tampering) always warns.
- **Ownership.** The POSIX fallback cache is under world-writable `/tmp`,
  where a co-tenant could pre-create `deformops-<uid>`
  and plant a library between verification and loading.
  So the current user must own the cache root,
  every directory down to the file,
  and, for the `/tmp` fallback, the `deformops-<uid>` directory.
  None of them may be group- or world-writable,
  which a permissive umask on shared scratch or NFS can cause.
  Platforms without `os.getuid` skip the check.

### Fetching

- A 10 s socket timeout keeps a slow network from stalling first use.
- A named User-Agent avoids CDN and firewall rejections; `HTTPS_PROXY` is honored.
- Manifest fetches, failures included, are memoized per process and index directory,
  so an unreachable index costs one timeout per process.
  Network, HTTP status and malformed-JSON failures are recorded for the caller to report.
- A download writes a unique temporary file in the cache directory
  and replaces the target atomically on POSIX,
  so concurrent downloaders (DDP ranks, threads) never collide.
  A failed write still succeeds if another process cached a valid copy,
  and the temporary file is always removed.

## Local builds

Local compiles go through `torch.utils.cpp_extension` and need a CUDA toolkit and `deformops[jit]`.

### Build lock

Torch's `load()` serializes builds with `FileBaton`,
an `O_CREAT | O_EXCL` lock file with no liveness check,
which `FileBaton.wait()` polls forever.
A build killed early (Ctrl-C, out of memory, walltime) leaves the file,
and every later build of that extension hangs without output.

`serialized_build` holds an `flock` on `.deformops-jit.flock` instead;
anything driving `torch.utils.cpp_extension` should use it.

- The OS releases an `flock` when the holder exits, even on `SIGKILL`.
- Under the lock, `serialized_build` removes a leftover torch `lock` file.
- `rebuild` clears the build directory under the lock but keeps the sidecar file.
  Unlinking it would not release the holder's lock on the old inode,
  while a concurrent `open(path, "w")` would lock a new inode at the same name.
- The lock is skipped, and the build proceeds,
  for a `None` build directory,
  a platform without `fcntl` (Windows),
  or a directory that cannot be created or written.

### CUDA architectures

`define_library` passes its own `-gencode` flags to `load()`.
Without them torch reads the visible devices and appends `+PTX` to the last entry,
which raises an `IndexError` when no device is visible and `TORCH_CUDA_ARCH_LIST` is unset.
Any `extra_cuda_cflags` entry containing `arch` skips that detection.

- `TORCH_CUDA_ARCH_LIST` holds space- or semicolon-separated `major.minor` entries,
  each optionally suffixed `+PTX`.
  `native` counts as unset.
  Named aliases such as `Ampere` are rejected, since their expansion depends on the torch version.
- Otherwise every visible device's compute capability is used, not only device 0's,
  deduplicated, sorted numerically, with `+PTX` on the highest.
  Mixed GPU generations need gencode for every device,
  or the library fails on all but device 0.
  Numeric sorting puts `10.0` after `8.6`, so `+PTX` goes to the newest card,
  and suffixed numbers such as `9.0a` still parse.
- A `sm_*` and `compute_*` pair is ordered `compute_*` (PTX) first,
  as torch's `_get_cuda_arch_flags` does for one architecture.
- With the variable unset and no device visible, resolution returns `None`.
  Finding an existing library needs no architecture;
  `define_library` raises only when it reaches a compile.
- The flags enter the build fingerprint in a stable order,
  so the build directory does not change between runs.

Only `tests/deformops/_internal/test_cuda_arch_flags.py` calls `_cuda_arch_flags`.

### Build fingerprint and directory

The fingerprint covers:

- every source file;
- every `.cuh`, `.h` and `.hpp` header under the include root or beside a source file, recursively;
- the compile flags and `with_cuda`;
- the torch and CUDA environment string.

Headers beside a `.cu` count, so editing one rebuilds.
Headers are keyed by path relative to their root,
so same-named headers in both roots do not collide.

The build directory is `<name>-<environment>-<archtag>-<hash>`,
where the environment holds the torch, CUDA and cuDNN versions.
The default root is namespaced by package and environment;
`TORCH_EXTENSIONS_DIR` overrides it.

The architecture is a path component, not part of the fingerprint,
so a call with no resolvable architecture can still find a build made with a device present.

- With the architecture unresolved,
  a directory matching everything else is reused,
  since no kernel can run here anyway.
  Without a match, `<name>-<environment>-unresolved-<hash>` is used.
- `rebuild` skips that lookup,
  because it clears the returned directory,
  which could belong to another architecture's build.

A build killed mid-link (walltime, a full quota) leaves a truncated library under an unchanged fingerprint.
It is removed on the failed load, and the caller compiles again.

`DEFORMOPS_OPT_LEVEL` overrides the `opt_level` argument (default 2) for debug builds;
unset or empty keeps the argument.

### Capability detection

Distribution metadata does not record selected extras, so capabilities are detected.
A CUDA build of torch depends on Triton, which may be present without `deformops[triton]`.

Toolchain detection checks only for ninja.
Torch's load path raises a bare `RuntimeError` without it,
while it reports a missing CUDA toolchain clearly.

## Ahead-of-time builds

`setup.py` compiles kernels during `build`, `build_ext`, `bdist_wheel`, `install`, `develop` and `editable_wheel`.
Metadata and sdist commands neither import torch nor need a CUDA toolkit.

### Selection

- A checkout compiles nothing by default.
  An unpacked sdist, recognized by its root `PKG-INFO`,
  compiles the non-experimental kernels unless `DEFORMOPS_EXTENSIONS` says otherwise.
- The default selection compiles only against a CUDA build of torch
  importable in the build environment,
  so the library matches the torch that loads it.
  An isolated build compiles nothing,
  and without a CUDA toolkit the default skips without a message.
- The default selection never fails the install.
  It writes the reason to stderr and to `build-skipped.txt` in the extension package,
  and the runtime reports it when the CUDA backend finds no bundled library.
  A CPU build of torch is skipped without a report.
- Kernels named through `DEFORMOPS_EXTENSIONS` or `FORCE_CUDA` are required:
  a build that cannot compile them fails.
- `DEFORMOPS_EXTENSIONS=all` and the prebuilt libraries exclude the experimental configurations.

### Compilation

- `OPT_LEVEL` (default `-O3`) is passed as given.
- The shared CUDA include directory is absolute,
  because ninja runs from `build/temp.*`.
- distutils maps every source to `.o`,
  so `<stem>.cpp` and `<stem>.cu` would collide and ninja would abort.
  `build_ext` compiles `.cu` to `<stem>.cuda.o`, as torch's JIT does.
- Ahead-of-time builds pass `-DDEFORMOPS_NO_FP64`,
  removing a quarter of the scalar-type dispatch;
  float64 serves gradient checks only.
  The JIT path keeps it.
  Asking an ahead-of-time library for float64 raises an error
  naming `DEFORMOPS_COMPILE=always` (a CUDA toolkit and `deformops[jit]`).
  See [precision](precision.md).
- Every extension undefines
  `__CUDA_NO_HALF_OPERATORS__`, `__CUDA_NO_HALF_CONVERSIONS__` and `__CUDA_NO_HALF2_OPERATORS__`,
  which torch's extension build defines,
  because the FP16 and BF16 kernel paths use those operators.

### Packaging

- Libraries are bare `<name>.so`, the name the runtime looks for,
  because they load through `torch.ops.load_library`, not as Python modules.
- Built libraries install into `deformops.core._ext`,
  a name `setup.py`, the runtime loader and the Nix derivations read from `core/configs.py`.
- `package_data` ships the prebuilt libraries (`_ext/*.so`, `*.pyd`, `build-skipped.txt`)
  and the CUDA sources and headers,
  so an install without a prebuilt library can compile one.
  A wheel built from a checkout bundles no library.
  `include-package-data` covers the sdist; wheels need the explicit list.
- `DEFORMOPS_WHEEL_PLATLIB=1` builds a platform wheel.
  Package data would get a `py3-none-any` tag,
  so the custom `bdist_wheel` forces a platform tag.
  It keeps `py3-none`, since a stable-ABI library does not depend on the Python or PyTorch version.
- `VERSION` holds the base version.
  `pyproject.toml` declares the version dynamic,
  so `setup.py` can append a PEP 440 local identifier (`+cuXXX`) for CUDA wheels,
  from `CUDA_VERSION`,
  or from `torch.version.cuda` under `FORCE_CUDA=1` with a visible device.
  Without distribution metadata (editable or `PYTHONPATH` use),
  `deformops.__version__` reads the checkout's `VERSION`.

## Names and paths shared across build systems

- **Extension package.** `deformops.core._ext` (`EXT_PACKAGE` in `core/configs.py`) holds the prebuilt libraries.
  `setup.py` compiles into it,
  the Nix ops derivation copies out of it,
  the Nix wheel derivation copies back in,
  and `build.py` loads from it.
  A mismatched path builds wheels whose kernels cannot load;
  `tests/deformops/_internal/test_ext_path.py` compares the four.
- **Extension names.** An extension name is also its schema prefix (`deformops_cuda::<name>_*`)
  and the file stem of its translation units and Python module,
  so `msda_half2atomic` builds from `experimental/msda_half2atomic.{cpp,cu}`.
  The JIT path finds the same files through `importlib.resources.files(package)`,
  so they cannot be renamed independently.
- **Prefix list.** `configs.py` writes out the multi-scale prefixes
  instead of importing `_schema.OP_PREFIXES`,
  because `setup.py` loads it by path without the package.
  A test compares the two.
- **Supported CUDA versions.** The `pyproject.toml` extras,
  the `flake.nix` variant list behind the `ops-*` and `wheel-*` derivations,
  and the Pages workflow matrix declare them separately.
  The workflow runs only when dispatched,
  so `tests/deformops/_internal/test_cuda_variants_agree.py` parses all three
  instead of waiting for a release build to fail.

## Stable ABI

Every kernel compiles with `TORCH_TARGET_VERSION` set to `MINIMUM_TORCH_ABI`, the oldest supported torch,
so a library built against newer headers still loads there,
and torch rejects headers outside its stable surface.

- `TORCH_CHECK` and the `CHECK_*` macros throw `std::runtime_error`,
  so the library does not import `c10::detail::torchCheckFail`,
  which would bind it to one PyTorch C++ ABI.
  Message arguments are concatenated,
  and each macro is a `do { ... } while (0)` statement.
  CUDA device code compiles to PTX and has no C++ ABI imports.
- `dispatch_scalar_type` lists the supported dtypes once for every entry point;
  its `Double` branch compiles unless `DEFORMOPS_NO_FP64` is defined.
- The current CUDA stream comes from the stable-ABI C shim.
  Both shim calls report failure through their return value
  and leave the out-parameters unset for a non-CUDA tensor,
  so the caller checks before reading the handle.
- Boxed unboxing uses `torch::stable::detail::{to,from}`;
  the global `to` and `from` are deprecated and warn on every use.

## Tests

- `test_msda_lazy_binding.py` stubs `define_library` with a counter on the CPU.
  On CUDA it runs `msda_fp32acc.py` in a fresh subprocess,
  so earlier tests' loads do not interfere,
  and checks for one load, then none.
  Its GPU shapes match `test_msda_fake_tensors.py`, where `D` (8) divides the tuned `channels_per_thread`.
- `test_msda_prebuild.py` gives each module a fresh `lazy_extension` closure through monkeypatch,
  because a module whose real `load` already ran would skip `define_library` regardless of the stub.
- `test_msda_schema_registration.py` checks a fresh interpreter in a subprocess.
  It writes out the prefix list,
  so it fails when `_schema.OP_PREFIXES` changes.
