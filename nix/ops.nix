# Builds the stable-ABI CUDA operator libraries (.so) for the deformops package.
#
# Unlike ``nix/wheel.nix`` (which bundles these libraries into a wheel), this
# derivation compiles every extension in ``EXTENSION_CONFIGS`` and emits the bare
# ``<name>.so`` files. Because the kernels are written against the Torch Stable
# ABI (``STABLE_TORCH_LIBRARY_FRAGMENT`` + ``stable_compat.cuh``), the resulting
# libraries link no ``libc10`` / ``libtorch`` C++ symbols and are therefore
# independent of the Python and PyTorch version; only the CUDA version matters.
# One build per CUDA version, with the outputs hosted on a package index.
#
# Arguments:
#   pkgs        - Package set configured for a specific CUDA version.
#                 Pass as `inherit (pkgs.cudaPackages_12_X) pkgs` from flake.nix.
#   pythonEnv   - A venv (e.g. from uv2nix's `mkVirtualEnv`) with `torch` and
#                 `setuptools` importable. setup.py imports torch at build time
#                 for CUDAExtension (Python API + C++/CMake headers); torch
#                 also carries the stable-ABI headers the kernels compile
#                 against. Not nixpkgs' `python.withPackages torch`, which
#                 is a from-source CUDA build; the prebuilt PyPI wheel torch
#                 suffices, and the stable-ABI .so does not depend on which
#                 torch built it.
#   name        - CUDA version label used for the output directory (e.g. "cu128").
#   cudaVersion - CUDA version number string (e.g. "12.8").
#   src         - Source tree of the deformops project (typically `self`).
{
  pkgs,
  pythonEnv,
  name,
  cudaVersion,
  src,
}: let
  inherit (pkgs) lib;
  cudaPkgs = pkgs.cudaPackages;
  versionStr = lib.strings.removeSuffix "\n" (builtins.readFile "${src}/VERSION");
in
  pkgs.stdenv.mkDerivation {
    pname = "deformops-ops";
    version = "${versionStr}+${name}";

    inherit src;

    nativeBuildInputs = [
      pythonEnv
      pkgs.ninja
      cudaPkgs.cudatoolkit
      cudaPkgs.backendStdenv.cc
    ];

    buildInputs = [
      cudaPkgs.cuda_cudart
      cudaPkgs.cudnn
    ];

    # Compile every extension in EXTENSION_CONFIGS with CUDA kernels enabled.
    DEFORMOPS_EXTENSIONS = "all";
    FORCE_CUDA = "1";
    CUDA_VERSION = cudaVersion;
    CUDA_HOME = "${cudaPkgs.cudatoolkit}";
    CUDA_PATH = "${cudaPkgs.cudatoolkit}";
    # The build sandbox has no GPU, so torch.utils.cpp_extension cannot
    # autodetect a target architecture. Use nixpkgs' per-CUDA-version
    # capability list, so the build targets exactly what this toolkit supports.
    #
    # The highest architecture also carries `+PTX`, which embeds the kernels'
    # PTX intermediate code next to that architecture's binary image. On a GPU
    # newer than every listed architecture, the CUDA driver compiles the PTX
    # at load time. The capabilities are sorted first because the list's order
    # is not guaranteed.
    TORCH_CUDA_ARCH_LIST = let
      caps = lib.sort lib.versionOlder cudaPkgs.backendStdenv.cudaCapabilities;
    in
      lib.concatStringsSep ";" (lib.init caps ++ ["${lib.last caps}+PTX"]);

    buildPhase = ''
      runHook preBuild
      # --inplace writes the compiled libraries next to the sources at
      # sources/deformops/core/_ext/<name>.<pytag>.so
      python setup.py build_ext --inplace
      runHook postBuild
    '';

    # Strip the interpreter/ABI tag from the filename. The stable-ABI libraries
    # are dlopen'd via torch.ops.load_library (no PyInit_* symbol), so the tag
    # is meaningless and the loader / index expect a bare ``<name>.so``.
    installPhase = ''
      runHook preInstall
      mkdir -p "$out"
      shopt -s nullglob
      for f in sources/deformops/core/_ext/*.so; do
        base="$(basename "$f")"
        stem="''${base%%.*}"
        cp "$f" "$out/$stem.so"
      done
      if [ -z "$(ls -A "$out")" ]; then
        echo "error: no .so files were produced" >&2
        exit 1
      fi
      runHook postInstall
    '';

    meta = {
      description = "Stable-ABI deformops operator libraries for CUDA ${cudaVersion} (${name})";
      platforms = ["x86_64-linux"];
    };
  }
