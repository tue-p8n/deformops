# Builds a per-CUDA deformops wheel that bundles the prebuilt stable-ABI
# operator libraries produced by ./ops.nix. No compilation happens here: the
# .so are copied into the package tree and picked up as package data, and the
# wheel is tagged ``py3-none-<platform>`` (see setup.py::_platlib_bdist_wheel)
# with a ``+cuXXX`` local version. These wheels are served from the project's
# own PEP 503 index (GitHub Pages), not PyPI.
#
# Arguments:
#   pkgs        - Package set configured for a specific CUDA version.
#   name        - CUDA version label used in the wheel version (e.g. "cu128").
#   cudaVersion - CUDA version number string (e.g. "12.8").
#   src         - Source tree of the deformops project (typically `self`).
#   opsLib      - Derivation from ./ops.nix whose $out holds the <name>.so files.
{
  pkgs,
  name,
  cudaVersion,
  src,
  opsLib,
}: let
  python = pkgs.python313;

  # setup.py needs setuptools + wheel to build the wheel. torch is not
  # needed: nothing is compiled on the bundle path.
  pythonEnv = python.withPackages (
    ps:
      with ps; [
        setuptools
        wheel
      ]
  );
in
  pkgs.stdenv.mkDerivation {
    pname = "deformops-wheel";
    version = let
      inherit (pkgs) lib;
      versionStr = lib.strings.removeSuffix "\n" (builtins.readFile "${src}/VERSION");
    in "${versionStr}+${name}";

    inherit src;

    nativeBuildInputs = [pythonEnv];

    # Tag the wheel platform-specific + python/abi-agnostic and stamp the
    # +cuXXX local version. No FORCE_CUDA / DEFORMOPS_EXTENSIONS: the prebuilt
    # .so are bundled instead of compiled.
    DEFORMOPS_WHEEL_PLATLIB = "1";
    CUDA_VERSION = cudaVersion;

    buildPhase = ''
      runHook preBuild
      # Drop the prebuilt stable-ABI libraries where package_data expects them.
      mkdir -p sources/deformops/core/_ext
      cp ${opsLib}/*.so sources/deformops/core/_ext/
      python setup.py bdist_wheel
      runHook postBuild
    '';

    installPhase = ''
      runHook preInstall
      mkdir -p $out
      cp dist/*.whl "$out/"
      runHook postInstall
    '';

    meta = {
      description = "deformops wheel bundling prebuilt CUDA ${cudaVersion} (${name}) kernels";
      platforms = ["x86_64-linux"];
    };
  }
