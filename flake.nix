{
  description = "Deformable grid sampling operations.";

  inputs = {
    tue-p8n.url = "github:tue-p8n/nix";
    nixpkgs.follows = "tue-p8n/nixpkgs";
    flake-parts.follows = "tue-p8n/flake-parts";
    nixpkgs-cuda.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
  };

  outputs =
    inputs:
    inputs.flake-parts.lib.mkFlake { inherit inputs; } {
      imports = [
        inputs.tue-p8n.flakeModules.cuda
        inputs.tue-p8n.flakeModules.formatting
      ];
      systems = [ "x86_64-linux" ];
      perSystem =
        {
          p8n,
          pkgs,
          system,
          ...
        }:
        let
          inherit (pkgs) lib;

          nixpkgsCuda = import inputs.nixpkgs-cuda {
            inherit system;
            config.allowUnfree = true;
          };

          projectName = "deformops";
          pyproject = p8n.uv.readProject ./.;

          # One CPU environment, shared by the interactive shell and every
          # check. The uv builder merges every dependency group with the extra
          # it resolves from the accelerator, so this matches what
          # `uv sync --dev --extra cpu` installs.
          cpuVenv = pyproject.mkVenv { accelerator = "cpu"; };

          # Same CPU environment plus the `triton` extra, for the typecheck
          # check only. Without Triton, pyright cannot resolve `triton`, and
          # every `@triton.jit` kernel degrades to a plain function with
          # errors at each launch site. `cpuVenv` stays Triton-free, because
          # the test check and the wheel build use it as a CPU install.
          typecheckVenv = pyproject.mkVenv {
            accelerator = "cpu";
            extras = [ "triton" ];
          };

          # The ruff from the lock, so treefmt and the lint check use the same
          # version as the dev shells rather than the one nixpkgs carries. The
          # venv has no `meta.mainProgram`, so `lib.getExe`, which treefmt uses
          # to find a formatter's binary, needs it set to `ruff`.
          ruffEnv =
            let
              venv = cpuVenv.pythonSet.mkVirtualEnv "deformops-ruff" {
                ruff = [ ];
              };
            in
            venv
            // {
              meta = venv.meta // {
                mainProgram = "ruff";
              };
            };

          # Only these paths feed the per-CUDA kernel builds (ops.nix /
          # wheel.nix).
          opsSrc = lib.fileset.toSource {
            root = ./.;
            fileset = lib.fileset.unions [
              ./setup.py
              ./pyproject.toml
              ./MANIFEST.in
              ./VERSION
              ./sources/deformops
            ];
          };

          cudaVariants = [
            {
              extra = "cu126";
              accel = "cuda12_6";
              version = "12.6";
              pkgs = pkgs.cudaPackages_12_6.pkgs;
            }
            {
              extra = "cu128";
              accel = "cuda12_8";
              version = "12.8";
              pkgs = pkgs.cudaPackages_12_8.pkgs;
            }
            {
              extra = "cu129";
              accel = "cuda12_9";
              version = "12.9";
              pkgs = pkgs.cudaPackages_12_9.pkgs;
            }
            {
              extra = "cu130";
              accel = "cuda13_0";
              version = "13.0";
              pkgs = pkgs.cudaPackages_13_0.pkgs;
            }
            {
              extra = "cu132";
              accel = "cuda13_2";
              version = "13.2";
              pkgs = pkgs.cudaPackages_13_2.pkgs;
            }
            {
              extra = "cu134";
              accel = "cuda13_4";
              version = "13.4";
              pkgs = pkgs.cudaPackages_13_4.pkgs;
            }
          ];

          # Commands needing a device run inside the CUDA shell, whose shell
          # hook sets up the host driver. `nix develop .#cu128` resolves
          # relative to the caller's working directory, so these apps only
          # work when run from the repository.
          gpuApp =
            name: command:
            pkgs.writeShellApplication {
              inherit name;
              runtimeInputs = [ pkgs.nix ];
              text = ''
                exec nix develop .#cu128 --command ${command}
              '';
            };

          # Generates compile_flags.txt pointing to the current venv's torch
          # and CUDA include directories so clangd works immediately upon
          # entering the shell or through direnv in editors.
          clangdHook = ''
            if [ -n "''${REPO_ROOT:-}" ] && command -v python >/dev/null 2>&1; then
              _torch_dir=$(python -c "import torch; print(torch.__path__[0])" 2>/dev/null || true)
              if [ -n "$_torch_dir" ]; then
                _cuda_home="''${CUDA_HOME:-''${CUDA_PATH:-}}"
                {
                  echo "-I$_torch_dir/include"
                  echo "-I$_torch_dir/include/torch/csrc/api/include"
                  if [ -n "$_cuda_home" ]; then
                    echo "-I$_cuda_home/include"
                    echo "--cuda-path=$_cuda_home"
                  fi
                  echo "-Isources/deformops/core/cuda"
                } > "''${REPO_ROOT}/compile_flags.txt"
              fi
              unset _torch_dir _cuda_home
            fi
          '';

          mkCudaShell =
            accel:
            (pyproject.mkShell {
              preCommit = p8n.preCommit;
              accelerator = accel;
              packages = [
                pkgs.clang-tools
                pkgs.cudaPackages.nsight_compute
                pkgs.ninja
                # AOTInductor packaging invokes the openssl CLI.
                pkgs.openssl
              ];
            }).overrideAttrs
              (prev: {
                shellHook = (prev.shellHook or "") + clangdHook;
              });

          # Pure generation of compile_flags.txt pointing to the default CUDA
          # toolkit (12.8) and torch headers. `nix build .#compile-flags` produces
          # $out/compile_flags.txt; `nix run .#clangd-setup` copies it to
          # the repository root.
          compileFlags =
            let
              cu128 = builtins.head (builtins.filter (v: v.extra == "cu128") cudaVariants);
              cu128Venv = pyproject.mkVenv { accelerator = cu128.accel; };
              cudaHome = "${cu128.pkgs.cudaPackages.cudatoolkit}";
            in
            pkgs.runCommand "compile-flags" { nativeBuildInputs = [ cu128Venv ]; } ''
              TORCH_DIR=$(python -c "import torch; print(torch.__path__[0])")
              mkdir -p "$out/bin"
              {
                echo "-I$TORCH_DIR/include"
                echo "-I$TORCH_DIR/include/torch/csrc/api/include"
                echo "-I${cudaHome}/include"
                echo "--cuda-path=${cudaHome}"
                echo "-Isources/deformops/core/cuda"
              } > "$out/compile_flags.txt"

              cat <<EOF > "$out/bin/clangd-setup"
              #!/usr/bin/env bash
              set -euo pipefail
              target_dir=\$(git rev-parse --show-toplevel 2>/dev/null || pwd)
              install -m 644 "$out/compile_flags.txt" "\$target_dir/compile_flags.txt"
              echo "Generated \$target_dir/compile_flags.txt"
              EOF
              chmod +x "$out/bin/clangd-setup"
            '';
        in
        {
          p8n.cuda.overlays = [
            (_final: _prev: {
              cudaPackages_13_4 = nixpkgsCuda.cudaPackages_13_4;
            })
          ];

          devShells =
            builtins.listToAttrs (
              map (v: {
                name = v.extra;
                value = mkCudaShell v.accel;
              }) cudaVariants
            )
            // {
              default = mkCudaShell "cuda12_8";

              # The fast local loop. The shell owns its environment, so the
              # CPU and GPU builds of torch never share a directory.
              cpu = pyproject.mkShell {
                preCommit = p8n.preCommit;
                accelerator = "cpu";
                packages = [
                  pkgs.pyright
                  pkgs.clang-tools
                  pkgs.ninja
                  # AOTInductor packaging invokes the openssl CLI.
                  pkgs.openssl
                ];
              };
            };

          packages =
            builtins.listToAttrs (
              lib.concatMap (
                v:
                let
                  # The resolved package set, read from mkProject's passthru
                  # (evaluation only), so the ops derivation can swap in the
                  # reduced source tree without re-resolving the lock.
                  ps = (pyproject.mkProject { accelerator = v.accel; }).pythonSet;
                  opsPythonSet = ps.overrideScope (
                    _final: prev: {
                      ${projectName} = prev.${projectName}.overrideAttrs (_: {
                        src = opsSrc;
                      });
                    }
                  );
                  opsPythonEnv = opsPythonSet.mkVirtualEnv "${projectName}-ops-${v.extra}-venv" {
                    ${projectName} = [ v.extra ];
                    setuptools = [ ];
                    ninja = [ ];
                  };
                  opsLib = import ./nix/ops.nix {
                    inherit (v) pkgs;
                    pythonEnv = opsPythonEnv;
                    name = v.extra;
                    cudaVersion = v.version;
                    src = opsSrc;
                  };
                in
                [
                  {
                    name = v.extra;
                    value = pyproject.mkVenv {
                      accelerator = v.accel;
                    };
                  }
                  {
                    name = "ops-${v.extra}";
                    value = opsLib;
                  }
                  {
                    # Per-CUDA wheel that bundles the prebuilt .so from opsLib.
                    name = "wheel-${v.extra}";
                    value = import ./nix/wheel.nix {
                      inherit (v) pkgs;
                      inherit opsLib;
                      name = v.extra;
                      cudaVersion = v.version;
                      src = opsSrc;
                    };
                  }
                ]
              ) cudaVariants
            )
            // {
              default = pyproject.mkVenv {
                accelerator = "cuda12_8";
              };
              cpu = cpuVenv;

              # The pure-Python wheel that goes to PyPI: no kernels, py3-none-any.
              # A checkout compiles no kernels by default, which keeps it pure. Built
              # through setuptools' PEP 517 hook directly, because the `build`
              # front end is not in the locked environment.
              wheel = pkgs.runCommand "deformops-wheel" { nativeBuildInputs = [ cpuVenv ]; } ''
                                export HOME="$TMPDIR"
                                cp -r ${./.} src
                                chmod -R u+w src
                                cd src
                                mkdir -p "$out"
                                python -c "
                from setuptools import build_meta
                print(build_meta.build_wheel('$out'))
                "
              '';

              compile-flags = compileFlags;
            };

          apps =
            let
              app = pkg: name: {
                type = "app";
                program = "${pkg}/bin/${name}";
              };
            in
            {
              # Not `-m gpu`: that marker deselects the CUDA tests gated by a
              # module-level skip or the device fixture instead, including the
              # gradcheck file. "not integration" runs everything the host can
              # execute.
              #
              # `not fp64` excludes the double-precision tests that route
              # through a compiled CUDA extension (see the `fp64` marker), so
              # this run exercises the prebuilt-library path. `test-gpu-fp64`
              # below runs them with a just-in-time build.
              #
              # No `-n auto`: each xdist worker holds its own CUDA context.
              test-gpu = app (gpuApp "test-gpu" ''pytest --benchmark-disable -m "not integration and not fp64" -v'') "test-gpu";

              test-gpu-quick = app (gpuApp "test-gpu-quick" ''pytest --benchmark-disable -m "gpu and not fp64" -v'') "test-gpu-quick";

              # Prebuilt libraries exclude double precision, so the fp64 tests
              # compile their own. `env` because `nix develop --command` execs
              # its argument without a shell.
              test-gpu-fp64 = app (gpuApp "test-gpu-fp64" "env DEFORMOPS_COMPILE=always pytest --benchmark-disable -m fp64 -v") "test-gpu-fp64";

              check-gpu = app (gpuApp "check-gpu" ''
                python -c "
                import torch
                from deformops.registry import Backend, msda
                assert torch.cuda.is_available(), 'torch cannot see the GPU inside nix develop'
                print('device:', torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
                print('cudnn:', torch.backends.cudnn.version())
                print(msda.load(Backend.CUDA, download=False, compile=True))
                "'') "check-gpu";

              # Runs in the CUDA shell because the CUDA include directory
              # and CUDA_HOME it writes only resolve there.
              clangd = app (gpuApp "clangd" ''
                bash -c "
                  TORCH_DIR=\$(python -c \"import torch; print(torch.__path__[0])\")
                  CUDA_HOME=\"\''${CUDA_HOME:-\''${CUDA_PATH:-\$(python -c \"from torch.utils.cpp_extension import CUDA_HOME; print(CUDA_HOME or ''')\")}}\"
                  {
                    echo \"-I\$TORCH_DIR/include\"
                    echo \"-I\$TORCH_DIR/include/torch/csrc/api/include\"
                    if [ -n \"\$CUDA_HOME\" ]; then
                      echo \"-I\$CUDA_HOME/include\"
                      echo \"--cuda-path=\$CUDA_HOME\"
                    fi
                    echo \"-Isources/deformops/backends/cuda/include\"
                  } > compile_flags.txt
                  echo \"Generated compile_flags.txt\"
                "
              '') "clangd";
            };

          checks = {
            # Ruff diagnostics and formatting, checked directly. The
            # `treefmt` check catches unfixable diagnostics only because
            # treefmt-nix's `check --fix` overrides the shared formatting
            # module's `--fix-only`; this check does not depend on which
            # option wins, and its failure names lint or formatting.
            lint = pkgs.runCommand "deformops-lint" { nativeBuildInputs = [ ruffEnv ]; } ''
              export HOME="$TMPDIR"
              cd ${./.}
              ruff check --no-cache .
              ruff format --check --no-cache .
              touch "$out"
            '';

            # Strict mode without torch's own types is a wall of unknown-type
            # errors rather than real findings, so this runs against the CPU
            # environment rather than a bare pyright.
            typecheck =
              pkgs.runCommand "deformops-typecheck"
                {
                  nativeBuildInputs = [
                    typecheckVenv
                    pkgs.pyright
                  ];
                }
                ''
                  export HOME="$TMPDIR"
                  cp -r ${./.} src
                  chmod -R u+w src
                  cd src
                  pyright --pythonpath ${typecheckVenv}/bin/python
                  touch "$out"
                '';

            # The whole CPU suite, the same markers CI uses, the coverage gate
            # included.
            tests =
              pkgs.runCommand "deformops-tests"
                {
                  nativeBuildInputs = [ cpuVenv ];
                }
                ''
                  export HOME="$TMPDIR"
                  # Checks never use the network.
                  export DEFORMOPS_DOWNLOAD=0
                  export TORCH_EXTENSIONS_DIR="$TMPDIR/torch-extensions"
                  export TRITON_CACHE_DIR="$TMPDIR/triton"
                  cp -r ${./.} src
                  chmod -R u+w src
                  cd src
                  # Each xdist worker loads its own copy of torch, so cap
                  # the worker count at 4 whatever NIX_BUILD_CORES reports.
                  workers="''${NIX_BUILD_CORES:-4}"
                  if [ "$workers" -eq 0 ] || [ "$workers" -gt 4 ]; then
                    workers=4
                  fi
                  # pytest-sugar undercounts skips under xdist, so it is
                  # disabled and the summary is pytest's own.
                  pytest \
                    -n "$workers" \
                    -p no:cacheprovider \
                    -p no:sugar \
                    --benchmark-disable \
                    -m "not gpu and not integration" \
                    --cov --cov-report=term
                  touch "$out"
                '';

          };

          # The imported formatting module enables `ruff.check` and
          # `ruff.format`, which treefmt-nix renames to the separate
          # `ruff-check` and `ruff-format` programs; there is no combined
          # `ruff` program to pin, so both get the same package.
          treefmt.programs.ruff-check.package = ruffEnv;
          treefmt.programs.ruff-format.package = ruffEnv;
          treefmt.settings.global.excludes = [
            ".envrc"
          ];

          # The gate for CUDA sources is clang-format, run by the treefmt
          # check. clang-tidy is off: it cannot resolve the torch and ATen
          # headers these sources include outside a CUDA shell.
          treefmt.programs.clang-tidy.enable = false;
          treefmt.programs.clang-format.includes = [
            "*.c"
            "*.cc"
            "*.cpp"
            "*.h"
            "*.hh"
            "*.hpp"
            "*.cuh"
            "*.cu"
          ];
        };
    };
}
