# Security policy

## Supported versions

Security fixes go to the latest release line, currently 3.0.x.

## Reporting a vulnerability

Report a vulnerability privately,
through [GitHub's private vulnerability reporting](https://github.com/tue-p8n/deformops/security/advisories/new)
or by email to kurt@computer.org.
Do not open a public issue for it.

Include the affected version,
the environment (PyTorch, CUDA and Python versions),
and the steps or code that reproduce the problem.

## Scope

`deformops` loads compiled code at runtime.
Reports about the following are in scope:
the download and checksum verification of prebuilt operator libraries,
the JIT build and its cache directories,
and memory safety in the CUDA kernels.
`deformops` fetches the prebuilt-library index over HTTPS
and verifies each library against the checksum in the index manifest.
`DEFORMOPS_OPS_ALLOW_INSECURE` disables the HTTPS requirement;
leave it unset outside a trusted private mirror.
