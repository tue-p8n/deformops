"""Run one operator with the kernel library the environment selects.

A subprocess of ``test_binary_jit_parity.py``: saves the forward output and input
gradients to ``DEFORMOPS_PARITY_OUT`` and prints one JSON timing line.
"""

import json
import os
import sys
import time

import torch
from deformops.registry import Backend, msda
from deformops.registry import attn as attn_op

seed = int(os.getenv("DEFORMOPS_SEED", "1958"))
torch.manual_seed(seed)
device = torch.device("cuda")

op_name = os.getenv("DEFORMOPS_OP", "attn")
dtype_str = os.getenv("DEFORMOPS_DTYPE", "float32")
dtype = getattr(torch, dtype_str)
out_path = os.environ["DEFORMOPS_PARITY_OUT"]

# Common params
N = 2
chunk_step = 64

if op_name == "attn":
    H, W, C = 32, 32, 128
    G, C_per_group = 8, 16
    KH, KW = 3, 3
    stride, pad, dilation = 1, 1, 1
    HO, WO = 32, 32

    value = torch.randn((N, H, W, C), device=device, dtype=dtype, requires_grad=True)
    offset_mask = torch.randn(
        (N, HO, WO, G * KH * KW * 3), device=device, dtype=dtype, requires_grad=True
    )
    grad_tensors = {"value": value, "offset_mask": offset_mask}

    def forward_fn():
        return attn_op(
            value,
            offset_mask,
            kernel_size=(KH, KW),
            stride=(stride, stride),
            padding=(pad, pad),
            dilation=(dilation, dilation),
            groups=G,
            group_dims=C_per_group,
            spread=1.0,
        )

elif op_name == "msda":
    M, D, L, P, Q = 8, 16, 4, 4, 300
    shapes = torch.tensor([(32, 32)] * L, dtype=torch.long, device=device)
    start_index = torch.cat(
        [torch.tensor([0], device=device), (shapes[:-1, 0] * shapes[:-1, 1])]
    ).cumsum(0)
    S = (shapes[:, 0] * shapes[:, 1]).sum().item()

    value = torch.randn((N, S, M, D), device=device, dtype=dtype, requires_grad=True)
    locs = torch.rand(
        (N, Q, M, L, P, 2), device=device, dtype=dtype, requires_grad=True
    )
    attn_raw = torch.randn(
        (N, Q, M, L, P), device=device, dtype=dtype, requires_grad=True
    )
    attn_softmaxed = attn_raw.softmax(-1)

    grad_tensors = {"value": value, "locs": locs, "attn": attn_raw}

    def forward_fn():
        # Explicit CUDA: automatic selection could pick Triton and skip the library.
        return msda(
            value,
            shapes,
            start_index,
            locs,
            attn_softmaxed,
            chunk_step=chunk_step,
            method="linear",
            backend=Backend.CUDA,
        )

else:
    print(f"unknown op {op_name!r}", file=sys.stderr)
    sys.exit(1)

# Warmup
for _i in range(5):
    _ = forward_fn()

# Timing
torch.cuda.synchronize()
start = time.time()
iterations = 50
for _i in range(iterations):
    output = forward_fn()
torch.cuda.synchronize()
end = time.time()

output.sum().backward()
torch.cuda.synchronize()

torch.save(
    {
        "output": output.detach().cpu(),
        "grads": {
            name: tensor.grad.detach().cpu()
            for name, tensor in grad_tensors.items()
            if tensor.grad is not None
        },
    },
    out_path,
)
print(json.dumps({"avg_time": (end - start) / iterations}))
