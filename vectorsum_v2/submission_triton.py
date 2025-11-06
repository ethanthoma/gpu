#!POPCORN leaderboard vectorsum_v2

import torch
import triton
import triton.language as tl
from task import input_t, output_t


@triton.jit
def sum_kernel(
    x_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0, eviction_policy="evict_first")
    block_sum = tl.sum(x, axis=0)
    tl.store(x_ptr + pid, block_sum)


def custom_kernel(data: input_t) -> output_t:
    input, output = data
    n_elements = input.numel()

    if n_elements >= 10_000_000:
        BLOCK_SIZE = 8192
    elif n_elements >= 1_000_000:
        BLOCK_SIZE = 4096
    elif n_elements >= 100_000:
        BLOCK_SIZE = 2048
    elif n_elements >= 10_000:
        BLOCK_SIZE = 1024
    else:
        BLOCK_SIZE = 512

    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    n_blocks = triton.cdiv(n_elements, BLOCK_SIZE)
    grid = (n_blocks,)

    sum_kernel[grid](input, n_elements, BLOCK_SIZE=BLOCK_SIZE)

    return input[:n_blocks].sum()
