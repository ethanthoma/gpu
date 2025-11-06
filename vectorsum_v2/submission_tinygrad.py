#!POPCORN leaderboard vectorsum_v2
try:
    import tinygrad

    _ = tinygrad.Tensor.custom_kernel
except (ImportError, AttributeError):
    import subprocess
    import sys

    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "--force-reinstall", "git+https://github.com/tinygrad/tinygrad.git"]
    )
    import tinygrad

import math

import torch
from task import input_t, output_t
from tinygrad import Context, Tensor, UOp, dtypes
from tinygrad.uop.ops import AddrSpace, AxisType, KernelInfo, Ops

THREADS_PER_BLOCK = 256
WARP_SIZE = 32
BLOCK_SIZE = 256


def sum_kernel(B: UOp, A: UOp) -> UOp:
    sdata = UOp(op=Ops.DEFINE_LOCAL, dtype=dtypes.float32.ptr(BLOCK_SIZE, AddrSpace.LOCAL), arg=0)

    n_blocks = math.ceil(A.shape[0] / BLOCK_SIZE)
    tid = UOp.special(THREADS_PER_BLOCK, "lidx0")
    block_idx_x = UOp.special(n_blocks, "gidx0")
    gid = block_idx_x * BLOCK_SIZE + tid

    valid = gid < A.shape[0]
    value = valid.where(A[gid], 0.0)
    sdata = sdata[tid].set(value)
    sdata = sdata.after(sdata.barrier())

    sid = UOp.range(math.ceil(math.log2(BLOCK_SIZE)), 0, AxisType.LOOP)
    s = 1 << sid
    index = s.mul(2).mul(tid)
    valid_access = (index + s) < BLOCK_SIZE
    index_valid = index.valid(valid_access)
    load = sdata.after(sid)[index_valid]
    new_val = load + sdata[(index + s).valid(valid_access)]
    val = sdata[index_valid].set(new_val)
    val = sdata.after(val.barrier()).end(sid)

    final_sum = sdata.after(val)[0]
    B = B[block_idx_x].set(final_sum)

    return B.sink(arg=KernelInfo(name="sum_kernel", opts_to_apply=()))


def custom_kernel(data: input_t) -> output_t:
    input, output = data
    A = Tensor.from_blob(input.data_ptr(), dtype=dtypes.float32, shape=input.shape)
    n_blocks = math.ceil(A.shape[0] / BLOCK_SIZE)
    B = Tensor.empty((n_blocks,), dtype=dtypes.float32)

    with Context(DEBUG=0):
        Tensor.realize(A, B)

    B = Tensor.custom_kernel(B, A, fxn=sum_kernel)[0].sum()
    B.realize()

    with Context(DEBUG=0):
        return torch.from_numpy(B.numpy()).to(output.device)
