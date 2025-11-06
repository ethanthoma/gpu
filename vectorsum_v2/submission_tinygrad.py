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

THREADS_PER_BLOCK = 128
WARP_SIZE = 32
BLOCK_SIZE = 128


def sum_kernel(B: UOp, A: UOp) -> UOp:
    sdata = UOp(op=Ops.DEFINE_LOCAL, dtype=dtypes.float32.ptr(BLOCK_SIZE, AddrSpace.LOCAL), arg=0)

    tid = UOp.special(THREADS_PER_BLOCK, "lidx0")
    bid = UOp.special(math.ceil(A.shape[0] / BLOCK_SIZE), "gidx0")
    gid = bid * BLOCK_SIZE + tid

    valid = gid < A.shape[0]
    value = valid.where(A[gid], 0.0)
    sdata = sdata[tid].set(value)
    sdata = sdata.after(sdata.barrier())

    sid = UOp.range(math.ceil(math.log2(BLOCK_SIZE)), 0, AxisType.LOOP)
    s = 1 << sid
    cond = (tid.mod(s << 1)).eq(0)
    load = sdata.after(sid)[tid]
    new_val = load + sdata[tid + s]
    val = sdata[tid].set(cond.where(new_val, load))
    val = sdata.after(val.barrier()).end(sid)

    final_sum = sdata.after(val)[0]
    B = B[bid].set(final_sum)

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
