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

import ctypes
import math

import torch
from task import input_t, output_t
from tinygrad import Context, Tensor, UOp, dtypes
from tinygrad.uop.ops import AddrSpace, AxisType, KernelInfo, Ops

THREADS_PER_BLOCK = 256 * 4
WARP_SIZE = 32
BLOCK_SIZE = 256 * 4


def torch_to_tinygrad(torch_tensor: torch.Tensor) -> Tensor:
    return Tensor.from_blob(torch_tensor.data_ptr(), dtype=dtypes.float32, shape=torch_tensor.shape)


def tinygrad_to_torch(tg_tensor: Tensor, dtype: torch.dtype = torch.float32, shape: tuple = ()) -> torch.Tensor:
    result = torch.empty(shape, dtype=dtype, device="cuda")
    tg_buf = tg_tensor.uop.base.buffer

    cudart = ctypes.CDLL("libcudart.so")
    cudart.cudaMemcpy(
        ctypes.c_void_p(result.data_ptr()),
        ctypes.c_void_p(tg_buf._buf.value),
        ctypes.c_size_t(tg_buf.nbytes),
        ctypes.c_int(3),
    )

    return result


def sum_kernel(B: UOp, A: UOp) -> UOp:
    sdata = UOp(op=Ops.DEFINE_LOCAL, dtype=dtypes.float32.ptr(BLOCK_SIZE, AddrSpace.LOCAL), arg=0)

    n_blocks = math.ceil(A.shape[0] / (BLOCK_SIZE * 2))
    tid = UOp.special(THREADS_PER_BLOCK, "lidx0")
    block_idx_x = UOp.special(n_blocks, "gidx0")
    gid = block_idx_x * BLOCK_SIZE * 2 + tid

    value = A[gid.valid(gid < A.shape[0])] + A[(gid + BLOCK_SIZE).valid(gid + BLOCK_SIZE < A.shape[0])]
    sdata = sdata[tid].set(value)
    sdata = sdata.after(sdata.barrier())

    sid = UOp.range(math.ceil(math.log2(BLOCK_SIZE)), 0, AxisType.LOOP)
    s = (BLOCK_SIZE // 2) >> sid

    cond = tid < s
    load = sdata.after(sid)[tid.valid(cond)]
    val = load + sdata[(tid + s).valid(cond)]
    on_cond = sdata[tid.valid(cond)].set(val)

    val = sdata.after(on_cond.barrier()).end(sid)

    final_sum = sdata.after(val)[0]
    B = B[block_idx_x].set(final_sum)

    return B.sink(arg=KernelInfo(name="sum_kernel", opts_to_apply=()))


def custom_kernel(data: input_t) -> output_t:
    input, output = data
    A = torch_to_tinygrad(input)
    n_blocks = math.ceil(A.shape[0] / (BLOCK_SIZE * 2))
    B = Tensor.empty((n_blocks,), dtype=dtypes.float32)

    print(A.shape[0], n_blocks)

    with Context(DEBUG=0):
        Tensor.realize(A, B)

    B = Tensor.custom_kernel(B, A, fxn=sum_kernel)[0].sum()
    B.realize()

    with Context(DEBUG=0):
        return tinygrad_to_torch(B, dtype=torch.float32, shape=())
