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

import torch
from task import input_t, output_t
from tinygrad import Context, Tensor, UOp, dtypes
from tinygrad.uop.ops import AddrSpace, AxisType, KernelInfo, Ops


def sum_kernel(B: UOp, A: UOp) -> UOp:
    acc = UOp(op=Ops.DEFINE_REG, dtype=dtypes.float64.ptr(1, AddrSpace.REG), arg=0)
    acc = acc[0].set(0)
    i = UOp.range(A.shape[0], 0, axis_type=AxisType.REDUCE)
    acc = acc.after(acc[0].store(acc[0] + A[i].cast(dtypes.float64)).end(i))
    out = B[0].set(acc[0]).cast(dtypes.float32)
    return out.sink(arg=KernelInfo(name="sum_kernel", opts_to_apply=()))


def custom_kernel(data: input_t) -> output_t:
    input, output = data
    A = Tensor.from_blob(input.data_ptr(), dtype=dtypes.float32, shape=input.shape)
    B = Tensor.empty((), dtype=dtypes.float64)
    with Context(DEBUG=0):
        Tensor.realize(A, B)

    B = Tensor.custom_kernel(B, A, fxn=sum_kernel)[0]
    B.realize()

    return torch.tensor(B.item(), device=input.device, dtype=torch.float32)
