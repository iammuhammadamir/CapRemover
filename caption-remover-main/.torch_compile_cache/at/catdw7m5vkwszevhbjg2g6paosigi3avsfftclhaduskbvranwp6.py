
from ctypes import c_void_p, c_long
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align

from torch import device, empty_strided
from torch._inductor.codecache import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels
from torch._inductor.codegen.multi_kernel import MultiKernelCall

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
alloc_from_pool = torch.ops.inductor._alloc_from_pool
reinterpret_tensor = torch.ops.inductor._reinterpret_tensor
async_compile = AsyncCompile()


# kernel path: /workspace/caption-remover-main/src/stages/inpaint/../../../.torch_compile_cache/cn/ccnewn4q2tc46l3ct2lma6u7fqrw7ptegcxlzgidetgydbnaqnhu.py
# Source Nodes: [sample_1], Original ATen: [aten.silu]
# sample_1 => convert_element_type_3, convert_element_type_4, mul, sigmoid
triton_poi_fused_silu_0 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.pointwise(
    size_hints=[2048], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp16', 1: '*fp16', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_silu_0', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'backend_hash': 'a1fa794fabf29fbd973066d9ab65d9c16aea0ed2e098514ffda941ec2a0f87ad'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1280
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), xmask).to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (x0), xmask).to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp2.to(tl.float32)
    tmp4 = tl.sigmoid(tmp3)
    tmp5 = tmp3 * tmp4
    tmp6 = tmp5.to(tl.float32)
    tl.store(in_out_ptr0 + (x0), tmp6, xmask)
''', device_str='cuda')

import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid, split_scan_grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_raw_stream


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1 = args
    args.clear()
    assert_size_stride(arg0_1, (1280, 320), (320, 1))
    assert_size_stride(arg1_1, (1280, ), (1, ))
    assert_size_stride(arg2_1, (1280, 1280), (1280, 1))
    assert_size_stride(arg3_1, (1280, ), (1, ))
    assert_size_stride(arg4_1, (1, 320), (320, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((1, 1280), (1280, 1), torch.float16)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(arg4_1, reinterpret_tensor(arg0_1, (320, 1280), (1, 320), 0), out=buf0)
        del arg0_1
        del arg4_1
        buf1 = buf0; del buf0  # reuse
        # Source Nodes: [sample_1], Original ATen: [aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_silu_0.run(buf1, arg1_1, 1280, grid=grid(1280), stream=stream0)
        del arg1_1
        buf2 = empty_strided_cuda((1, 1280), (1280, 1), torch.float16)
        # Source Nodes: [sample_1, sample_2], Original ATen: [aten.addmm, aten.silu]
        extern_kernels.addmm(arg3_1, buf1, reinterpret_tensor(arg2_1, (1280, 1280), (1, 1280), 0), alpha=1, beta=1, out=buf2)
        del arg2_1
        del arg3_1
        del buf1
    return (buf2, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((1280, 320), (320, 1), device='cuda:0', dtype=torch.float16)
    arg1_1 = rand_strided((1280, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg2_1 = rand_strided((1280, 1280), (1280, 1), device='cuda:0', dtype=torch.float16)
    arg3_1 = rand_strided((1280, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg4_1 = rand_strided((1, 320), (320, 1), device='cuda:0', dtype=torch.float16)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
