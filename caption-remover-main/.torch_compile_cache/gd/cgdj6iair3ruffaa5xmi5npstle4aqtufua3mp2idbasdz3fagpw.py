
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


# kernel path: /workspace/caption-remover-main/src/stages/inpaint/../../../.torch_compile_cache/7d/c7dsumkjfchobmyjofmdmauevjopqud3vgzz2db5tzvndwi7zcua.py
# Source Nodes: [l__self___conv_lora_a_default_0, result], Original ATen: [aten.convolution]
# l__self___conv_lora_a_default_0 => convolution_1
# result => convolution
triton_poi_fused_convolution_0 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.pointwise(
    size_hints=[8192, 4096], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_0', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': 'a1fa794fabf29fbd973066d9ab65d9c16aea0ed2e098514ffda941ec2a0f87ad'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 7040
    xnumel = 2700
    yoffset = tl.program_id(1) * (tl.program_id(2) + 1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 320
    y1 = (yindex // 320)
    tmp0 = tl.load(in_ptr0 + (x2 + (2700*y3)), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tl.store(out_ptr0 + (y0 + (320*x2) + (864000*y1)), tmp0, xmask & ymask)
    tl.store(out_ptr1 + (y0 + (320*x2) + (864000*y1)), tmp0, xmask & ymask)
''', device_str='cuda')

import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid, split_scan_grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_raw_stream


# kernel path: /workspace/caption-remover-main/src/stages/inpaint/../../../.torch_compile_cache/pk/cpk4j76nxnnpgjbttn2ud5j5bmhgvjcaaeij6ncna4iikcg5lyl5.py
# Source Nodes: [result], Original ATen: [aten.convolution]
# result => convolution
triton_poi_fused_convolution_1 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.pointwise(
    size_hints=[131072, 16], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp16', 1: '*fp16', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_1', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': 'a1fa794fabf29fbd973066d9ab65d9c16aea0ed2e098514ffda941ec2a0f87ad'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 102400
    xnumel = 9
    yoffset = tl.program_id(1) * (tl.program_id(2) + 1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 320
    y1 = (yindex // 320)
    tmp0 = tl.load(in_ptr0 + (x2 + (9*y3)), xmask, eviction_policy='evict_last').to(tl.float32)
    tl.store(out_ptr0 + (y0 + (320*x2) + (2880*y1)), tmp0, xmask)
''', device_str='cuda')


# kernel path: /workspace/caption-remover-main/src/stages/inpaint/../../../.torch_compile_cache/52/c524wmyzb3ubyrehlmtx3pdtbbcarn4bs72xy4feimabmrl6d6xx.py
# Source Nodes: [l__self___conv_lora_a_default_0], Original ATen: [aten.convolution]
# l__self___conv_lora_a_default_0 => convolution_1
triton_poi_fused_convolution_2 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.pointwise(
    size_hints=[32768, 16], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp16', 1: '*fp16', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_2', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': 'a1fa794fabf29fbd973066d9ab65d9c16aea0ed2e098514ffda941ec2a0f87ad'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 20480
    xnumel = 9
    yoffset = tl.program_id(1) * (tl.program_id(2) + 1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 320
    y1 = (yindex // 320)
    tmp0 = tl.load(in_ptr0 + (x2 + (9*y3)), xmask, eviction_policy='evict_last').to(tl.float32)
    tl.store(out_ptr0 + (y0 + (320*x2) + (2880*y1)), tmp0, xmask)
''', device_str='cuda')


# kernel path: /workspace/caption-remover-main/src/stages/inpaint/../../../.torch_compile_cache/fl/cflmorxwram7b53hs73bzvriztaxm7ly3y2s7goaejppuevdgnlk.py
# Source Nodes: [mul, result, result_1], Original ATen: [aten.add, aten.convolution, aten.mul]
# mul => mul
# result => convolution
# result_1 => add
triton_poi_fused_add_convolution_mul_3 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.pointwise(
    size_hints=[16384, 512], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_mul_3', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': 'a1fa794fabf29fbd973066d9ab65d9c16aea0ed2e098514ffda941ec2a0f87ad'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 14850
    xnumel = 320
    yoffset = tl.program_id(1) * (tl.program_id(2) + 1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 675
    y1 = (yindex // 675)
    tmp0 = tl.load(in_ptr0 + (x2 + (320*y3)), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x2), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp3 = tl.load(in_ptr2 + (x2 + (320*y3)), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp4 = 0.125
    tmp5 = tmp3 * tmp4
    tmp6 = tmp2 + tmp5
    tl.store(out_ptr0 + (y0 + (675*x2) + (216000*y1)), tmp6, xmask & ymask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1 = args
    args.clear()
    assert_size_stride(arg0_1, (320, 320, 3, 3), (2880, 9, 3, 1))
    assert_size_stride(arg1_1, (320, ), (1, ))
    assert_size_stride(arg2_1, (64, 320, 3, 3), (2880, 9, 3, 1))
    assert_size_stride(arg3_1, (320, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(arg4_1, (22, 320, 30, 90), (864000, 2700, 90, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((22, 320, 30, 90), (864000, 1, 28800, 320), torch.float16)
        buf3 = empty_strided_cuda((22, 320, 30, 90), (864000, 1, 28800, 320), torch.float16)
        # Source Nodes: [l__self___conv_lora_a_default_0, result], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_0.run(arg4_1, buf0, buf3, 7040, 2700, grid=grid(7040, 2700), stream=stream0)
        del arg4_1
        buf1 = empty_strided_cuda((320, 320, 3, 3), (2880, 1, 960, 320), torch.float16)
        # Source Nodes: [result], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_1.run(arg0_1, buf1, 102400, 9, grid=grid(102400, 9), stream=stream0)
        del arg0_1
        # Source Nodes: [result], Original ATen: [aten.convolution]
        buf2 = extern_kernels.convolution(buf0, buf1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf2, (22, 320, 15, 45), (216000, 1, 14400, 320))
        del buf0
        del buf1
        buf4 = empty_strided_cuda((64, 320, 3, 3), (2880, 1, 960, 320), torch.float16)
        # Source Nodes: [l__self___conv_lora_a_default_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_2.run(arg2_1, buf4, 20480, 9, grid=grid(20480, 9), stream=stream0)
        del arg2_1
        # Source Nodes: [l__self___conv_lora_a_default_0], Original ATen: [aten.convolution]
        buf5 = extern_kernels.convolution(buf3, buf4, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf5, (22, 64, 15, 45), (43200, 1, 2880, 64))
        del buf3
        del buf4
        # Source Nodes: [l__self___conv_lora_b_default_0], Original ATen: [aten.convolution]
        buf6 = extern_kernels.convolution(buf5, arg3_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf6, (22, 320, 15, 45), (216000, 1, 14400, 320))
        del arg3_1
        del buf5
        buf7 = empty_strided_cuda((22, 320, 15, 45), (216000, 675, 45, 1), torch.float16)
        # Source Nodes: [mul, result, result_1], Original ATen: [aten.add, aten.convolution, aten.mul]
        triton_poi_fused_add_convolution_mul_3.run(buf2, arg1_1, buf6, buf7, 14850, 320, grid=grid(14850, 320), stream=stream0)
        del arg1_1
        del buf2
        del buf6
    return (buf7, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((320, 320, 3, 3), (2880, 9, 3, 1), device='cuda:0', dtype=torch.float16)
    arg1_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg2_1 = rand_strided((64, 320, 3, 3), (2880, 9, 3, 1), device='cuda:0', dtype=torch.float16)
    arg3_1 = rand_strided((320, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float16)
    arg4_1 = rand_strided((22, 320, 30, 90), (864000, 2700, 90, 1), device='cuda:0', dtype=torch.float16)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
