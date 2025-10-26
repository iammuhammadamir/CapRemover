
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


# kernel path: /workspace/caption-remover-main/src/stages/inpaint/../../../.torch_compile_cache/p6/cp673lhy53oajer3dw2fsjzzwhu2v2cjo2cnsti2cbekek2ccblq.py
# Source Nodes: [hidden_states], Original ATen: [aten._to_copy, aten._unsafe_index]
# hidden_states => _unsafe_index, convert_element_type, convert_element_type_5
triton_poi_fused__to_copy__unsafe_index_0 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.pointwise(
    size_hints=[8388608], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp16', 1: '*fp16', 2: 'i32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy__unsafe_index_0', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': 'a1fa794fabf29fbd973066d9ab65d9c16aea0ed2e098514ffda941ec2a0f87ad'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ks0, ks1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 5181440
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 29440) % 8
    x1 = (xindex // 1280) % 23
    x0 = xindex % 1280
    x3 = (xindex // 235520)
    x4 = xindex
    tmp0 = x2
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 1.0
    tmp3 = tmp1 * tmp2
    tmp4 = 0.0
    tmp5 = tmp3 + tmp4
    tmp6 = tmp5 + tmp4
    tmp7 = (1/8)*ks0
    tmp8 = tmp7.to(tl.float32)
    tmp9 = tmp6 * tmp8
    tmp10 = tmp9.to(tl.int64)
    tmp11 = x1
    tmp12 = tmp11.to(tl.float32)
    tmp13 = tmp12 * tmp2
    tmp14 = tmp13 + tmp4
    tmp15 = tmp14 + tmp4
    tmp16 = (1/23)*ks1
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tmp15 * tmp17
    tmp19 = tmp18.to(tl.int64)
    tmp20 = tl.load(in_ptr0 + (tmp19 + (ks1*tmp10) + (ks0*ks1*x0) + (1280*ks0*ks1*x3)), None, eviction_policy='evict_last').to(tl.float32)
    tmp21 = tmp20.to(tl.float32)
    tmp22 = tmp21.to(tl.float32)
    tl.store(out_ptr0 + (x4), tmp22, None)
''', device_str='cuda')

import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid, split_scan_grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_raw_stream


# kernel path: /workspace/caption-remover-main/src/stages/inpaint/../../../.torch_compile_cache/27/c276moojhljonmz5yo6mn23eqzalakxiumpalfrdmzsn2p7tgbup.py
# Source Nodes: [hidden_states, result], Original ATen: [aten._to_copy, aten._unsafe_index, aten.convolution]
# hidden_states => _unsafe_index, convert_element_type, convert_element_type_5
# result => convolution
triton_poi_fused__to_copy__unsafe_index_convolution_1 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.pointwise(
    size_hints=[2097152, 16], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp16', 1: '*fp16', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy__unsafe_index_convolution_1', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': 'a1fa794fabf29fbd973066d9ab65d9c16aea0ed2e098514ffda941ec2a0f87ad'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1638400
    xnumel = 9
    yoffset = tl.program_id(1) * (tl.program_id(2) + 1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 1280
    y1 = (yindex // 1280)
    tmp0 = tl.load(in_ptr0 + (x2 + (9*y3)), xmask, eviction_policy='evict_last').to(tl.float32)
    tl.store(out_ptr0 + (y0 + (1280*x2) + (11520*y1)), tmp0, xmask)
''', device_str='cuda')


# kernel path: /workspace/caption-remover-main/src/stages/inpaint/../../../.torch_compile_cache/pj/cpjq6czc56ktomq27pdxbs27wrtitebu2iqsyn27gwmvuo2f3acy.py
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
    size_hints=[131072, 16], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp16', 1: '*fp16', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_2', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': 'a1fa794fabf29fbd973066d9ab65d9c16aea0ed2e098514ffda941ec2a0f87ad'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 81920
    xnumel = 9
    yoffset = tl.program_id(1) * (tl.program_id(2) + 1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 1280
    y1 = (yindex // 1280)
    tmp0 = tl.load(in_ptr0 + (x2 + (9*y3)), xmask, eviction_policy='evict_last').to(tl.float32)
    tl.store(out_ptr0 + (y0 + (1280*x2) + (11520*y1)), tmp0, xmask)
''', device_str='cuda')


# kernel path: /workspace/caption-remover-main/src/stages/inpaint/../../../.torch_compile_cache/hh/chh7iszk6ffhaazzfhwqeazairgxtcdh6ogpnu4oipif47wazn2t.py
# Source Nodes: [hidden_states, mul, result, result_1], Original ATen: [aten._to_copy, aten._unsafe_index, aten.add, aten.convolution, aten.mul]
# hidden_states => _unsafe_index, convert_element_type, convert_element_type_5
# mul => mul_4
# result => convolution
# result_1 => add_4
triton_poi_fused__to_copy__unsafe_index_add_convolution_mul_3 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.pointwise(
    size_hints=[4096, 2048], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy__unsafe_index_add_convolution_mul_3', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': 'a1fa794fabf29fbd973066d9ab65d9c16aea0ed2e098514ffda941ec2a0f87ad'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 4048
    xnumel = 1280
    yoffset = tl.program_id(1) * (tl.program_id(2) + 1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 184
    y1 = (yindex // 184)
    tmp0 = tl.load(in_ptr0 + (x2 + (1280*y3)), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x2), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp3 = tl.load(in_ptr2 + (x2 + (1280*y3)), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp4 = 0.125
    tmp5 = tmp3 * tmp4
    tmp6 = tmp2 + tmp5
    tl.store(out_ptr0 + (y0 + (184*x2) + (235520*y1)), tmp6, xmask & ymask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1 = args
    args.clear()
    s0 = arg4_1
    s1 = arg5_1
    assert_size_stride(arg0_1, (1280, 1280, 3, 3), (11520, 9, 3, 1))
    assert_size_stride(arg1_1, (1280, ), (1, ))
    assert_size_stride(arg2_1, (64, 1280, 3, 3), (11520, 9, 3, 1))
    assert_size_stride(arg3_1, (1280, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(arg6_1, (22, 1280, s0, s1), (1280*s0*s1, s0*s1, s1, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((22, 1280, 8, 23), (235520, 1, 29440, 1280), torch.float16)
        # Source Nodes: [hidden_states], Original ATen: [aten._to_copy, aten._unsafe_index]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy__unsafe_index_0.run(arg6_1, buf0, s0, s1, 5181440, grid=grid(5181440), stream=stream0)
        del arg6_1
        buf1 = empty_strided_cuda((1280, 1280, 3, 3), (11520, 1, 3840, 1280), torch.float16)
        # Source Nodes: [hidden_states, result], Original ATen: [aten._to_copy, aten._unsafe_index, aten.convolution]
        triton_poi_fused__to_copy__unsafe_index_convolution_1.run(arg0_1, buf1, 1638400, 9, grid=grid(1638400, 9), stream=stream0)
        del arg0_1
        # Source Nodes: [hidden_states, result], Original ATen: [aten._to_copy, aten._unsafe_index, aten.convolution]
        buf2 = extern_kernels.convolution(buf0, buf1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf2, (22, 1280, 8, 23), (235520, 1, 29440, 1280))
        del buf1
        buf3 = empty_strided_cuda((64, 1280, 3, 3), (11520, 1, 3840, 1280), torch.float16)
        # Source Nodes: [l__self___conv_lora_a_default_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_2.run(arg2_1, buf3, 81920, 9, grid=grid(81920, 9), stream=stream0)
        del arg2_1
        # Source Nodes: [l__self___conv_lora_a_default_0], Original ATen: [aten.convolution]
        buf4 = extern_kernels.convolution(buf0, buf3, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf4, (22, 64, 8, 23), (11776, 1, 1472, 64))
        del buf3
        # Source Nodes: [l__self___conv_lora_b_default_0], Original ATen: [aten.convolution]
        buf5 = extern_kernels.convolution(buf4, arg3_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf5, (22, 1280, 8, 23), (235520, 1, 29440, 1280))
        del arg3_1
        del buf4
        buf6 = reinterpret_tensor(buf0, (22, 1280, 8, 23), (235520, 184, 23, 1), 0); del buf0  # reuse
        # Source Nodes: [hidden_states, mul, result, result_1], Original ATen: [aten._to_copy, aten._unsafe_index, aten.add, aten.convolution, aten.mul]
        triton_poi_fused__to_copy__unsafe_index_add_convolution_mul_3.run(buf2, arg1_1, buf5, buf6, 4048, 1280, grid=grid(4048, 1280), stream=stream0)
        del arg1_1
        del buf2
        del buf5
    return (buf6, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((1280, 1280, 3, 3), (11520, 9, 3, 1), device='cuda:0', dtype=torch.float16)
    arg1_1 = rand_strided((1280, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg2_1 = rand_strided((64, 1280, 3, 3), (11520, 9, 3, 1), device='cuda:0', dtype=torch.float16)
    arg3_1 = rand_strided((1280, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float16)
    arg4_1 = 4
    arg5_1 = 12
    arg6_1 = rand_strided((22, 1280, 4, 12), (61440, 48, 12, 1), device='cuda:0', dtype=torch.float16)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
