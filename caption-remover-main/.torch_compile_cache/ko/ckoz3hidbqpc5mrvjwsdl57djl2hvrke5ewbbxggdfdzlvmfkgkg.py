
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


# kernel path: /workspace/caption-remover-main/src/stages/inpaint/../../../.torch_compile_cache/i2/ci2v2khhhpbxve3motrhacupwiwmsdhbzxu7nz7qv74d5f7d5mk6.py
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
    size_hints=[67108864], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp16', 1: '*fp16', 2: 'i32', 3: 'i32', 4: 'i32', 5: 'i32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy__unsafe_index_0', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': 'a1fa794fabf29fbd973066d9ab65d9c16aea0ed2e098514ffda941ec2a0f87ad'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ks0, ks1, ks2, ks3, ks4, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // ks1) % ks0
    x0 = xindex % ks1
    x2 = (xindex // ks4)
    x3 = xindex
    tmp0 = x1
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 1.0
    tmp3 = tmp1 * tmp2
    tmp4 = 0.0
    tmp5 = tmp3 + tmp4
    tmp6 = tmp5 + tmp4
    tmp7 = ks2*(1/ks0)
    tmp8 = tmp7.to(tl.float32)
    tmp9 = tmp6 * tmp8
    tmp10 = tmp9.to(tl.int64)
    tmp11 = x0
    tmp12 = tmp11.to(tl.float32)
    tmp13 = tmp12 * tmp2
    tmp14 = tmp13 + tmp4
    tmp15 = tmp14 + tmp4
    tmp16 = ks3*(1/ks1)
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tmp15 * tmp17
    tmp19 = tmp18.to(tl.int64)
    tmp20 = tl.load(in_ptr0 + (tmp19 + (ks3*tmp10) + (ks2*ks3*x2)), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp21 = tmp20.to(tl.float32)
    tmp22 = tmp21.to(tl.float32)
    tl.store(out_ptr0 + (x3), tmp22, xmask)
''', device_str='cuda')

import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid, split_scan_grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_raw_stream


# kernel path: /workspace/caption-remover-main/src/stages/inpaint/../../../.torch_compile_cache/tq/ctq4bsafsxv4mc34xsjmvkxbqx27tvgnci3ifyxhdbzxvspezojy.py
# Source Nodes: [mul, result, result_1], Original ATen: [aten.add, aten.convolution, aten.mul]
# mul => mul_4
# result => convolution
# result_1 => add_4
triton_poi_fused_add_convolution_mul_1 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.pointwise(
    size_hints=[67108864], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_mul_1', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'backend_hash': 'a1fa794fabf29fbd973066d9ab65d9c16aea0ed2e098514ffda941ec2a0f87ad'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, ks0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // ks0) % 640
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp3 = tl.load(in_ptr1 + (x3), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp4 = 0.125
    tmp5 = tmp3 * tmp4
    tmp6 = tmp2 + tmp5
    tl.store(in_out_ptr0 + (x3), tmp6, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1 = args
    args.clear()
    s1 = arg4_1
    s2 = arg5_1
    s3 = arg7_1
    s4 = arg8_1
    assert_size_stride(arg0_1, (640, 640, 3, 3), (5760, 9, 3, 1))
    assert_size_stride(arg1_1, (640, ), (1, ))
    assert_size_stride(arg2_1, (64, 640, 3, 3), (5760, 9, 3, 1))
    assert_size_stride(arg3_1, (640, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(arg6_1, (22, 640, s1, s2), (640*s1*s2, s1*s2, s2, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        ps0 = s3*s4
        buf0 = empty_strided_cuda((22, 640, s3, s4), (640*s3*s4, s3*s4, s4, 1), torch.float16)
        # Source Nodes: [hidden_states], Original ATen: [aten._to_copy, aten._unsafe_index]
        triton_poi_fused__to_copy__unsafe_index_0_xnumel = 14080*s3*s4
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy__unsafe_index_0.run(arg6_1, buf0, s3, s4, s1, s2, ps0, triton_poi_fused__to_copy__unsafe_index_0_xnumel, grid=grid(triton_poi_fused__to_copy__unsafe_index_0_xnumel), stream=stream0)
        del arg6_1
        # Source Nodes: [result], Original ATen: [aten.convolution]
        buf1 = extern_kernels.convolution(buf0, arg0_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1, (22, 640, s3, s4), (640*s3*s4, s3*s4, s4, 1))
        del arg0_1
        # Source Nodes: [l__self___conv_lora_a_default_0], Original ATen: [aten.convolution]
        buf2 = extern_kernels.convolution(buf0, arg2_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf2, (22, 64, s3, s4), (64*s3*s4, s3*s4, s4, 1))
        del arg2_1
        del buf0
        # Source Nodes: [l__self___conv_lora_b_default_0], Original ATen: [aten.convolution]
        buf3 = extern_kernels.convolution(buf2, arg3_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf3, (22, 640, s3, s4), (640*s3*s4, s3*s4, s4, 1))
        del arg3_1
        del buf2
        buf4 = buf1; del buf1  # reuse
        # Source Nodes: [mul, result, result_1], Original ATen: [aten.add, aten.convolution, aten.mul]
        triton_poi_fused_add_convolution_mul_1_xnumel = 14080*s3*s4
        triton_poi_fused_add_convolution_mul_1.run(buf4, arg1_1, buf3, ps0, triton_poi_fused_add_convolution_mul_1_xnumel, grid=grid(triton_poi_fused_add_convolution_mul_1_xnumel), stream=stream0)
        del arg1_1
        del buf3
    return (buf4, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((640, 640, 3, 3), (5760, 9, 3, 1), device='cuda:0', dtype=torch.float16)
    arg1_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg2_1 = rand_strided((64, 640, 3, 3), (5760, 9, 3, 1), device='cuda:0', dtype=torch.float16)
    arg3_1 = rand_strided((640, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float16)
    arg4_1 = 15
    arg5_1 = 45
    arg6_1 = rand_strided((22, 640, 15, 45), (432000, 675, 45, 1), device='cuda:0', dtype=torch.float16)
    arg7_1 = 30
    arg8_1 = 90
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
