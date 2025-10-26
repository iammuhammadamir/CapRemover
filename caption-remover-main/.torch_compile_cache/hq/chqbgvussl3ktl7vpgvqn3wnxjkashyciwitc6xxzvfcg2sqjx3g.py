
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


# kernel path: /workspace/caption-remover-main/src/stages/inpaint/../../../.torch_compile_cache/kz/ckzsbhwm4cwoc22utokix4sfchh5t6sl6v4gnh453d7lebvzj4mz.py
# Source Nodes: [hidden_states_1], Original ATen: [aten.native_group_norm]
# hidden_states_1 => convert_element_type, var_mean
triton_red_fused_native_group_norm_0 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.reduction(
    size_hints=[1024, 16384],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_group_norm_0', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': 'a1fa794fabf29fbd973066d9ab65d9c16aea0ed2e098514ffda941ec2a0f87ad'}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, ks0, ks1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 704
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    tmp3_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp3_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp3_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (80*ks0*ks1*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp2 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
        tmp3_mean_next, tmp3_m2_next, tmp3_weight_next = triton_helpers.welford_reduce(
            tmp2, tmp3_mean, tmp3_m2, tmp3_weight, roffset == 0
        )
        tmp3_mean = tl.where(rmask & xmask, tmp3_mean_next, tmp3_mean)
        tmp3_m2 = tl.where(rmask & xmask, tmp3_m2_next, tmp3_m2)
        tmp3_weight = tl.where(rmask & xmask, tmp3_weight_next, tmp3_weight)
    tmp3_tmp, tmp4_tmp, tmp5_tmp = triton_helpers.welford(
        tmp3_mean, tmp3_m2, tmp3_weight, 1
    )
    tmp3 = tmp3_tmp[:, None]
    tmp4 = tmp4_tmp[:, None]
    tmp5 = tmp5_tmp[:, None]
    tl.store(out_ptr0 + (x0), tmp3, xmask)
    tl.store(out_ptr1 + (x0), tmp4, xmask)
''', device_str='cuda')

import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid, split_scan_grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_raw_stream


# kernel path: /workspace/caption-remover-main/src/stages/inpaint/../../../.torch_compile_cache/uq/cuqhbd4iufu5opvizq5ranznfrqgdj2vqfsxcni3ifnpjeojprqp.py
# Source Nodes: [hidden_states_1, hidden_states_2], Original ATen: [aten.native_group_norm, aten.silu]
# hidden_states_1 => add_1, mul_2
# hidden_states_2 => convert_element_type_5, mul_3, sigmoid
triton_poi_fused_native_group_norm_silu_1 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.pointwise(
    size_hints=[16777216], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*fp16', 4: '*fp16', 5: '*fp16', 6: 'i32', 7: 'i32', 8: 'i32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 7, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_group_norm_silu_1', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': 'a1fa794fabf29fbd973066d9ab65d9c16aea0ed2e098514ffda941ec2a0f87ad'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, ks0, ks1, ks2, ks3, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // ks0) % 2560
    x2 = (xindex // ks1)
    tmp0 = tl.load(in_ptr0 + (x3), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tl.load(in_ptr1 + ((32*x2) + (x1 // 80)), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + ((32*x2) + (x1 // 80)), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp18 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 - tmp2
    tmp5 = 80*ks2*ks3
    tmp6 = tmp5.to(tl.float32)
    tmp7 = 0.0
    tmp8 = tmp6 - tmp7
    tmp9 = triton_helpers.maximum(tmp7, tmp8)
    tmp10 = tmp4 / tmp9
    tmp11 = 1e-05
    tmp12 = tmp10 + tmp11
    tmp13 = libdevice.rsqrt(tmp12)
    tmp14 = tmp3 * tmp13
    tmp16 = tmp15.to(tl.float32)
    tmp17 = tmp14 * tmp16
    tmp19 = tmp18.to(tl.float32)
    tmp20 = tmp17 + tmp19
    tmp21 = tl.sigmoid(tmp20)
    tmp22 = tmp20 * tmp21
    tmp23 = tmp22.to(tl.float32)
    tl.store(out_ptr1 + (x3), tmp23, xmask)
''', device_str='cuda')


# kernel path: /workspace/caption-remover-main/src/stages/inpaint/../../../.torch_compile_cache/ro/cromvrozmkque5os3ox7besajknq5mxo4uclafmynf4fhsajgpoj.py
# Source Nodes: [temb], Original ATen: [aten.silu]
# temb => convert_element_type_6, convert_element_type_7, mul_5, sigmoid_1
triton_poi_fused_silu_2 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.pointwise(
    size_hints=[32768], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp16', 1: '*fp16', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_silu_2', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': 'a1fa794fabf29fbd973066d9ab65d9c16aea0ed2e098514ffda941ec2a0f87ad'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 28160
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = tl.sigmoid(tmp1)
    tmp3 = tmp1 * tmp2
    tmp4 = tmp3.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''', device_str='cuda')


# kernel path: /workspace/caption-remover-main/src/stages/inpaint/../../../.torch_compile_cache/xd/cxdfotzcizqqusvzl6zcrzdystbnegyzkpbu35j7nzmkmlrzbtjl.py
# Source Nodes: [hidden_states_4, hidden_states_5, mul, result, result_1], Original ATen: [aten.add, aten.convolution, aten.mul, aten.native_group_norm]
# hidden_states_4 => add_4
# hidden_states_5 => convert_element_type_15, var_mean_1
# mul => mul_4
# result => convolution
# result_1 => add_2
triton_red_fused_add_convolution_mul_native_group_norm_3 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.reduction(
    size_hints=[1024, 8192],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: '*fp16', 5: '*fp16', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32', 10: 'i32', 11: 'i32', 12: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(11, 12))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_convolution_mul_native_group_norm_3', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'backend_hash': 'a1fa794fabf29fbd973066d9ab65d9c16aea0ed2e098514ffda941ec2a0f87ad'}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, ks0, ks1, ks2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 704
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x4 = xindex
    x0 = xindex % 32
    tmp16_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp16_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp16_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r5 = rindex
        r3 = (rindex // ks2)
        tmp0 = tl.load(in_out_ptr0 + (r5 + (40*ks0*ks1*x4)), rmask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp1 = tl.load(in_ptr0 + (r3 + (40*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp3 = tl.load(in_ptr1 + (r5 + (40*ks0*ks1*x4)), rmask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp7 = tl.load(in_ptr2 + (r3 + (40*x4)), rmask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp8 = tl.load(in_ptr3 + (r3 + (40*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp10 = tl.load(in_ptr4 + (r3 + (40*x4)), rmask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp2 = tmp0 + tmp1
        tmp4 = 0.125
        tmp5 = tmp3 * tmp4
        tmp6 = tmp2 + tmp5
        tmp9 = tmp7 + tmp8
        tmp11 = tmp10 * tmp4
        tmp12 = tmp9 + tmp11
        tmp13 = tmp6 + tmp12
        tmp14 = tmp13.to(tl.float32)
        tmp15 = tl.broadcast_to(tmp14, [XBLOCK, RBLOCK])
        tmp16_mean_next, tmp16_m2_next, tmp16_weight_next = triton_helpers.welford_reduce(
            tmp15, tmp16_mean, tmp16_m2, tmp16_weight, roffset == 0
        )
        tmp16_mean = tl.where(rmask & xmask, tmp16_mean_next, tmp16_mean)
        tmp16_m2 = tl.where(rmask & xmask, tmp16_m2_next, tmp16_m2)
        tmp16_weight = tl.where(rmask & xmask, tmp16_weight_next, tmp16_weight)
        tl.store(in_out_ptr0 + (r5 + (40*ks0*ks1*x4)), tmp13, rmask & xmask)
    tmp16_tmp, tmp17_tmp, tmp18_tmp = triton_helpers.welford(
        tmp16_mean, tmp16_m2, tmp16_weight, 1
    )
    tmp16 = tmp16_tmp[:, None]
    tmp17 = tmp17_tmp[:, None]
    tmp18 = tmp18_tmp[:, None]
    tl.store(out_ptr0 + (x4), tmp16, xmask)
    tl.store(out_ptr1 + (x4), tmp17, xmask)
''', device_str='cuda')


# kernel path: /workspace/caption-remover-main/src/stages/inpaint/../../../.torch_compile_cache/cm/ccm4t6vs7oidp5wujjwbwlzvkzb2pdtcsxf4pzxpai67de2z6kay.py
# Source Nodes: [hidden_states_5, hidden_states_6], Original ATen: [aten.native_group_norm, aten.silu]
# hidden_states_5 => add_6, mul_9
# hidden_states_6 => convert_element_type_20, mul_10, sigmoid_2
triton_poi_fused_native_group_norm_silu_4 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*fp16', 4: '*fp16', 5: '*fp16', 6: 'i32', 7: 'i32', 8: 'i32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 7, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_group_norm_silu_4', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': 'a1fa794fabf29fbd973066d9ab65d9c16aea0ed2e098514ffda941ec2a0f87ad'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, ks0, ks1, ks2, ks3, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // ks0) % 1280
    x2 = (xindex // ks1)
    tmp0 = tl.load(in_ptr0 + (x3), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tl.load(in_ptr1 + ((32*x2) + (x1 // 40)), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + ((32*x2) + (x1 // 40)), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp18 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 - tmp2
    tmp5 = 40*ks2*ks3
    tmp6 = tmp5.to(tl.float32)
    tmp7 = 0.0
    tmp8 = tmp6 - tmp7
    tmp9 = triton_helpers.maximum(tmp7, tmp8)
    tmp10 = tmp4 / tmp9
    tmp11 = 1e-05
    tmp12 = tmp10 + tmp11
    tmp13 = libdevice.rsqrt(tmp12)
    tmp14 = tmp3 * tmp13
    tmp16 = tmp15.to(tl.float32)
    tmp17 = tmp14 * tmp16
    tmp19 = tmp18.to(tl.float32)
    tmp20 = tmp17 + tmp19
    tmp21 = tl.sigmoid(tmp20)
    tmp22 = tmp20 * tmp21
    tmp23 = tmp22.to(tl.float32)
    tl.store(out_ptr1 + (x3), tmp23, xmask)
''', device_str='cuda')


# kernel path: /workspace/caption-remover-main/src/stages/inpaint/../../../.torch_compile_cache/us/cusm4gig7o4kstclmfh6unp2hj34oodjvgz25clybu5vvijl7lal.py
# Source Nodes: [add_5, mul_2, mul_3, output_tensor, result_10, result_6, result_7, result_9], Original ATen: [aten.add, aten.convolution, aten.div, aten.mul]
# add_5 => add_9
# mul_2 => mul_11
# mul_3 => mul_12
# output_tensor => div
# result_10 => add_8
# result_6 => convolution_3
# result_7 => add_7
# result_9 => convolution_6
triton_poi_fused_add_convolution_div_mul_5 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: '*fp16', 5: '*fp16', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_div_mul_5', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'backend_hash': 'a1fa794fabf29fbd973066d9ab65d9c16aea0ed2e098514ffda941ec2a0f87ad'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, ks0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // ks0) % 1280
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp3 = tl.load(in_ptr1 + (x3), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp7 = tl.load(in_ptr2 + (x3), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp8 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp10 = tl.load(in_ptr4 + (x3), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp4 = 0.125
    tmp5 = tmp3 * tmp4
    tmp6 = tmp2 + tmp5
    tmp9 = tmp7 + tmp8
    tmp11 = tmp10 * tmp4
    tmp12 = tmp9 + tmp11
    tmp13 = tmp6 + tmp12
    tmp14 = 1.0
    tmp15 = tmp13 / tmp14
    tl.store(in_out_ptr0 + (x3), tmp15, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1 = args
    args.clear()
    s1 = arg20_1
    s2 = arg21_1
    assert_size_stride(arg0_1, (2560, ), (1, ))
    assert_size_stride(arg1_1, (2560, ), (1, ))
    assert_size_stride(arg2_1, (1280, 2560, 3, 3), (23040, 9, 3, 1))
    assert_size_stride(arg3_1, (1280, ), (1, ))
    assert_size_stride(arg4_1, (64, 2560, 3, 3), (23040, 9, 3, 1))
    assert_size_stride(arg5_1, (1280, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(arg6_1, (1280, 1280), (1280, 1))
    assert_size_stride(arg7_1, (1280, ), (1, ))
    assert_size_stride(arg8_1, (64, 1280), (1280, 1))
    assert_size_stride(arg9_1, (1280, 64), (64, 1))
    assert_size_stride(arg10_1, (1280, ), (1, ))
    assert_size_stride(arg11_1, (1280, ), (1, ))
    assert_size_stride(arg12_1, (1280, 1280, 3, 3), (11520, 9, 3, 1))
    assert_size_stride(arg13_1, (1280, ), (1, ))
    assert_size_stride(arg14_1, (64, 1280, 3, 3), (11520, 9, 3, 1))
    assert_size_stride(arg15_1, (1280, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(arg16_1, (1280, 2560, 1, 1), (2560, 1, 1, 1))
    assert_size_stride(arg17_1, (1280, ), (1, ))
    assert_size_stride(arg18_1, (64, 2560, 1, 1), (2560, 1, 1, 1))
    assert_size_stride(arg19_1, (1280, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(arg22_1, (22, 2560, s1, s2), (2560*s1*s2, s1*s2, s2, 1))
    assert_size_stride(arg23_1, (22, 1280), (1280, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((22, 32, 1, 1), (32, 1, 704, 704), torch.float32)
        buf1 = empty_strided_cuda((22, 32, 1, 1), (32, 1, 704, 704), torch.float32)
        # Source Nodes: [hidden_states_1], Original ATen: [aten.native_group_norm]
        triton_red_fused_native_group_norm_0_rnumel = 80*s1*s2
        stream0 = get_raw_stream(0)
        triton_red_fused_native_group_norm_0.run(arg22_1, buf0, buf1, s1, s2, 704, triton_red_fused_native_group_norm_0_rnumel, grid=grid(704), stream=stream0)
        ps0 = s1*s2
        ps1 = 2560*s1*s2
        buf4 = empty_strided_cuda((22, 2560, s1, s2), (2560*s1*s2, s1*s2, s2, 1), torch.float16)
        # Source Nodes: [hidden_states_1, hidden_states_2], Original ATen: [aten.native_group_norm, aten.silu]
        triton_poi_fused_native_group_norm_silu_1_xnumel = 56320*s1*s2
        triton_poi_fused_native_group_norm_silu_1.run(arg22_1, buf0, buf1, arg0_1, arg1_1, buf4, ps0, ps1, s1, s2, triton_poi_fused_native_group_norm_silu_1_xnumel, grid=grid(triton_poi_fused_native_group_norm_silu_1_xnumel), stream=stream0)
        del arg0_1
        del arg1_1
        # Source Nodes: [result], Original ATen: [aten.convolution]
        buf5 = extern_kernels.convolution(buf4, arg2_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf5, (22, 1280, s1, s2), (1280*s1*s2, s1*s2, s2, 1))
        del arg2_1
        # Source Nodes: [l__self___conv1_lora_a_default_0], Original ATen: [aten.convolution]
        buf6 = extern_kernels.convolution(buf4, arg4_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf6, (22, 64, s1, s2), (64*s1*s2, s1*s2, s2, 1))
        del arg4_1
        del buf4
        # Source Nodes: [l__self___conv1_lora_b_default_0], Original ATen: [aten.convolution]
        buf7 = extern_kernels.convolution(buf6, arg5_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf7, (22, 1280, s1, s2), (1280*s1*s2, s1*s2, s2, 1))
        del arg5_1
        del buf6
        buf8 = empty_strided_cuda((22, 1280), (1280, 1), torch.float16)
        # Source Nodes: [temb], Original ATen: [aten.silu]
        triton_poi_fused_silu_2.run(arg23_1, buf8, 28160, grid=grid(28160), stream=stream0)
        del arg23_1
        buf9 = empty_strided_cuda((22, 1280), (1280, 1), torch.float16)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf8, reinterpret_tensor(arg6_1, (1280, 1280), (1, 1280), 0), out=buf9)
        del arg6_1
        buf10 = empty_strided_cuda((22, 64), (64, 1), torch.float16)
        # Source Nodes: [l__self___time_emb_proj_lora_a_default_0], Original ATen: [aten.mm]
        extern_kernels.mm(buf8, reinterpret_tensor(arg8_1, (1280, 64), (1, 1280), 0), out=buf10)
        del arg8_1
        buf11 = buf8; del buf8  # reuse
        # Source Nodes: [l__self___time_emb_proj_lora_b_default_0], Original ATen: [aten.mm]
        extern_kernels.mm(buf10, reinterpret_tensor(arg9_1, (64, 1280), (1, 64), 0), out=buf11)
        del arg9_1
        del buf10
        buf12 = buf5; del buf5  # reuse
        buf13 = buf1; del buf1  # reuse
        buf14 = buf0; del buf0  # reuse
        # Source Nodes: [hidden_states_4, hidden_states_5, mul, result, result_1], Original ATen: [aten.add, aten.convolution, aten.mul, aten.native_group_norm]
        triton_red_fused_add_convolution_mul_native_group_norm_3_rnumel = 40*s1*s2
        triton_red_fused_add_convolution_mul_native_group_norm_3.run(buf12, arg3_1, buf7, buf9, arg7_1, buf11, buf13, buf14, s1, s2, ps0, 704, triton_red_fused_add_convolution_mul_native_group_norm_3_rnumel, grid=grid(704), stream=stream0)
        del arg3_1
        del arg7_1
        del buf11
        del buf9
        # Source Nodes: [result_9], Original ATen: [aten.convolution]
        buf16 = extern_kernels.convolution(arg22_1, arg16_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf16, (22, 1280, s1, s2), (1280*s1*s2, s1*s2, s2, 1))
        del arg16_1
        # Source Nodes: [l__self___conv_shortcut_lora_a_default_0], Original ATen: [aten.convolution]
        buf17 = extern_kernels.convolution(arg22_1, arg18_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf17, (22, 64, s1, s2), (64*s1*s2, s1*s2, s2, 1))
        del arg18_1
        del arg22_1
        # Source Nodes: [l__self___conv_shortcut_lora_b_default_0], Original ATen: [aten.convolution]
        buf18 = extern_kernels.convolution(buf17, arg19_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf18, (22, 1280, s1, s2), (1280*s1*s2, s1*s2, s2, 1))
        del arg19_1
        del buf17
        ps2 = 1280*s1*s2
        buf20 = buf7; del buf7  # reuse
        # Source Nodes: [hidden_states_5, hidden_states_6], Original ATen: [aten.native_group_norm, aten.silu]
        triton_poi_fused_native_group_norm_silu_4_xnumel = 28160*s1*s2
        triton_poi_fused_native_group_norm_silu_4.run(buf12, buf13, buf14, arg10_1, arg11_1, buf20, ps0, ps2, s1, s2, triton_poi_fused_native_group_norm_silu_4_xnumel, grid=grid(triton_poi_fused_native_group_norm_silu_4_xnumel), stream=stream0)
        del arg10_1
        del arg11_1
        del buf12
        del buf13
        del buf14
        # Source Nodes: [result_6], Original ATen: [aten.convolution]
        buf21 = extern_kernels.convolution(buf20, arg12_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf21, (22, 1280, s1, s2), (1280*s1*s2, s1*s2, s2, 1))
        del arg12_1
        # Source Nodes: [l__self___conv2_lora_a_default_0], Original ATen: [aten.convolution]
        buf22 = extern_kernels.convolution(buf20, arg14_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf22, (22, 64, s1, s2), (64*s1*s2, s1*s2, s2, 1))
        del arg14_1
        del buf20
        # Source Nodes: [l__self___conv2_lora_b_default_0], Original ATen: [aten.convolution]
        buf23 = extern_kernels.convolution(buf22, arg15_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf23, (22, 1280, s1, s2), (1280*s1*s2, s1*s2, s2, 1))
        del arg15_1
        del buf22
        buf24 = buf16; del buf16  # reuse
        # Source Nodes: [add_5, mul_2, mul_3, output_tensor, result_10, result_6, result_7, result_9], Original ATen: [aten.add, aten.convolution, aten.div, aten.mul]
        triton_poi_fused_add_convolution_div_mul_5_xnumel = 28160*s1*s2
        triton_poi_fused_add_convolution_div_mul_5.run(buf24, arg17_1, buf18, buf21, arg13_1, buf23, ps0, triton_poi_fused_add_convolution_div_mul_5_xnumel, grid=grid(triton_poi_fused_add_convolution_div_mul_5_xnumel), stream=stream0)
        del arg13_1
        del arg17_1
        del buf18
        del buf21
        del buf23
    return (buf24, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg1_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg2_1 = rand_strided((1280, 2560, 3, 3), (23040, 9, 3, 1), device='cuda:0', dtype=torch.float16)
    arg3_1 = rand_strided((1280, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg4_1 = rand_strided((64, 2560, 3, 3), (23040, 9, 3, 1), device='cuda:0', dtype=torch.float16)
    arg5_1 = rand_strided((1280, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float16)
    arg6_1 = rand_strided((1280, 1280), (1280, 1), device='cuda:0', dtype=torch.float16)
    arg7_1 = rand_strided((1280, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg8_1 = rand_strided((64, 1280), (1280, 1), device='cuda:0', dtype=torch.float16)
    arg9_1 = rand_strided((1280, 64), (64, 1), device='cuda:0', dtype=torch.float16)
    arg10_1 = rand_strided((1280, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg11_1 = rand_strided((1280, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg12_1 = rand_strided((1280, 1280, 3, 3), (11520, 9, 3, 1), device='cuda:0', dtype=torch.float16)
    arg13_1 = rand_strided((1280, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg14_1 = rand_strided((64, 1280, 3, 3), (11520, 9, 3, 1), device='cuda:0', dtype=torch.float16)
    arg15_1 = rand_strided((1280, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float16)
    arg16_1 = rand_strided((1280, 2560, 1, 1), (2560, 1, 1, 1), device='cuda:0', dtype=torch.float16)
    arg17_1 = rand_strided((1280, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg18_1 = rand_strided((64, 2560, 1, 1), (2560, 1, 1, 1), device='cuda:0', dtype=torch.float16)
    arg19_1 = rand_strided((1280, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float16)
    arg20_1 = 8
    arg21_1 = 23
    arg22_1 = rand_strided((22, 2560, 8, 23), (471040, 184, 23, 1), device='cuda:0', dtype=torch.float16)
    arg23_1 = rand_strided((22, 1280), (1280, 1), device='cuda:0', dtype=torch.float16)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
