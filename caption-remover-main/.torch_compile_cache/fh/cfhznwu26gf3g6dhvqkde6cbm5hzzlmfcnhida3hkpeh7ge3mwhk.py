
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


# kernel path: /workspace/caption-remover-main/src/stages/inpaint/../../../.torch_compile_cache/rz/crzbpbvacnahhooa6es7irs2i3p5cr6qnr7n2vu6o5zp2nmyzvie.py
# Source Nodes: [hidden_states_2], Original ATen: [aten.native_group_norm]
# hidden_states_2 => convert_element_type, var_mean
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
    size_hints=[1024, 65536],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_group_norm_0', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': 'a1fa794fabf29fbd973066d9ab65d9c16aea0ed2e098514ffda941ec2a0f87ad'}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, ks0, ks1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 32)
    x0 = xindex % 32
    tmp16_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp16_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp16_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (x1*((17 + (220*ks0*ks1)) // 18))
        tmp1 = 220*ks0*ks1
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + ((ks0*ks1*(((r2 + (x1*((17 + (220*ks0*ks1)) // 18))) // (22*ks0*ks1)) % 10)) + (10*ks0*ks1*x0) + (320*ks0*ks1*(((r2 + (x1*((17 + (220*ks0*ks1)) // 18))) % (22*ks0*ks1)) // (ks0*ks1))) + (((r2 + (x1*((17 + (220*ks0*ks1)) // 18))) % (22*ks0*ks1)) % (ks0*ks1))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp4 = tmp3.to(tl.float32)
        tmp5 = tl.full(tmp4.shape, 0, tmp4.dtype)
        tmp6 = tl.where(tmp2, tmp4, tmp5)
        tmp7 = 0.0
        tmp8 = tl.full(tmp7.shape, 0, tmp7.dtype)
        tmp9 = tl.where(tmp2, tmp7, tmp8)
        tmp10 = 1.0
        tmp11 = tl.full(tmp10.shape, 0, tmp10.dtype)
        tmp12 = tl.where(tmp2, tmp10, tmp11)
        tmp13 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
        tmp14 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
        tmp15 = tl.broadcast_to(tmp12, [XBLOCK, RBLOCK])
        tmp16_mean_next, tmp16_m2_next, tmp16_weight_next = triton_helpers.welford_combine(
            tmp16_mean, tmp16_m2, tmp16_weight,
            tmp13, tmp14, tmp15
        )
        tmp16_mean = tl.where(rmask & xmask, tmp16_mean_next, tmp16_mean)
        tmp16_m2 = tl.where(rmask & xmask, tmp16_m2_next, tmp16_m2)
        tmp16_weight = tl.where(rmask & xmask, tmp16_weight_next, tmp16_weight)
    tmp16_tmp, tmp17_tmp, tmp18_tmp = triton_helpers.welford(
        tmp16_mean, tmp16_m2, tmp16_weight, 1
    )
    tmp16 = tmp16_tmp[:, None]
    tmp17 = tmp17_tmp[:, None]
    tmp18 = tmp18_tmp[:, None]
    tl.store(out_ptr0 + (x3), tmp16, xmask)
    tl.store(out_ptr1 + (x3), tmp17, xmask)
    tl.store(out_ptr2 + (x3), tmp18, xmask)
''', device_str='cuda')

import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid, split_scan_grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_raw_stream


# kernel path: /workspace/caption-remover-main/src/stages/inpaint/../../../.torch_compile_cache/xu/cxuipnka3pv5je3dzh4v7jl6y435phrfdgzqcawitbvhrrl2cgwo.py
# Source Nodes: [hidden_states_2], Original ATen: [aten.native_group_norm]
# hidden_states_2 => convert_element_type, var_mean
triton_per_fused_native_group_norm_1 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.persistent_reduction(
    size_hints=[32, 32],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_group_norm_1', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': 'a1fa794fabf29fbd973066d9ab65d9c16aea0ed2e098514ffda941ec2a0f87ad'}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 32
    rnumel = 18
    RBLOCK: tl.constexpr = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (32*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (32*r1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + (32*r1)), rmask & xmask, other=0.0)
    tmp3 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp5 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp7 = tl.where(rmask & xmask, tmp3, 0)
    tmp8 = tl.where(rmask & xmask, tmp4, 0)
    tmp9 = tl.where(rmask & xmask, tmp5, 0)
    tmp10, tmp11, tmp12 = triton_helpers.welford(tmp7, tmp8, tmp9, 1)
    tmp13 = tmp10[:, None]
    tmp14 = tmp11[:, None]
    tmp15 = tmp12[:, None]
    tl.store(out_ptr0 + (x0), tmp13, xmask)
    tl.store(out_ptr1 + (x0), tmp14, xmask)
''', device_str='cuda')


# kernel path: /workspace/caption-remover-main/src/stages/inpaint/../../../.torch_compile_cache/2s/c2sqt3j4s7tj5g74uezknhnwokv6nqxzlllaykixh6n7bjyhf3ur.py
# Source Nodes: [hidden_states_2], Original ATen: [aten.native_group_norm]
# hidden_states_2 => add_1, convert_element_type_1, mul_3
triton_poi_fused_native_group_norm_2 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.pointwise(
    size_hints=[33554432], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*fp16', 4: '*fp16', 5: '*fp16', 6: 'i32', 7: 'i32', 8: 'i32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_group_norm_2', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': 'a1fa794fabf29fbd973066d9ab65d9c16aea0ed2e098514ffda941ec2a0f87ad'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ks0, ks1, ks2, ks3, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % ks0
    x1 = (xindex // ks0) % ks1
    x2 = (xindex // ks2) % 22
    x3 = (xindex // ks3)
    x5 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (ks0*x1) + (ks0*ks1*x3) + (320*ks0*ks1*x2) + (320*ks0*ks1*((x0 + (ks0*x1)) // (ks0*ks1)))), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tl.load(in_ptr1 + ((x3 // 10)), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + ((x3 // 10)), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr3 + (x3), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp18 = tl.load(in_ptr4 + (x3), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 - tmp2
    tmp5 = 220*ks0*ks1
    tmp6 = tmp5.to(tl.float32)
    tmp7 = 0.0
    tmp8 = tmp6 - tmp7
    tmp9 = triton_helpers.maximum(tmp7, tmp8)
    tmp10 = tmp4 / tmp9
    tmp11 = 1e-06
    tmp12 = tmp10 + tmp11
    tmp13 = libdevice.rsqrt(tmp12)
    tmp14 = tmp3 * tmp13
    tmp16 = tmp15.to(tl.float32)
    tmp17 = tmp14 * tmp16
    tmp19 = tmp18.to(tl.float32)
    tmp20 = tmp17 + tmp19
    tmp21 = tmp20.to(tl.float32)
    tl.store(out_ptr0 + (x5), tmp21, xmask)
''', device_str='cuda')


# kernel path: /workspace/caption-remover-main/src/stages/inpaint/../../../.torch_compile_cache/wg/cwgeb7gwcdl7i5v4it2mmmull7j5ptfnurmg5h65n3zwtdvasoov.py
# Source Nodes: [encoder_hidden_states, hidden_states_4, norm_hidden_states], Original ATen: [aten.add, aten.native_layer_norm]
# encoder_hidden_states => add_5
# hidden_states_4 => add_2
# norm_hidden_states => add_3, add_4, convert_element_type_6, convert_element_type_7, mul_5, mul_6, rsqrt_1, sub_1, var_mean_1
triton_per_fused_add_native_layer_norm_3 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.persistent_reduction(
    size_hints=[65536, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: '*fp16', 5: '*fp16', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_3', 'mutated_arg_names': [], 'no_x_dim': True, 'backend_hash': 'a1fa794fabf29fbd973066d9ab65d9c16aea0ed2e098514ffda941ec2a0f87ad'}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel):
    XBLOCK: tl.constexpr = 1
    rnumel = 320
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    x2 = xindex % 22
    tmp0 = tl.load(in_ptr0 + (r1 + (320*x0)), rmask & xmask, other=0.0).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp27 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp30 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp34 = tl.load(in_ptr4 + (r1 + (320*x2)), rmask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp2.to(tl.float32)
    tmp4 = tl.broadcast_to(tmp3, [RBLOCK])
    tmp6 = tl.where(rmask & xmask, tmp4, 0)
    tmp7 = tl.broadcast_to(tmp4, [RBLOCK])
    tmp9 = tl.where(rmask & xmask, tmp7, 0)
    tmp10 = triton_helpers.promote_to_tensor(tl.sum(tmp9, 0))
    tmp11 = tl.full([1], 320, tl.int32)
    tmp12 = tmp11.to(tl.float32)
    tmp13 = tmp10 / tmp12
    tmp14 = tmp4 - tmp13
    tmp15 = tmp14 * tmp14
    tmp16 = tl.broadcast_to(tmp15, [RBLOCK])
    tmp18 = tl.where(rmask & xmask, tmp16, 0)
    tmp19 = triton_helpers.promote_to_tensor(tl.sum(tmp18, 0))
    tmp20 = tmp3 - tmp13
    tmp21 = 320.0
    tmp22 = tmp19 / tmp21
    tmp23 = 1e-05
    tmp24 = tmp22 + tmp23
    tmp25 = libdevice.rsqrt(tmp24)
    tmp26 = tmp20 * tmp25
    tmp28 = tmp27.to(tl.float32)
    tmp29 = tmp26 * tmp28
    tmp31 = tmp30.to(tl.float32)
    tmp32 = tmp29 + tmp31
    tmp33 = tmp32.to(tl.float32)
    tmp35 = tmp33 + tmp34
    tl.store(out_ptr2 + (r1 + (320*x0)), tmp35, rmask & xmask)
''', device_str='cuda')


# kernel path: /workspace/caption-remover-main/src/stages/inpaint/../../../.torch_compile_cache/xw/cxwyz7x5dxpylpbzzcbyaqoiampgqgxdidaju3hxkizpyaa275oc.py
# Source Nodes: [attn_output, encoder_hidden_states_1, hidden_states_11, hidden_states_4, norm_hidden_states_2], Original ATen: [aten.add, aten.div, aten.native_layer_norm]
# attn_output => div
# encoder_hidden_states_1 => add_9
# hidden_states_11 => add_6
# hidden_states_4 => add_2
# norm_hidden_states_2 => add_7, add_8, convert_element_type_17, convert_element_type_18, mul_11, mul_12, rsqrt_2, sub_2, var_mean_2
triton_per_fused_add_div_native_layer_norm_4 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.persistent_reduction(
    size_hints=[65536, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: '*fp16', 5: '*fp16', 6: '*fp16', 7: '*fp16', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_native_layer_norm_4', 'mutated_arg_names': [], 'no_x_dim': True, 'backend_hash': 'a1fa794fabf29fbd973066d9ab65d9c16aea0ed2e098514ffda941ec2a0f87ad'}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr2, xnumel, rnumel):
    XBLOCK: tl.constexpr = 1
    rnumel = 320
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    x2 = xindex % 22
    tmp0 = tl.load(in_ptr0 + (r1 + (320*x0)), rmask & xmask, other=0.0).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp5 = tl.load(in_ptr2 + (r1 + (320*x0)), rmask & xmask, other=0.0).to(tl.float32)
    tmp6 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp33 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp36 = tl.load(in_ptr5 + (r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp40 = tl.load(in_ptr6 + (r1 + (320*x2)), rmask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp3 = 1.0
    tmp4 = tmp2 / tmp3
    tmp7 = tmp5 + tmp6
    tmp8 = tmp4 + tmp7
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tl.broadcast_to(tmp9, [RBLOCK])
    tmp12 = tl.where(rmask & xmask, tmp10, 0)
    tmp13 = tl.broadcast_to(tmp10, [RBLOCK])
    tmp15 = tl.where(rmask & xmask, tmp13, 0)
    tmp16 = triton_helpers.promote_to_tensor(tl.sum(tmp15, 0))
    tmp17 = tl.full([1], 320, tl.int32)
    tmp18 = tmp17.to(tl.float32)
    tmp19 = tmp16 / tmp18
    tmp20 = tmp10 - tmp19
    tmp21 = tmp20 * tmp20
    tmp22 = tl.broadcast_to(tmp21, [RBLOCK])
    tmp24 = tl.where(rmask & xmask, tmp22, 0)
    tmp25 = triton_helpers.promote_to_tensor(tl.sum(tmp24, 0))
    tmp26 = tmp9 - tmp19
    tmp27 = 320.0
    tmp28 = tmp25 / tmp27
    tmp29 = 1e-05
    tmp30 = tmp28 + tmp29
    tmp31 = libdevice.rsqrt(tmp30)
    tmp32 = tmp26 * tmp31
    tmp34 = tmp33.to(tl.float32)
    tmp35 = tmp32 * tmp34
    tmp37 = tmp36.to(tl.float32)
    tmp38 = tmp35 + tmp37
    tmp39 = tmp38.to(tl.float32)
    tmp41 = tmp39 + tmp40
    tl.store(out_ptr2 + (r1 + (320*x0)), tmp41, rmask & xmask)
''', device_str='cuda')


# kernel path: /workspace/caption-remover-main/src/stages/inpaint/../../../.torch_compile_cache/ol/col6tyxxvg52gyimmarfkjgtwnxyxnekqv6qkd3jrcgjqdb4fbeq.py
# Source Nodes: [attn_output, attn_output_1, hidden_states_11, hidden_states_18, hidden_states_4, norm_hidden_states_4], Original ATen: [aten.add, aten.div, aten.native_layer_norm]
# attn_output => div
# attn_output_1 => div_1
# hidden_states_11 => add_6
# hidden_states_18 => add_10
# hidden_states_4 => add_2
# norm_hidden_states_4 => add_11, add_12, convert_element_type_28, convert_element_type_29, mul_17, mul_18, rsqrt_3, sub_3, var_mean_3
triton_per_fused_add_div_native_layer_norm_5 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.persistent_reduction(
    size_hints=[65536, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: '*fp16', 5: '*fp16', 6: '*fp16', 7: '*fp16', 8: '*fp16', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_native_layer_norm_5', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': True, 'backend_hash': 'a1fa794fabf29fbd973066d9ab65d9c16aea0ed2e098514ffda941ec2a0f87ad'}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr2, xnumel, rnumel):
    XBLOCK: tl.constexpr = 1
    rnumel = 320
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (r1 + (320*x0)), rmask & xmask, other=0.0).to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp5 = tl.load(in_ptr1 + (r1 + (320*x0)), rmask & xmask, other=0.0).to(tl.float32)
    tmp6 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp9 = tl.load(in_ptr3 + (r1 + (320*x0)), rmask & xmask, other=0.0).to(tl.float32)
    tmp10 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp38 = tl.load(in_ptr5 + (r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp41 = tl.load(in_ptr6 + (r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp3 = 1.0
    tmp4 = tmp2 / tmp3
    tmp7 = tmp5 + tmp6
    tmp8 = tmp7 / tmp3
    tmp11 = tmp9 + tmp10
    tmp12 = tmp8 + tmp11
    tmp13 = tmp4 + tmp12
    tmp14 = tmp13.to(tl.float32)
    tmp15 = tl.broadcast_to(tmp14, [RBLOCK])
    tmp17 = tl.where(rmask & xmask, tmp15, 0)
    tmp18 = tl.broadcast_to(tmp15, [RBLOCK])
    tmp20 = tl.where(rmask & xmask, tmp18, 0)
    tmp21 = triton_helpers.promote_to_tensor(tl.sum(tmp20, 0))
    tmp22 = tl.full([1], 320, tl.int32)
    tmp23 = tmp22.to(tl.float32)
    tmp24 = tmp21 / tmp23
    tmp25 = tmp15 - tmp24
    tmp26 = tmp25 * tmp25
    tmp27 = tl.broadcast_to(tmp26, [RBLOCK])
    tmp29 = tl.where(rmask & xmask, tmp27, 0)
    tmp30 = triton_helpers.promote_to_tensor(tl.sum(tmp29, 0))
    tmp31 = tmp14 - tmp24
    tmp32 = 320.0
    tmp33 = tmp30 / tmp32
    tmp34 = 1e-05
    tmp35 = tmp33 + tmp34
    tmp36 = libdevice.rsqrt(tmp35)
    tmp37 = tmp31 * tmp36
    tmp39 = tmp38.to(tl.float32)
    tmp40 = tmp37 * tmp39
    tmp42 = tmp41.to(tl.float32)
    tmp43 = tmp40 + tmp42
    tmp44 = tmp43.to(tl.float32)
    tl.store(in_out_ptr0 + (r1 + (320*x0)), tmp13, rmask & xmask)
    tl.store(out_ptr2 + (r1 + (320*x0)), tmp44, rmask & xmask)
''', device_str='cuda')


# kernel path: /workspace/caption-remover-main/src/stages/inpaint/../../../.torch_compile_cache/og/cogdaaz7e7sgq63pkwpudsqddkst2dxt5wj65jvk3wntccs6xsyf.py
# Source Nodes: [gelu, hidden_states_21], Original ATen: [aten.gelu, aten.mul]
# gelu => add_13, convert_element_type_33, convert_element_type_34, erf, mul_20, mul_21, mul_22
# hidden_states_21 => mul_23
triton_poi_fused_gelu_mul_6 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.pointwise(
    size_hints=[134217728], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_mul_6', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': 'a1fa794fabf29fbd973066d9ab65d9c16aea0ed2e098514ffda941ec2a0f87ad'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 1280
    x1 = (xindex // 1280)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (2560*x1)), xmask).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp3 = tl.load(in_ptr0 + (1280 + x0 + (2560*x1)), xmask).to(tl.float32)
    tmp4 = tl.load(in_ptr1 + (1280 + x0), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tmp5.to(tl.float32)
    tmp7 = 0.5
    tmp8 = tmp6 * tmp7
    tmp9 = 0.7071067811865476
    tmp10 = tmp6 * tmp9
    tmp11 = libdevice.erf(tmp10)
    tmp12 = 1.0
    tmp13 = tmp11 + tmp12
    tmp14 = tmp8 * tmp13
    tmp15 = tmp14.to(tl.float32)
    tmp16 = tmp2 * tmp15
    tl.store(out_ptr0 + (x2), tmp16, xmask)
''', device_str='cuda')


# kernel path: /workspace/caption-remover-main/src/stages/inpaint/../../../.torch_compile_cache/k7/ck7uaxjtklgc5js2zdc2hghngpkwrorv57vk3dh2h2ivaijb3ocj.py
# Source Nodes: [hidden_states_25], Original ATen: [aten.add]
# hidden_states_25 => add_14
triton_poi_fused_add_7 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.pointwise(
    size_hints=[33554432], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_7', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'backend_hash': 'a1fa794fabf29fbd973066d9ab65d9c16aea0ed2e098514ffda941ec2a0f87ad'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 320
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask).to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp3 = tl.load(in_ptr1 + (x2), xmask).to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tl.store(in_out_ptr0 + (x2), tmp4, xmask)
''', device_str='cuda')


# kernel path: /workspace/caption-remover-main/src/stages/inpaint/../../../.torch_compile_cache/vn/cvnkzziuaamxderstgk37kmqtpw6ts5oyjn2wokngygz3uqzhmby.py
# Source Nodes: [output], Original ATen: [aten.add]
# output => add_15
triton_poi_fused_add_8 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: 'i32', 5: 'i32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_8', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': 'a1fa794fabf29fbd973066d9ab65d9c16aea0ed2e098514ffda941ec2a0f87ad'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ks0, ks1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 7040
    yoffset = tl.program_id(1) * (tl.program_id(2) + 1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 320
    tmp0 = tl.load(in_ptr0 + (y3 + (7040*x2)), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last').to(tl.float32)
    tmp3 = tl.load(in_ptr2 + (x2 + (ks0*ks1*y3)), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tl.store(out_ptr0 + (x2 + (ks0*ks1*y3)), tmp4, xmask & ymask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1 = args
    args.clear()
    s1 = arg27_1
    s2 = arg28_1
    assert_size_stride(arg0_1, (320, ), (1, ))
    assert_size_stride(arg1_1, (320, ), (1, ))
    assert_size_stride(arg2_1, (320, 320), (320, 1))
    assert_size_stride(arg3_1, (320, ), (1, ))
    assert_size_stride(arg4_1, (320, ), (1, ))
    assert_size_stride(arg5_1, (320, ), (1, ))
    assert_size_stride(arg6_1, (320, 320), (320, 1))
    assert_size_stride(arg7_1, (320, 320), (320, 1))
    assert_size_stride(arg8_1, (320, 320), (320, 1))
    assert_size_stride(arg9_1, (320, 320), (320, 1))
    assert_size_stride(arg10_1, (320, ), (1, ))
    assert_size_stride(arg11_1, (320, ), (1, ))
    assert_size_stride(arg12_1, (320, ), (1, ))
    assert_size_stride(arg13_1, (320, 320), (320, 1))
    assert_size_stride(arg14_1, (320, 320), (320, 1))
    assert_size_stride(arg15_1, (320, 320), (320, 1))
    assert_size_stride(arg16_1, (320, 320), (320, 1))
    assert_size_stride(arg17_1, (320, ), (1, ))
    assert_size_stride(arg18_1, (320, ), (1, ))
    assert_size_stride(arg19_1, (320, ), (1, ))
    assert_size_stride(arg20_1, (2560, 320), (320, 1))
    assert_size_stride(arg21_1, (2560, ), (1, ))
    assert_size_stride(arg22_1, (320, 1280), (1280, 1))
    assert_size_stride(arg23_1, (320, ), (1, ))
    assert_size_stride(arg24_1, (320, 320), (320, 1))
    assert_size_stride(arg25_1, (320, ), (1, ))
    assert_size_stride(arg26_1, (1, 32, 320), (10240, 320, 1))
    assert_size_stride(arg29_1, (22, 320, s1, s2), (320*s1*s2, s1*s2, s2, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((1, 32, 1, 1, 18), (576, 1, 576, 576, 32), torch.float32)
        buf1 = empty_strided_cuda((1, 32, 1, 1, 18), (576, 1, 576, 576, 32), torch.float32)
        buf2 = empty_strided_cuda((1, 32, 1, 1, 18), (576, 1, 576, 576, 32), torch.float32)
        # Source Nodes: [hidden_states_2], Original ATen: [aten.native_group_norm]
        triton_red_fused_native_group_norm_0_rnumel = ((17 + (220*s1*s2)) // 18)
        stream0 = get_raw_stream(0)
        triton_red_fused_native_group_norm_0.run(arg29_1, buf0, buf1, buf2, s1, s2, 576, triton_red_fused_native_group_norm_0_rnumel, grid=grid(576), stream=stream0)
        buf3 = empty_strided_cuda((1, 32, 1, 1), (32, 1, 32, 32), torch.float32)
        buf4 = empty_strided_cuda((1, 32, 1, 1), (32, 1, 32, 32), torch.float32)
        # Source Nodes: [hidden_states_2], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_1.run(buf0, buf1, buf2, buf3, buf4, 32, 18, grid=grid(32), stream=stream0)
        del buf0
        del buf1
        del buf2
        ps0 = s1*s2
        ps1 = 22*s1*s2
        buf6 = empty_strided_cuda((1, 320, 22, s1, s2), (7040*s1*s2, 22*s1*s2, s1*s2, s2, 1), torch.float16)
        # Source Nodes: [hidden_states_2], Original ATen: [aten.native_group_norm]
        triton_poi_fused_native_group_norm_2_xnumel = 7040*s1*s2
        triton_poi_fused_native_group_norm_2.run(arg29_1, buf3, buf4, arg0_1, arg1_1, buf6, s2, s1, ps0, ps1, triton_poi_fused_native_group_norm_2_xnumel, grid=grid(triton_poi_fused_native_group_norm_2_xnumel), stream=stream0)
        del arg0_1
        del arg1_1
        del buf3
        del buf4
        buf7 = empty_strided_cuda((s1*s2, 22, 320), (7040, 320, 1), torch.float16)
        # Source Nodes: [hidden_states_4], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf6, (s1*s2, 22, 320), (1, s1*s2, 22*s1*s2), 0), reinterpret_tensor(arg2_1, (s1*s2, 320, 320), (0, 1, 320), 0), out=buf7)
        del arg2_1
        buf11 = reinterpret_tensor(buf6, (s1*s2, 22, 320), (7040, 320, 1), 0); del buf6  # reuse
        # Source Nodes: [encoder_hidden_states, hidden_states_4, norm_hidden_states], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_3_xnumel = 22*s1*s2
        triton_per_fused_add_native_layer_norm_3.run(buf7, arg3_1, arg4_1, arg5_1, arg26_1, buf11, triton_per_fused_add_native_layer_norm_3_xnumel, 320, grid=grid(triton_per_fused_add_native_layer_norm_3_xnumel), stream=stream0)
        del arg4_1
        del arg5_1
        buf12 = empty_strided_cuda((22*s1*s2, 320), (320, 1), torch.float16)
        # Source Nodes: [query], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf11, (22*s1*s2, 320), (320, 1), 0), reinterpret_tensor(arg6_1, (320, 320), (1, 320), 0), out=buf12)
        del arg6_1
        buf13 = empty_strided_cuda((22*s1*s2, 320), (320, 1), torch.float16)
        # Source Nodes: [key], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf11, (22*s1*s2, 320), (320, 1), 0), reinterpret_tensor(arg7_1, (320, 320), (1, 320), 0), out=buf13)
        del arg7_1
        buf14 = empty_strided_cuda((22*s1*s2, 320), (320, 1), torch.float16)
        # Source Nodes: [value], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf11, (22*s1*s2, 320), (320, 1), 0), reinterpret_tensor(arg8_1, (320, 320), (1, 320), 0), out=buf14)
        del arg8_1
        # Source Nodes: [hidden_states_5], Original ATen: [aten._scaled_dot_product_flash_attention]
        buf15 = aten._scaled_dot_product_flash_attention.default(reinterpret_tensor(buf12, (s1*s2, 8, 22, 40), (7040, 40, 320, 1), 0), reinterpret_tensor(buf13, (s1*s2, 8, 22, 40), (7040, 40, 320, 1), 0), reinterpret_tensor(buf14, (s1*s2, 8, 22, 40), (7040, 40, 320, 1), 0), scale=0.15811388300841897)
        buf16 = buf15[0]
        del buf15
        buf21 = buf14; del buf14  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf16, (22*s1*s2, 320), (320, 1), 0), reinterpret_tensor(arg9_1, (320, 320), (1, 320), 0), out=buf21)
        del arg9_1
        buf25 = reinterpret_tensor(buf16, (s1*s2, 22, 320), (7040, 320, 1), 0); del buf16  # reuse
        # Source Nodes: [attn_output, encoder_hidden_states_1, hidden_states_11, hidden_states_4, norm_hidden_states_2], Original ATen: [aten.add, aten.div, aten.native_layer_norm]
        triton_per_fused_add_div_native_layer_norm_4_xnumel = 22*s1*s2
        triton_per_fused_add_div_native_layer_norm_4.run(buf21, arg10_1, buf7, arg3_1, arg11_1, arg12_1, arg26_1, buf25, triton_per_fused_add_div_native_layer_norm_4_xnumel, 320, grid=grid(triton_per_fused_add_div_native_layer_norm_4_xnumel), stream=stream0)
        del arg11_1
        del arg12_1
        del arg26_1
        buf26 = buf13; del buf13  # reuse
        # Source Nodes: [query_2], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf25, (22*s1*s2, 320), (320, 1), 0), reinterpret_tensor(arg13_1, (320, 320), (1, 320), 0), out=buf26)
        del arg13_1
        buf27 = buf12; del buf12  # reuse
        # Source Nodes: [key_2], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf25, (22*s1*s2, 320), (320, 1), 0), reinterpret_tensor(arg14_1, (320, 320), (1, 320), 0), out=buf27)
        del arg14_1
        buf28 = reinterpret_tensor(buf11, (22*s1*s2, 320), (320, 1), 0); del buf11  # reuse
        # Source Nodes: [value_2], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf25, (22*s1*s2, 320), (320, 1), 0), reinterpret_tensor(arg15_1, (320, 320), (1, 320), 0), out=buf28)
        del arg15_1
        del buf25
        # Source Nodes: [hidden_states_12], Original ATen: [aten._scaled_dot_product_flash_attention]
        buf29 = aten._scaled_dot_product_flash_attention.default(reinterpret_tensor(buf26, (s1*s2, 8, 22, 40), (7040, 40, 320, 1), 0), reinterpret_tensor(buf27, (s1*s2, 8, 22, 40), (7040, 40, 320, 1), 0), reinterpret_tensor(buf28, (s1*s2, 8, 22, 40), (7040, 40, 320, 1), 0), scale=0.15811388300841897)
        del buf26
        del buf27
        buf30 = buf29[0]
        del buf29
        buf35 = buf28; del buf28  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf30, (22*s1*s2, 320), (320, 1), 0), reinterpret_tensor(arg16_1, (320, 320), (1, 320), 0), out=buf35)
        del arg16_1
        buf36 = reinterpret_tensor(buf35, (s1*s2, 22, 320), (7040, 320, 1), 0); del buf35  # reuse
        buf40 = reinterpret_tensor(buf30, (s1*s2, 22, 320), (7040, 320, 1), 0); del buf30  # reuse
        # Source Nodes: [attn_output, attn_output_1, hidden_states_11, hidden_states_18, hidden_states_4, norm_hidden_states_4], Original ATen: [aten.add, aten.div, aten.native_layer_norm]
        triton_per_fused_add_div_native_layer_norm_5_xnumel = 22*s1*s2
        triton_per_fused_add_div_native_layer_norm_5.run(buf36, arg17_1, buf21, arg10_1, buf7, arg3_1, arg18_1, arg19_1, buf40, triton_per_fused_add_div_native_layer_norm_5_xnumel, 320, grid=grid(triton_per_fused_add_div_native_layer_norm_5_xnumel), stream=stream0)
        del arg10_1
        del arg17_1
        del arg18_1
        del arg19_1
        del arg3_1
        del buf21
        del buf7
        buf41 = empty_strided_cuda((22*s1*s2, 2560), (2560, 1), torch.float16)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf40, (22*s1*s2, 320), (320, 1), 0), reinterpret_tensor(arg20_1, (320, 2560), (1, 320), 0), out=buf41)
        del arg20_1
        buf42 = empty_strided_cuda((s1*s2, 22, 1280), (28160, 1280, 1), torch.float16)
        # Source Nodes: [gelu, hidden_states_21], Original ATen: [aten.gelu, aten.mul]
        triton_poi_fused_gelu_mul_6_xnumel = 28160*s1*s2
        triton_poi_fused_gelu_mul_6.run(buf41, arg21_1, buf42, triton_poi_fused_gelu_mul_6_xnumel, grid=grid(triton_poi_fused_gelu_mul_6_xnumel), stream=stream0)
        del arg21_1
        del buf41
        buf43 = reinterpret_tensor(buf40, (22*s1*s2, 320), (320, 1), 0); del buf40  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf42, (22*s1*s2, 1280), (1280, 1), 0), reinterpret_tensor(arg22_1, (1280, 320), (1, 1280), 0), out=buf43)
        del arg22_1
        del buf42
        buf44 = reinterpret_tensor(buf43, (s1*s2, 22, 320), (7040, 320, 1), 0); del buf43  # reuse
        # Source Nodes: [hidden_states_25], Original ATen: [aten.add]
        triton_poi_fused_add_7_xnumel = 7040*s1*s2
        triton_poi_fused_add_7.run(buf44, arg23_1, buf36, triton_poi_fused_add_7_xnumel, grid=grid(triton_poi_fused_add_7_xnumel), stream=stream0)
        del arg23_1
        buf45 = reinterpret_tensor(buf36, (22*s1*s2, 320), (320, 1), 0); del buf36  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf44, (22*s1*s2, 320), (320, 1), 0), reinterpret_tensor(arg24_1, (320, 320), (1, 320), 0), out=buf45)
        del arg24_1
        buf46 = reinterpret_tensor(buf44, (22, 320, s1, s2), (320*s1*s2, s1*s2, s2, 1), 0); del buf44  # reuse
        # Source Nodes: [output], Original ATen: [aten.add]
        triton_poi_fused_add_8_xnumel = s1*s2
        triton_poi_fused_add_8.run(buf45, arg25_1, arg29_1, buf46, s1, s2, 7040, triton_poi_fused_add_8_xnumel, grid=grid(7040, triton_poi_fused_add_8_xnumel), stream=stream0)
        del arg25_1
        del arg29_1
        del buf45
    return (buf46, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg1_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg2_1 = rand_strided((320, 320), (320, 1), device='cuda:0', dtype=torch.float16)
    arg3_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg4_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg5_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg6_1 = rand_strided((320, 320), (320, 1), device='cuda:0', dtype=torch.float16)
    arg7_1 = rand_strided((320, 320), (320, 1), device='cuda:0', dtype=torch.float16)
    arg8_1 = rand_strided((320, 320), (320, 1), device='cuda:0', dtype=torch.float16)
    arg9_1 = rand_strided((320, 320), (320, 1), device='cuda:0', dtype=torch.float16)
    arg10_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg11_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg12_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg13_1 = rand_strided((320, 320), (320, 1), device='cuda:0', dtype=torch.float16)
    arg14_1 = rand_strided((320, 320), (320, 1), device='cuda:0', dtype=torch.float16)
    arg15_1 = rand_strided((320, 320), (320, 1), device='cuda:0', dtype=torch.float16)
    arg16_1 = rand_strided((320, 320), (320, 1), device='cuda:0', dtype=torch.float16)
    arg17_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg18_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg19_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg20_1 = rand_strided((2560, 320), (320, 1), device='cuda:0', dtype=torch.float16)
    arg21_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg22_1 = rand_strided((320, 1280), (1280, 1), device='cuda:0', dtype=torch.float16)
    arg23_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg24_1 = rand_strided((320, 320), (320, 1), device='cuda:0', dtype=torch.float16)
    arg25_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg26_1 = rand_strided((1, 32, 320), (10240, 320, 1), device='cuda:0', dtype=torch.float16)
    arg27_1 = 30
    arg28_1 = 90
    arg29_1 = rand_strided((22, 320, 30, 90), (864000, 2700, 90, 1), device='cuda:0', dtype=torch.float16)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
