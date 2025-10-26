
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


# kernel path: /workspace/caption-remover-main/src/stages/inpaint/../../../.torch_compile_cache/mu/cmuyetyonfmigo6jswrppxfsypbwipnhxy7jqpabxqhw2qnvzqlz.py
# Source Nodes: [hidden_states], Original ATen: [aten.native_group_norm]
# hidden_states => convert_element_type, var_mean
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
    triton_meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
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
        tmp0 = tl.load(in_ptr0 + (r1 + (20*ks0*ks1*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
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


# kernel path: /workspace/caption-remover-main/src/stages/inpaint/../../../.torch_compile_cache/mf/cmfr5zwt6o2amkywwfuwo2cfzswkekjpbwpsgca537hs6cw27qsv.py
# Source Nodes: [hidden_states], Original ATen: [aten.native_group_norm]
# hidden_states => add_1, convert_element_type_1, mul_2
triton_poi_fused_native_group_norm_1 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_group_norm_1', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': 'a1fa794fabf29fbd973066d9ab65d9c16aea0ed2e098514ffda941ec2a0f87ad'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ks0, ks1, ks2, ks3, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // ks0) % 640
    x2 = (xindex // ks1)
    tmp0 = tl.load(in_ptr0 + (x3), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tl.load(in_ptr1 + ((32*x2) + (x1 // 20)), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + ((32*x2) + (x1 // 20)), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp18 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 - tmp2
    tmp5 = 20*ks2*ks3
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
    tl.store(out_ptr0 + (x3), tmp21, xmask)
''', device_str='cuda')


# kernel path: /workspace/caption-remover-main/src/stages/inpaint/../../../.torch_compile_cache/ce/ccehsxhd25ziwkgruplxbpazyec5xvwv4igp7me7k2nuy4twka22.py
# Source Nodes: [encoder_hidden_states], Original ATen: [aten.native_layer_norm]
# encoder_hidden_states => clone, convert_element_type_4, var_mean_1
triton_red_fused_native_layer_norm_2 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.reduction(
    size_hints=[16384, 1024],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32', 7: 'i32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_2', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': 'a1fa794fabf29fbd973066d9ab65d9c16aea0ed2e098514ffda941ec2a0f87ad'}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, ks0, ks1, ks2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    rnumel = 640
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % ks0
    x1 = (xindex // ks0)
    tmp9_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp9_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp9_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (ks1*ks2*r2) + (640*ks1*ks2*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (r2), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp3 = tl.load(in_ptr2 + (x0 + (ks1*ks2*r2) + (640*ks1*ks2*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp2 = tmp0 + tmp1
        tmp4 = 0.125
        tmp5 = tmp3 * tmp4
        tmp6 = tmp2 + tmp5
        tmp7 = tmp6.to(tl.float32)
        tmp8 = tl.broadcast_to(tmp7, [XBLOCK, RBLOCK])
        tmp9_mean_next, tmp9_m2_next, tmp9_weight_next = triton_helpers.welford_reduce(
            tmp8, tmp9_mean, tmp9_m2, tmp9_weight, roffset == 0
        )
        tmp9_mean = tl.where(rmask & xmask, tmp9_mean_next, tmp9_mean)
        tmp9_m2 = tl.where(rmask & xmask, tmp9_m2_next, tmp9_m2)
        tmp9_weight = tl.where(rmask & xmask, tmp9_weight_next, tmp9_weight)
    tmp9_tmp, tmp10_tmp, tmp11_tmp = triton_helpers.welford(
        tmp9_mean, tmp9_m2, tmp9_weight, 1
    )
    tmp9 = tmp9_tmp[:, None]
    tmp10 = tmp10_tmp[:, None]
    tmp11 = tmp11_tmp[:, None]
    tl.store(out_ptr0 + (x3), tmp9, xmask)
    tl.store(out_ptr1 + (x3), tmp10, xmask)
''', device_str='cuda')


# kernel path: /workspace/caption-remover-main/src/stages/inpaint/../../../.torch_compile_cache/h7/ch7cb3qvfeeoxpgaq4smdlfjwsvn4n7fj3ynl3jonwdzq6atj4ay.py
# Source Nodes: [encoder_hidden_states], Original ATen: [aten.native_layer_norm]
# encoder_hidden_states => add_3, add_4, clone, convert_element_type_4, convert_element_type_5, mul_5, mul_6, rsqrt_1, sub_1, var_mean_1
triton_poi_fused_native_layer_norm_3 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.pointwise(
    size_hints=[16384, 1024], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp32', 4: '*fp32', 5: '*fp16', 6: '*fp16', 7: '*fp16', 8: 'i32', 9: 'i32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_layer_norm_3', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': 'a1fa794fabf29fbd973066d9ab65d9c16aea0ed2e098514ffda941ec2a0f87ad'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, ks0, ks1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 14080
    yoffset = tl.program_id(1) * (tl.program_id(2) + 1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 640
    y1 = (yindex // 640)
    tmp0 = tl.load(in_ptr0 + (x2 + (ks0*ks1*y3)), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last').to(tl.float32)
    tmp3 = tl.load(in_ptr2 + (x2 + (ks0*ks1*y3)), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp8 = tl.load(in_ptr3 + (x2 + (ks0*ks1*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr4 + (x2 + (ks0*ks1*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr5 + (y0), ymask, eviction_policy='evict_last').to(tl.float32)
    tmp20 = tl.load(in_ptr6 + (y0), ymask, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp4 = 0.125
    tmp5 = tmp3 * tmp4
    tmp6 = tmp2 + tmp5
    tmp7 = tmp6.to(tl.float32)
    tmp9 = tmp7 - tmp8
    tmp11 = 640.0
    tmp12 = tmp10 / tmp11
    tmp13 = 1e-05
    tmp14 = tmp12 + tmp13
    tmp15 = libdevice.rsqrt(tmp14)
    tmp16 = tmp9 * tmp15
    tmp18 = tmp17.to(tl.float32)
    tmp19 = tmp16 * tmp18
    tmp21 = tmp20.to(tl.float32)
    tmp22 = tmp19 + tmp21
    tmp23 = tmp22.to(tl.float32)
    tl.store(out_ptr0 + (y0 + (640*x2) + (640*ks0*ks1*y1)), tmp23, xmask & ymask)
''', device_str='cuda')


# kernel path: /workspace/caption-remover-main/src/stages/inpaint/../../../.torch_compile_cache/iw/ciwr4czt7ds4otadumissrxlksapmk3ca3got5coyiiidsvkibat.py
# Source Nodes: [hidden_states_4], Original ATen: [aten._scaled_dot_product_flash_attention]
# hidden_states_4 => _scaled_dot_product_flash_attention
triton_poi_fused__scaled_dot_product_flash_attention_4 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp16', 1: '*fp16', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__scaled_dot_product_flash_attention_4', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'backend_hash': 'a1fa794fabf29fbd973066d9ab65d9c16aea0ed2e098514ffda941ec2a0f87ad'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), xmask).to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (x0), xmask).to(tl.float32)
    tmp2 = 0.125
    tmp3 = tmp1 * tmp2
    tmp4 = tmp0 + tmp3
    tl.store(in_out_ptr0 + (x0), tmp4, xmask)
''', device_str='cuda')


# kernel path: /workspace/caption-remover-main/src/stages/inpaint/../../../.torch_compile_cache/gx/cgx5giqusgzdqafltgrtsvjbm2mhwrmn27yhzqrtosclobejt64n.py
# Source Nodes: [attn_output, hidden_states_10, mul_5, residual_2, result_13], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm]
# attn_output => div
# hidden_states_10 => add_9
# mul_5 => mul_22
# residual_2 => add_10, add_11, convert_element_type_31, convert_element_type_32, mul_23, mul_24, rsqrt_2, sub_2, var_mean_2
# result_13 => add_8
triton_per_fused_add_div_mul_native_layer_norm_5 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.persistent_reduction(
    size_hints=[16384, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: '*fp16', 5: '*fp16', 6: '*fp16', 7: '*fp16', 8: '*fp16', 9: 'i32', 10: 'i32', 11: 'i32', 12: 'i32', 13: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 13), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(13,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_mul_native_layer_norm_5', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': True, 'backend_hash': 'a1fa794fabf29fbd973066d9ab65d9c16aea0ed2e098514ffda941ec2a0f87ad'}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr2, ks0, ks1, ks2, xnumel, rnumel):
    XBLOCK: tl.constexpr = 1
    rnumel = 640
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x0 = xindex % ks0
    x1 = (xindex // ks0)
    tmp0 = tl.load(in_out_ptr0 + (r2 + (640*x3)), rmask & xmask, other=0.0).to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (r2), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp3 = tl.load(in_ptr1 + (r2 + (640*x3)), rmask & xmask, other=0.0).to(tl.float32)
    tmp9 = tl.load(in_ptr2 + (x0 + (ks1*ks2*r2) + (640*ks1*ks2*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp10 = tl.load(in_ptr3 + (r2), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp12 = tl.load(in_ptr4 + (x0 + (ks1*ks2*r2) + (640*ks1*ks2*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp40 = tl.load(in_ptr5 + (r2), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp43 = tl.load(in_ptr6 + (r2), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp4 = 0.125
    tmp5 = tmp3 * tmp4
    tmp6 = tmp2 + tmp5
    tmp7 = 1.0
    tmp8 = tmp6 / tmp7
    tmp11 = tmp9 + tmp10
    tmp13 = tmp12 * tmp4
    tmp14 = tmp11 + tmp13
    tmp15 = tmp8 + tmp14
    tmp16 = tmp15.to(tl.float32)
    tmp17 = tl.broadcast_to(tmp16, [RBLOCK])
    tmp19 = tl.where(rmask & xmask, tmp17, 0)
    tmp20 = tl.broadcast_to(tmp17, [RBLOCK])
    tmp22 = tl.where(rmask & xmask, tmp20, 0)
    tmp23 = triton_helpers.promote_to_tensor(tl.sum(tmp22, 0))
    tmp24 = tl.full([1], 640, tl.int32)
    tmp25 = tmp24.to(tl.float32)
    tmp26 = tmp23 / tmp25
    tmp27 = tmp17 - tmp26
    tmp28 = tmp27 * tmp27
    tmp29 = tl.broadcast_to(tmp28, [RBLOCK])
    tmp31 = tl.where(rmask & xmask, tmp29, 0)
    tmp32 = triton_helpers.promote_to_tensor(tl.sum(tmp31, 0))
    tmp33 = tmp16 - tmp26
    tmp34 = 640.0
    tmp35 = tmp32 / tmp34
    tmp36 = 1e-05
    tmp37 = tmp35 + tmp36
    tmp38 = libdevice.rsqrt(tmp37)
    tmp39 = tmp33 * tmp38
    tmp41 = tmp40.to(tl.float32)
    tmp42 = tmp39 * tmp41
    tmp44 = tmp43.to(tl.float32)
    tmp45 = tmp42 + tmp44
    tmp46 = tmp45.to(tl.float32)
    tl.store(in_out_ptr0 + (r2 + (640*x3)), tmp15, rmask & xmask)
    tl.store(out_ptr2 + (r2 + (640*x3)), tmp46, rmask & xmask)
''', device_str='cuda')


# kernel path: /workspace/caption-remover-main/src/stages/inpaint/../../../.torch_compile_cache/5i/c5iinhbmxaxlvb6umpn7tioipzp5236dkq3etussiyudl5beaaxi.py
# Source Nodes: [hidden_states_11], Original ATen: [aten._scaled_dot_product_flash_attention]
# hidden_states_11 => _scaled_dot_product_flash_attention_1
triton_poi_fused__scaled_dot_product_flash_attention_6 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.pointwise(
    size_hints=[2097152], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp16', 1: '*fp16', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__scaled_dot_product_flash_attention_6', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'backend_hash': 'a1fa794fabf29fbd973066d9ab65d9c16aea0ed2e098514ffda941ec2a0f87ad'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1084160
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), xmask).to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (x0), xmask).to(tl.float32)
    tmp2 = 0.125
    tmp3 = tmp1 * tmp2
    tmp4 = tmp0 + tmp3
    tl.store(in_out_ptr0 + (x0), tmp4, xmask)
''', device_str='cuda')


# kernel path: /workspace/caption-remover-main/src/stages/inpaint/../../../.torch_compile_cache/hs/chspzsjbqborhbdmnh24pxm36xh5y3qs4qrnihpbwjkf6vbgcxlm.py
# Source Nodes: [attn_output_1, hidden_states_17, mul_9, norm_hidden_states_2, result_25], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm]
# attn_output_1 => div_1
# hidden_states_17 => add_16
# mul_9 => mul_34
# norm_hidden_states_2 => add_17, add_18, convert_element_type_58, convert_element_type_59, mul_35, mul_36, rsqrt_3, sub_3, var_mean_3
# result_25 => add_15
triton_per_fused_add_div_mul_native_layer_norm_7 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.persistent_reduction(
    size_hints=[16384, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: '*fp16', 5: '*fp16', 6: '*fp16', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_mul_native_layer_norm_7', 'mutated_arg_names': [], 'no_x_dim': True, 'backend_hash': 'a1fa794fabf29fbd973066d9ab65d9c16aea0ed2e098514ffda941ec2a0f87ad'}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, xnumel, rnumel):
    XBLOCK: tl.constexpr = 1
    rnumel = 640
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (640*x0)), rmask & xmask, other=0.0).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp3 = tl.load(in_ptr2 + (r1 + (640*x0)), rmask & xmask, other=0.0).to(tl.float32)
    tmp9 = tl.load(in_ptr3 + (r1 + (640*x0)), rmask & xmask, other=0.0).to(tl.float32)
    tmp35 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp38 = tl.load(in_ptr5 + (r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp4 = 0.125
    tmp5 = tmp3 * tmp4
    tmp6 = tmp2 + tmp5
    tmp7 = 1.0
    tmp8 = tmp6 / tmp7
    tmp10 = tmp8 + tmp9
    tmp11 = tmp10.to(tl.float32)
    tmp12 = tl.broadcast_to(tmp11, [RBLOCK])
    tmp14 = tl.where(rmask & xmask, tmp12, 0)
    tmp15 = tl.broadcast_to(tmp12, [RBLOCK])
    tmp17 = tl.where(rmask & xmask, tmp15, 0)
    tmp18 = triton_helpers.promote_to_tensor(tl.sum(tmp17, 0))
    tmp19 = tl.full([1], 640, tl.int32)
    tmp20 = tmp19.to(tl.float32)
    tmp21 = tmp18 / tmp20
    tmp22 = tmp12 - tmp21
    tmp23 = tmp22 * tmp22
    tmp24 = tl.broadcast_to(tmp23, [RBLOCK])
    tmp26 = tl.where(rmask & xmask, tmp24, 0)
    tmp27 = triton_helpers.promote_to_tensor(tl.sum(tmp26, 0))
    tmp28 = tmp11 - tmp21
    tmp29 = 640.0
    tmp30 = tmp27 / tmp29
    tmp31 = 1e-05
    tmp32 = tmp30 + tmp31
    tmp33 = libdevice.rsqrt(tmp32)
    tmp34 = tmp28 * tmp33
    tmp36 = tmp35.to(tl.float32)
    tmp37 = tmp34 * tmp36
    tmp39 = tmp38.to(tl.float32)
    tmp40 = tmp37 + tmp39
    tmp41 = tmp40.to(tl.float32)
    tl.store(out_ptr2 + (r1 + (640*x0)), tmp41, rmask & xmask)
''', device_str='cuda')


# kernel path: /workspace/caption-remover-main/src/stages/inpaint/../../../.torch_compile_cache/jm/cjmz3kc3yphetz7o2g5rlnixvcqiknnru7x7xbpdqnhxwgruxy2n.py
# Source Nodes: [gelu, hidden_states_20], Original ATen: [aten.gelu, aten.mul]
# gelu => add_20, convert_element_type_67, convert_element_type_68, erf, mul_41, mul_42, mul_43
# hidden_states_20 => mul_44
triton_poi_fused_gelu_mul_8 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_mul_8', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': 'a1fa794fabf29fbd973066d9ab65d9c16aea0ed2e098514ffda941ec2a0f87ad'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 2560
    x1 = (xindex // 2560)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (5120*x1)), xmask).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp3 = tl.load(in_ptr2 + (x0 + (5120*x1)), xmask).to(tl.float32)
    tmp7 = tl.load(in_ptr0 + (2560 + x0 + (5120*x1)), xmask).to(tl.float32)
    tmp8 = tl.load(in_ptr1 + (2560 + x0), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp10 = tl.load(in_ptr2 + (2560 + x0 + (5120*x1)), xmask).to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp4 = 0.125
    tmp5 = tmp3 * tmp4
    tmp6 = tmp2 + tmp5
    tmp9 = tmp7 + tmp8
    tmp11 = tmp10 * tmp4
    tmp12 = tmp9 + tmp11
    tmp13 = tmp12.to(tl.float32)
    tmp14 = 0.5
    tmp15 = tmp13 * tmp14
    tmp16 = 0.7071067811865476
    tmp17 = tmp13 * tmp16
    tmp18 = libdevice.erf(tmp17)
    tmp19 = 1.0
    tmp20 = tmp18 + tmp19
    tmp21 = tmp15 * tmp20
    tmp22 = tmp21.to(tl.float32)
    tmp23 = tmp6 * tmp22
    tl.store(out_ptr0 + (x2), tmp23, xmask)
''', device_str='cuda')


# kernel path: /workspace/caption-remover-main/src/stages/inpaint/../../../.torch_compile_cache/ii/ciiimgrpse6goyrxpvjgbxdwjv37xysdkiyckcv2mfzxeaxfvokc.py
# Source Nodes: [hidden_states_25], Original ATen: [aten.clone]
# hidden_states_25 => clone_4
triton_poi_fused_clone_9 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.pointwise(
    size_hints=[16384, 1024], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: '*fp16', 5: '*fp16', 6: '*fp16', 7: '*fp16', 8: 'i32', 9: 'i32', 10: 'i32', 11: 'i32', 12: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 12), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(12,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_9', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': 'a1fa794fabf29fbd973066d9ab65d9c16aea0ed2e098514ffda941ec2a0f87ad'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, ks0, ks1, ks2, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    xnumel = 640
    yoffset = tl.program_id(1) * (tl.program_id(2) + 1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % ks0
    y1 = (yindex // ks0)
    tmp0 = tl.load(in_ptr0 + (x2 + (640*y3)), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x2), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp3 = tl.load(in_ptr2 + (x2 + (640*y3)), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp7 = tl.load(in_ptr3 + (x2 + (640*y3)), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp8 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp10 = tl.load(in_ptr5 + (x2 + (640*y3)), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp15 = tl.load(in_ptr6 + (x2 + (640*y3)), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp4 = 0.125
    tmp5 = tmp3 * tmp4
    tmp6 = tmp2 + tmp5
    tmp9 = tmp7 + tmp8
    tmp11 = tmp10 * tmp4
    tmp12 = tmp9 + tmp11
    tmp13 = 1.0
    tmp14 = tmp12 / tmp13
    tmp16 = tmp14 + tmp15
    tmp17 = tmp6 + tmp16
    tl.store(out_ptr0 + (y0 + (ks1*ks2*x2) + (640*ks1*ks2*y1)), tmp17, xmask & ymask)
''', device_str='cuda')


# kernel path: /workspace/caption-remover-main/src/stages/inpaint/../../../.torch_compile_cache/4v/c4vvhafanm4hu3m6lpicx2rt7ub5fenndcyhcxiamhbvse2lk6ry.py
# Source Nodes: [mul_13, output, result_33, result_34], Original ATen: [aten.add, aten.convolution, aten.mul]
# mul_13 => mul_49
# output => add_24
# result_33 => convolution_3
# result_34 => add_23
triton_poi_fused_add_convolution_mul_10 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_mul_10', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'backend_hash': 'a1fa794fabf29fbd973066d9ab65d9c16aea0ed2e098514ffda941ec2a0f87ad'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, ks0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // ks0) % 640
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp3 = tl.load(in_ptr1 + (x3), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp7 = tl.load(in_ptr2 + (x3), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp4 = 0.125
    tmp5 = tmp3 * tmp4
    tmp6 = tmp2 + tmp5
    tmp8 = tmp6 + tmp7
    tl.store(in_out_ptr0 + (x3), tmp8, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1 = args
    args.clear()
    s1 = arg50_1
    s2 = arg51_1
    assert_size_stride(arg0_1, (640, ), (1, ))
    assert_size_stride(arg1_1, (640, ), (1, ))
    assert_size_stride(arg2_1, (640, 640, 1, 1), (640, 1, 1, 1))
    assert_size_stride(arg3_1, (640, ), (1, ))
    assert_size_stride(arg4_1, (64, 640, 1, 1), (640, 1, 1, 1))
    assert_size_stride(arg5_1, (640, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(arg6_1, (640, ), (1, ))
    assert_size_stride(arg7_1, (640, ), (1, ))
    assert_size_stride(arg8_1, (640, 640), (640, 1))
    assert_size_stride(arg9_1, (64, 640), (640, 1))
    assert_size_stride(arg10_1, (640, 64), (64, 1))
    assert_size_stride(arg11_1, (640, 640), (640, 1))
    assert_size_stride(arg12_1, (64, 640), (640, 1))
    assert_size_stride(arg13_1, (640, 64), (64, 1))
    assert_size_stride(arg14_1, (640, 640), (640, 1))
    assert_size_stride(arg15_1, (64, 640), (640, 1))
    assert_size_stride(arg16_1, (640, 64), (64, 1))
    assert_size_stride(arg17_1, (640, 640), (640, 1))
    assert_size_stride(arg18_1, (640, ), (1, ))
    assert_size_stride(arg19_1, (64, 640), (640, 1))
    assert_size_stride(arg20_1, (640, 64), (64, 1))
    assert_size_stride(arg21_1, (640, ), (1, ))
    assert_size_stride(arg22_1, (640, ), (1, ))
    assert_size_stride(arg23_1, (640, 640), (640, 1))
    assert_size_stride(arg24_1, (64, 640), (640, 1))
    assert_size_stride(arg25_1, (640, 64), (64, 1))
    assert_size_stride(arg26_1, (640, 768), (768, 1))
    assert_size_stride(arg27_1, (64, 768), (768, 1))
    assert_size_stride(arg28_1, (640, 64), (64, 1))
    assert_size_stride(arg29_1, (640, 768), (768, 1))
    assert_size_stride(arg30_1, (64, 768), (768, 1))
    assert_size_stride(arg31_1, (640, 64), (64, 1))
    assert_size_stride(arg32_1, (640, 640), (640, 1))
    assert_size_stride(arg33_1, (640, ), (1, ))
    assert_size_stride(arg34_1, (64, 640), (640, 1))
    assert_size_stride(arg35_1, (640, 64), (64, 1))
    assert_size_stride(arg36_1, (640, ), (1, ))
    assert_size_stride(arg37_1, (640, ), (1, ))
    assert_size_stride(arg38_1, (5120, 640), (640, 1))
    assert_size_stride(arg39_1, (5120, ), (1, ))
    assert_size_stride(arg40_1, (64, 640), (640, 1))
    assert_size_stride(arg41_1, (5120, 64), (64, 1))
    assert_size_stride(arg42_1, (640, 2560), (2560, 1))
    assert_size_stride(arg43_1, (640, ), (1, ))
    assert_size_stride(arg44_1, (64, 2560), (2560, 1))
    assert_size_stride(arg45_1, (640, 64), (64, 1))
    assert_size_stride(arg46_1, (640, 640, 1, 1), (640, 1, 1, 1))
    assert_size_stride(arg47_1, (640, ), (1, ))
    assert_size_stride(arg48_1, (64, 640, 1, 1), (640, 1, 1, 1))
    assert_size_stride(arg49_1, (640, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(arg52_1, (22, 640, s1, s2), (640*s1*s2, s1*s2, s2, 1))
    assert_size_stride(arg53_1, (22, 77, 768), (59136, 768, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((22, 32, 1, 1), (32, 1, 704, 704), torch.float32)
        buf1 = empty_strided_cuda((22, 32, 1, 1), (32, 1, 704, 704), torch.float32)
        # Source Nodes: [hidden_states], Original ATen: [aten.native_group_norm]
        triton_red_fused_native_group_norm_0_rnumel = 20*s1*s2
        stream0 = get_raw_stream(0)
        triton_red_fused_native_group_norm_0.run(arg52_1, buf0, buf1, s1, s2, 704, triton_red_fused_native_group_norm_0_rnumel, grid=grid(704), stream=stream0)
        ps0 = s1*s2
        ps1 = 640*s1*s2
        buf3 = empty_strided_cuda((22, 640, s1, s2), (640*s1*s2, s1*s2, s2, 1), torch.float16)
        # Source Nodes: [hidden_states], Original ATen: [aten.native_group_norm]
        triton_poi_fused_native_group_norm_1_xnumel = 14080*s1*s2
        triton_poi_fused_native_group_norm_1.run(arg52_1, buf0, buf1, arg0_1, arg1_1, buf3, ps0, ps1, s1, s2, triton_poi_fused_native_group_norm_1_xnumel, grid=grid(triton_poi_fused_native_group_norm_1_xnumel), stream=stream0)
        del arg0_1
        del arg1_1
        del buf0
        del buf1
        # Source Nodes: [result], Original ATen: [aten.convolution]
        buf4 = extern_kernels.convolution(buf3, arg2_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf4, (22, 640, s1, s2), (640*s1*s2, s1*s2, s2, 1))
        del arg2_1
        # Source Nodes: [l__self___proj_in_lora_a_default_0], Original ATen: [aten.convolution]
        buf5 = extern_kernels.convolution(buf3, arg4_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf5, (22, 64, s1, s2), (64*s1*s2, s1*s2, s2, 1))
        del arg4_1
        # Source Nodes: [l__self___proj_in_lora_b_default_0], Original ATen: [aten.convolution]
        buf6 = extern_kernels.convolution(buf5, arg5_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf6, (22, 640, s1, s2), (640*s1*s2, s1*s2, s2, 1))
        del arg5_1
        buf7 = empty_strided_cuda((22, s1*s2, 1), (s1*s2, 1, 22*s1*s2), torch.float32)
        buf8 = empty_strided_cuda((22, s1*s2, 1), (s1*s2, 1, 22*s1*s2), torch.float32)
        # Source Nodes: [encoder_hidden_states], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_2_xnumel = 22*s1*s2
        triton_red_fused_native_layer_norm_2.run(buf4, arg3_1, buf6, buf7, buf8, ps0, s1, s2, triton_red_fused_native_layer_norm_2_xnumel, 640, grid=grid(triton_red_fused_native_layer_norm_2_xnumel), stream=stream0)
        buf10 = reinterpret_tensor(buf3, (22, s1*s2, 640), (640*s1*s2, 640, 1), 0); del buf3  # reuse
        # Source Nodes: [encoder_hidden_states], Original ATen: [aten.native_layer_norm]
        triton_poi_fused_native_layer_norm_3_xnumel = s1*s2
        triton_poi_fused_native_layer_norm_3.run(buf4, arg3_1, buf6, buf7, buf8, arg6_1, arg7_1, buf10, s1, s2, 14080, triton_poi_fused_native_layer_norm_3_xnumel, grid=grid(14080, triton_poi_fused_native_layer_norm_3_xnumel), stream=stream0)
        del arg6_1
        del arg7_1
        del buf7
        del buf8
        buf11 = empty_strided_cuda((22*s1*s2, 640), (640, 1), torch.float16)
        # Source Nodes: [result_3], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf10, (22*s1*s2, 640), (640, 1), 0), reinterpret_tensor(arg8_1, (640, 640), (1, 640), 0), out=buf11)
        del arg8_1
        buf12 = reinterpret_tensor(buf5, (22*s1*s2, 64), (64, 1), 0); del buf5  # reuse
        # Source Nodes: [l__self___transformer_blocks_0_attn1_to_q_lora_a_default_0], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf10, (22*s1*s2, 640), (640, 1), 0), reinterpret_tensor(arg9_1, (640, 64), (1, 640), 0), out=buf12)
        del arg9_1
        buf13 = empty_strided_cuda((22*s1*s2, 640), (640, 1), torch.float16)
        # Source Nodes: [l__self___transformer_blocks_0_attn1_to_q_lora_b_default_0], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf12, (22*s1*s2, 64), (64, 1), 0), reinterpret_tensor(arg10_1, (64, 640), (1, 64), 0), out=buf13)
        del arg10_1
        buf14 = empty_strided_cuda((22*s1*s2, 640), (640, 1), torch.float16)
        # Source Nodes: [result_6], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf10, (22*s1*s2, 640), (640, 1), 0), reinterpret_tensor(arg11_1, (640, 640), (1, 640), 0), out=buf14)
        del arg11_1
        buf15 = buf12; del buf12  # reuse
        # Source Nodes: [l__self___transformer_blocks_0_attn1_to_k_lora_a_default_0], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf10, (22*s1*s2, 640), (640, 1), 0), reinterpret_tensor(arg12_1, (640, 64), (1, 640), 0), out=buf15)
        del arg12_1
        buf16 = empty_strided_cuda((22*s1*s2, 640), (640, 1), torch.float16)
        # Source Nodes: [l__self___transformer_blocks_0_attn1_to_k_lora_b_default_0], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf15, (22*s1*s2, 64), (64, 1), 0), reinterpret_tensor(arg13_1, (64, 640), (1, 64), 0), out=buf16)
        del arg13_1
        buf17 = empty_strided_cuda((22*s1*s2, 640), (640, 1), torch.float16)
        # Source Nodes: [result_9], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf10, (22*s1*s2, 640), (640, 1), 0), reinterpret_tensor(arg14_1, (640, 640), (1, 640), 0), out=buf17)
        del arg14_1
        buf18 = buf15; del buf15  # reuse
        # Source Nodes: [l__self___transformer_blocks_0_attn1_to_v_lora_a_default_0], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf10, (22*s1*s2, 640), (640, 1), 0), reinterpret_tensor(arg15_1, (640, 64), (1, 640), 0), out=buf18)
        del arg15_1
        buf19 = reinterpret_tensor(buf10, (22*s1*s2, 640), (640, 1), 0); del buf10  # reuse
        # Source Nodes: [l__self___transformer_blocks_0_attn1_to_v_lora_b_default_0], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf18, (22*s1*s2, 64), (64, 1), 0), reinterpret_tensor(arg16_1, (64, 640), (1, 64), 0), out=buf19)
        del arg16_1
        buf20 = reinterpret_tensor(buf11, (22, 8, s1*s2, 80), (640*s1*s2, 80, 640, 1), 0); del buf11  # reuse
        # Source Nodes: [hidden_states_4], Original ATen: [aten._scaled_dot_product_flash_attention]
        triton_poi_fused__scaled_dot_product_flash_attention_4_xnumel = 14080*s1*s2
        triton_poi_fused__scaled_dot_product_flash_attention_4.run(buf20, buf13, triton_poi_fused__scaled_dot_product_flash_attention_4_xnumel, grid=grid(triton_poi_fused__scaled_dot_product_flash_attention_4_xnumel), stream=stream0)
        del buf13
        buf21 = reinterpret_tensor(buf14, (22, 8, s1*s2, 80), (640*s1*s2, 80, 640, 1), 0); del buf14  # reuse
        # Source Nodes: [hidden_states_4], Original ATen: [aten._scaled_dot_product_flash_attention]
        triton_poi_fused__scaled_dot_product_flash_attention_4_xnumel = 14080*s1*s2
        triton_poi_fused__scaled_dot_product_flash_attention_4.run(buf21, buf16, triton_poi_fused__scaled_dot_product_flash_attention_4_xnumel, grid=grid(triton_poi_fused__scaled_dot_product_flash_attention_4_xnumel), stream=stream0)
        del buf16
        buf22 = reinterpret_tensor(buf17, (22, 8, s1*s2, 80), (640*s1*s2, 80, 640, 1), 0); del buf17  # reuse
        # Source Nodes: [hidden_states_4], Original ATen: [aten._scaled_dot_product_flash_attention]
        triton_poi_fused__scaled_dot_product_flash_attention_4_xnumel = 14080*s1*s2
        triton_poi_fused__scaled_dot_product_flash_attention_4.run(buf22, buf19, triton_poi_fused__scaled_dot_product_flash_attention_4_xnumel, grid=grid(triton_poi_fused__scaled_dot_product_flash_attention_4_xnumel), stream=stream0)
        del buf19
        # Source Nodes: [hidden_states_4], Original ATen: [aten._scaled_dot_product_flash_attention]
        buf23 = aten._scaled_dot_product_flash_attention.default(buf20, buf21, buf22, scale=0.11180339887498948)
        del buf20
        buf24 = buf23[0]
        del buf23
        buf29 = reinterpret_tensor(buf22, (22*s1*s2, 640), (640, 1), 0); del buf22  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf24, (22*s1*s2, 640), (640, 1), 0), reinterpret_tensor(arg17_1, (640, 640), (1, 640), 0), out=buf29)
        del arg17_1
        buf30 = buf18; del buf18  # reuse
        # Source Nodes: [l__self___transformer_blocks_0_attn1_to_out_0_lora_a_default_0], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf24, (22*s1*s2, 640), (640, 1), 0), reinterpret_tensor(arg19_1, (640, 64), (1, 640), 0), out=buf30)
        del arg19_1
        buf31 = reinterpret_tensor(buf24, (22*s1*s2, 640), (640, 1), 0); del buf24  # reuse
        # Source Nodes: [l__self___transformer_blocks_0_attn1_to_out_0_lora_b_default_0], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf30, (22*s1*s2, 64), (64, 1), 0), reinterpret_tensor(arg20_1, (64, 640), (1, 64), 0), out=buf31)
        del arg20_1
        buf32 = reinterpret_tensor(buf29, (22, s1*s2, 640), (640*s1*s2, 640, 1), 0); del buf29  # reuse
        buf36 = reinterpret_tensor(buf21, (22, s1*s2, 640), (640*s1*s2, 640, 1), 0); del buf21  # reuse
        # Source Nodes: [attn_output, hidden_states_10, mul_5, residual_2, result_13], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm]
        triton_per_fused_add_div_mul_native_layer_norm_5_xnumel = 22*s1*s2
        triton_per_fused_add_div_mul_native_layer_norm_5.run(buf32, arg18_1, buf31, buf4, arg3_1, buf6, arg21_1, arg22_1, buf36, ps0, s1, s2, triton_per_fused_add_div_mul_native_layer_norm_5_xnumel, 640, grid=grid(triton_per_fused_add_div_mul_native_layer_norm_5_xnumel), stream=stream0)
        del arg18_1
        del arg21_1
        del arg22_1
        del arg3_1
        buf37 = reinterpret_tensor(buf6, (22*s1*s2, 640), (640, 1), 0); del buf6  # reuse
        # Source Nodes: [result_15], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf36, (22*s1*s2, 640), (640, 1), 0), reinterpret_tensor(arg23_1, (640, 640), (1, 640), 0), out=buf37)
        del arg23_1
        buf38 = buf30; del buf30  # reuse
        # Source Nodes: [l__self___transformer_blocks_0_attn2_to_q_lora_a_default_0], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf36, (22*s1*s2, 640), (640, 1), 0), reinterpret_tensor(arg24_1, (640, 64), (1, 640), 0), out=buf38)
        del arg24_1
        buf39 = reinterpret_tensor(buf36, (22*s1*s2, 640), (640, 1), 0); del buf36  # reuse
        # Source Nodes: [l__self___transformer_blocks_0_attn2_to_q_lora_b_default_0], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf38, (22*s1*s2, 64), (64, 1), 0), reinterpret_tensor(arg25_1, (64, 640), (1, 64), 0), out=buf39)
        del arg25_1
        buf40 = empty_strided_cuda((1694, 640), (640, 1), torch.float16)
        # Source Nodes: [result_18], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(arg53_1, (1694, 768), (768, 1), 0), reinterpret_tensor(arg26_1, (768, 640), (1, 768), 0), out=buf40)
        del arg26_1
        buf41 = empty_strided_cuda((1694, 64), (64, 1), torch.float16)
        # Source Nodes: [l__self___transformer_blocks_0_attn2_to_k_lora_a_default_0], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(arg53_1, (1694, 768), (768, 1), 0), reinterpret_tensor(arg27_1, (768, 64), (1, 768), 0), out=buf41)
        del arg27_1
        buf42 = empty_strided_cuda((1694, 640), (640, 1), torch.float16)
        # Source Nodes: [l__self___transformer_blocks_0_attn2_to_k_lora_b_default_0], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf41, (1694, 64), (64, 1), 0), reinterpret_tensor(arg28_1, (64, 640), (1, 64), 0), out=buf42)
        del arg28_1
        buf43 = empty_strided_cuda((1694, 640), (640, 1), torch.float16)
        # Source Nodes: [result_21], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(arg53_1, (1694, 768), (768, 1), 0), reinterpret_tensor(arg29_1, (768, 640), (1, 768), 0), out=buf43)
        del arg29_1
        buf44 = buf41; del buf41  # reuse
        # Source Nodes: [l__self___transformer_blocks_0_attn2_to_v_lora_a_default_0], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(arg53_1, (1694, 768), (768, 1), 0), reinterpret_tensor(arg30_1, (768, 64), (1, 768), 0), out=buf44)
        del arg30_1
        del arg53_1
        buf45 = empty_strided_cuda((1694, 640), (640, 1), torch.float16)
        # Source Nodes: [l__self___transformer_blocks_0_attn2_to_v_lora_b_default_0], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf44, (1694, 64), (64, 1), 0), reinterpret_tensor(arg31_1, (64, 640), (1, 64), 0), out=buf45)
        del arg31_1
        del buf44
        buf46 = reinterpret_tensor(buf37, (22, 8, s1*s2, 80), (640*s1*s2, 80, 640, 1), 0); del buf37  # reuse
        # Source Nodes: [hidden_states_11], Original ATen: [aten._scaled_dot_product_flash_attention]
        triton_poi_fused__scaled_dot_product_flash_attention_4_xnumel = 14080*s1*s2
        triton_poi_fused__scaled_dot_product_flash_attention_4.run(buf46, buf39, triton_poi_fused__scaled_dot_product_flash_attention_4_xnumel, grid=grid(triton_poi_fused__scaled_dot_product_flash_attention_4_xnumel), stream=stream0)
        buf47 = reinterpret_tensor(buf40, (22, 8, 77, 80), (49280, 80, 640, 1), 0); del buf40  # reuse
        # Source Nodes: [hidden_states_11], Original ATen: [aten._scaled_dot_product_flash_attention]
        triton_poi_fused__scaled_dot_product_flash_attention_6.run(buf47, buf42, 1084160, grid=grid(1084160), stream=stream0)
        del buf42
        buf48 = reinterpret_tensor(buf43, (22, 8, 77, 80), (49280, 80, 640, 1), 0); del buf43  # reuse
        # Source Nodes: [hidden_states_11], Original ATen: [aten._scaled_dot_product_flash_attention]
        triton_poi_fused__scaled_dot_product_flash_attention_6.run(buf48, buf45, 1084160, grid=grid(1084160), stream=stream0)
        del buf45
        # Source Nodes: [hidden_states_11], Original ATen: [aten._scaled_dot_product_flash_attention]
        buf49 = aten._scaled_dot_product_flash_attention.default(buf46, buf47, buf48, scale=0.11180339887498948)
        del buf47
        del buf48
        buf50 = buf49[0]
        del buf49
        buf55 = reinterpret_tensor(buf46, (22*s1*s2, 640), (640, 1), 0); del buf46  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf50, (22*s1*s2, 640), (640, 1), 0), reinterpret_tensor(arg32_1, (640, 640), (1, 640), 0), out=buf55)
        del arg32_1
        buf56 = buf38; del buf38  # reuse
        # Source Nodes: [l__self___transformer_blocks_0_attn2_to_out_0_lora_a_default_0], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf50, (22*s1*s2, 640), (640, 1), 0), reinterpret_tensor(arg34_1, (640, 64), (1, 640), 0), out=buf56)
        del arg34_1
        buf57 = reinterpret_tensor(buf50, (22*s1*s2, 640), (640, 1), 0); del buf50  # reuse
        # Source Nodes: [l__self___transformer_blocks_0_attn2_to_out_0_lora_b_default_0], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf56, (22*s1*s2, 64), (64, 1), 0), reinterpret_tensor(arg35_1, (64, 640), (1, 64), 0), out=buf57)
        del arg35_1
        buf61 = reinterpret_tensor(buf39, (22, s1*s2, 640), (640*s1*s2, 640, 1), 0); del buf39  # reuse
        # Source Nodes: [attn_output_1, hidden_states_17, mul_9, norm_hidden_states_2, result_25], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm]
        triton_per_fused_add_div_mul_native_layer_norm_7_xnumel = 22*s1*s2
        triton_per_fused_add_div_mul_native_layer_norm_7.run(buf55, arg33_1, buf57, buf32, arg36_1, arg37_1, buf61, triton_per_fused_add_div_mul_native_layer_norm_7_xnumel, 640, grid=grid(triton_per_fused_add_div_mul_native_layer_norm_7_xnumel), stream=stream0)
        del arg36_1
        del arg37_1
        buf62 = empty_strided_cuda((22*s1*s2, 5120), (5120, 1), torch.float16)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf61, (22*s1*s2, 640), (640, 1), 0), reinterpret_tensor(arg38_1, (640, 5120), (1, 640), 0), out=buf62)
        del arg38_1
        buf63 = buf56; del buf56  # reuse
        # Source Nodes: [l__self___transformer_blocks_0_ff_net_0_proj_lora_a_default_0], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf61, (22*s1*s2, 640), (640, 1), 0), reinterpret_tensor(arg40_1, (640, 64), (1, 640), 0), out=buf63)
        del arg40_1
        buf64 = empty_strided_cuda((22*s1*s2, 5120), (5120, 1), torch.float16)
        # Source Nodes: [l__self___transformer_blocks_0_ff_net_0_proj_lora_b_default_0], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf63, (22*s1*s2, 64), (64, 1), 0), reinterpret_tensor(arg41_1, (64, 5120), (1, 64), 0), out=buf64)
        del arg41_1
        buf65 = empty_strided_cuda((22, s1*s2, 2560), (2560*s1*s2, 2560, 1), torch.float16)
        # Source Nodes: [gelu, hidden_states_20], Original ATen: [aten.gelu, aten.mul]
        triton_poi_fused_gelu_mul_8_xnumel = 56320*s1*s2
        triton_poi_fused_gelu_mul_8.run(buf62, arg39_1, buf64, buf65, triton_poi_fused_gelu_mul_8_xnumel, grid=grid(triton_poi_fused_gelu_mul_8_xnumel), stream=stream0)
        del arg39_1
        del buf62
        del buf64
        buf66 = reinterpret_tensor(buf61, (22*s1*s2, 640), (640, 1), 0); del buf61  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf65, (22*s1*s2, 2560), (2560, 1), 0), reinterpret_tensor(arg42_1, (2560, 640), (1, 2560), 0), out=buf66)
        del arg42_1
        buf67 = buf63; del buf63  # reuse
        # Source Nodes: [l__self___transformer_blocks_0_ff_net_2_lora_a_default_0], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf65, (22*s1*s2, 2560), (2560, 1), 0), reinterpret_tensor(arg44_1, (2560, 64), (1, 2560), 0), out=buf67)
        del arg44_1
        del buf65
        buf68 = reinterpret_tensor(buf4, (22*s1*s2, 640), (640, 1), 0); del buf4  # reuse
        # Source Nodes: [l__self___transformer_blocks_0_ff_net_2_lora_b_default_0], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf67, (22*s1*s2, 64), (64, 1), 0), reinterpret_tensor(arg45_1, (64, 640), (1, 64), 0), out=buf68)
        del arg45_1
        del buf67
        buf69 = reinterpret_tensor(buf31, (22, 640, s1, s2), (640*s1*s2, s1*s2, s2, 1), 0); del buf31  # reuse
        # Source Nodes: [hidden_states_25], Original ATen: [aten.clone]
        triton_poi_fused_clone_9_ynumel = 22*s1*s2
        triton_poi_fused_clone_9.run(buf66, arg43_1, buf68, buf55, arg33_1, buf57, buf32, buf69, ps0, s1, s2, triton_poi_fused_clone_9_ynumel, 640, grid=grid(triton_poi_fused_clone_9_ynumel, 640), stream=stream0)
        del arg33_1
        del arg43_1
        del buf32
        del buf55
        del buf57
        del buf66
        del buf68
        # Source Nodes: [result_33], Original ATen: [aten.convolution]
        buf70 = extern_kernels.convolution(buf69, arg46_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf70, (22, 640, s1, s2), (640*s1*s2, s1*s2, s2, 1))
        del arg46_1
        # Source Nodes: [l__self___proj_out_lora_a_default_0], Original ATen: [aten.convolution]
        buf71 = extern_kernels.convolution(buf69, arg48_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf71, (22, 64, s1, s2), (64*s1*s2, s1*s2, s2, 1))
        del arg48_1
        del buf69
        # Source Nodes: [l__self___proj_out_lora_b_default_0], Original ATen: [aten.convolution]
        buf72 = extern_kernels.convolution(buf71, arg49_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf72, (22, 640, s1, s2), (640*s1*s2, s1*s2, s2, 1))
        del arg49_1
        del buf71
        buf73 = buf70; del buf70  # reuse
        # Source Nodes: [mul_13, output, result_33, result_34], Original ATen: [aten.add, aten.convolution, aten.mul]
        triton_poi_fused_add_convolution_mul_10_xnumel = 14080*s1*s2
        triton_poi_fused_add_convolution_mul_10.run(buf73, arg47_1, buf72, arg52_1, ps0, triton_poi_fused_add_convolution_mul_10_xnumel, grid=grid(triton_poi_fused_add_convolution_mul_10_xnumel), stream=stream0)
        del arg47_1
        del arg52_1
        del buf72
    return (buf73, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg1_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg2_1 = rand_strided((640, 640, 1, 1), (640, 1, 1, 1), device='cuda:0', dtype=torch.float16)
    arg3_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg4_1 = rand_strided((64, 640, 1, 1), (640, 1, 1, 1), device='cuda:0', dtype=torch.float16)
    arg5_1 = rand_strided((640, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float16)
    arg6_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg7_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg8_1 = rand_strided((640, 640), (640, 1), device='cuda:0', dtype=torch.float16)
    arg9_1 = rand_strided((64, 640), (640, 1), device='cuda:0', dtype=torch.float16)
    arg10_1 = rand_strided((640, 64), (64, 1), device='cuda:0', dtype=torch.float16)
    arg11_1 = rand_strided((640, 640), (640, 1), device='cuda:0', dtype=torch.float16)
    arg12_1 = rand_strided((64, 640), (640, 1), device='cuda:0', dtype=torch.float16)
    arg13_1 = rand_strided((640, 64), (64, 1), device='cuda:0', dtype=torch.float16)
    arg14_1 = rand_strided((640, 640), (640, 1), device='cuda:0', dtype=torch.float16)
    arg15_1 = rand_strided((64, 640), (640, 1), device='cuda:0', dtype=torch.float16)
    arg16_1 = rand_strided((640, 64), (64, 1), device='cuda:0', dtype=torch.float16)
    arg17_1 = rand_strided((640, 640), (640, 1), device='cuda:0', dtype=torch.float16)
    arg18_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg19_1 = rand_strided((64, 640), (640, 1), device='cuda:0', dtype=torch.float16)
    arg20_1 = rand_strided((640, 64), (64, 1), device='cuda:0', dtype=torch.float16)
    arg21_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg22_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg23_1 = rand_strided((640, 640), (640, 1), device='cuda:0', dtype=torch.float16)
    arg24_1 = rand_strided((64, 640), (640, 1), device='cuda:0', dtype=torch.float16)
    arg25_1 = rand_strided((640, 64), (64, 1), device='cuda:0', dtype=torch.float16)
    arg26_1 = rand_strided((640, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    arg27_1 = rand_strided((64, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    arg28_1 = rand_strided((640, 64), (64, 1), device='cuda:0', dtype=torch.float16)
    arg29_1 = rand_strided((640, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    arg30_1 = rand_strided((64, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    arg31_1 = rand_strided((640, 64), (64, 1), device='cuda:0', dtype=torch.float16)
    arg32_1 = rand_strided((640, 640), (640, 1), device='cuda:0', dtype=torch.float16)
    arg33_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg34_1 = rand_strided((64, 640), (640, 1), device='cuda:0', dtype=torch.float16)
    arg35_1 = rand_strided((640, 64), (64, 1), device='cuda:0', dtype=torch.float16)
    arg36_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg37_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg38_1 = rand_strided((5120, 640), (640, 1), device='cuda:0', dtype=torch.float16)
    arg39_1 = rand_strided((5120, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg40_1 = rand_strided((64, 640), (640, 1), device='cuda:0', dtype=torch.float16)
    arg41_1 = rand_strided((5120, 64), (64, 1), device='cuda:0', dtype=torch.float16)
    arg42_1 = rand_strided((640, 2560), (2560, 1), device='cuda:0', dtype=torch.float16)
    arg43_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg44_1 = rand_strided((64, 2560), (2560, 1), device='cuda:0', dtype=torch.float16)
    arg45_1 = rand_strided((640, 64), (64, 1), device='cuda:0', dtype=torch.float16)
    arg46_1 = rand_strided((640, 640, 1, 1), (640, 1, 1, 1), device='cuda:0', dtype=torch.float16)
    arg47_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg48_1 = rand_strided((64, 640, 1, 1), (640, 1, 1, 1), device='cuda:0', dtype=torch.float16)
    arg49_1 = rand_strided((640, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float16)
    arg50_1 = 15
    arg51_1 = 45
    arg52_1 = rand_strided((22, 640, 15, 45), (432000, 675, 45, 1), device='cuda:0', dtype=torch.float16)
    arg53_1 = rand_strided((22, 77, 768), (59136, 768, 1), device='cuda:0', dtype=torch.float16)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
