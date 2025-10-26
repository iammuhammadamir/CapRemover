
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


# kernel path: /workspace/caption-remover-main/src/stages/inpaint/../../../.torch_compile_cache/lm/clm5z5pftrivoivslxyktan2nppc6h7s2xhlil3lp6t7bdzq5k3m.py
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
    size_hints=[1024, 2048],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
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
        tmp0 = tl.load(in_ptr0 + (r1 + (40*ks0*ks1*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
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


# kernel path: /workspace/caption-remover-main/src/stages/inpaint/../../../.torch_compile_cache/72/c723gknmmjtv7c5rf2wxom53ey62pfewyq3nkd5osgcu2hntnci5.py
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
    size_hints=[2097152], 
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


# kernel path: /workspace/caption-remover-main/src/stages/inpaint/../../../.torch_compile_cache/ud/cude35jshqhz5ct57aam3djyhh2li7hivymg5ain5e6bfgciezqe.py
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
    size_hints=[16384, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32', 8: 'i32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_2', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': 'a1fa794fabf29fbd973066d9ab65d9c16aea0ed2e098514ffda941ec2a0f87ad'}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, ks0, ks1, ks2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % ks0
    x4 = (xindex // ks0)
    x1 = (xindex // ks0) % 10
    tmp9_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp9_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp9_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x5 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (ks1*ks2*r3) + (128*ks1*ks2*x4)), rmask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (r3 + (128*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp3 = tl.load(in_ptr2 + (x0 + (ks1*ks2*r3) + (128*ks1*ks2*x4)), rmask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
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
    tl.store(out_ptr0 + (x5), tmp9, xmask)
    tl.store(out_ptr1 + (x5), tmp10, xmask)
    tl.store(out_ptr2 + (x5), tmp11, xmask)
''', device_str='cuda')


# kernel path: /workspace/caption-remover-main/src/stages/inpaint/../../../.torch_compile_cache/sz/csziwyfk6pvmltftdofpube5i4ndpvylunzr3jqlesl7u7zvga7l.py
# Source Nodes: [encoder_hidden_states], Original ATen: [aten.native_layer_norm]
# encoder_hidden_states => clone, convert_element_type_4, var_mean_1
triton_per_fused_native_layer_norm_3 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.persistent_reduction(
    size_hints=[2048, 16],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32', 7: 'i32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_3', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': 'a1fa794fabf29fbd973066d9ab65d9c16aea0ed2e098514ffda941ec2a0f87ad'}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, ks0, ks1, ks2, xnumel, rnumel, XBLOCK : tl.constexpr):
    rnumel = 10
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % ks0
    x1 = (xindex // ks0)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (ks1*ks2*r2) + (10*ks1*ks2*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (ks1*ks2*r2) + (10*ks1*ks2*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + (ks1*ks2*r2) + (10*ks1*ks2*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
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
    tl.store(out_ptr0 + (x3), tmp13, xmask)
    tl.store(out_ptr1 + (x3), tmp14, xmask)
''', device_str='cuda')


# kernel path: /workspace/caption-remover-main/src/stages/inpaint/../../../.torch_compile_cache/27/c27zeinoepmswlg4nhkek5zzv7w6j37ozelqwowjzj3vrugotrjs.py
# Source Nodes: [encoder_hidden_states], Original ATen: [aten.native_layer_norm]
# encoder_hidden_states => add_3, add_4, clone, convert_element_type_4, convert_element_type_5, mul_5, mul_6, rsqrt_1, sub_1, var_mean_1
triton_poi_fused_native_layer_norm_4 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.pointwise(
    size_hints=[32768, 64], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp32', 4: '*fp32', 5: '*fp16', 6: '*fp16', 7: '*fp16', 8: 'i32', 9: 'i32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_layer_norm_4', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': 'a1fa794fabf29fbd973066d9ab65d9c16aea0ed2e098514ffda941ec2a0f87ad'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, ks0, ks1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 28160
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
    tmp11 = 1280.0
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
    tl.store(out_ptr0 + (y0 + (1280*x2) + (1280*ks0*ks1*y1)), tmp23, xmask & ymask)
''', device_str='cuda')


# kernel path: /workspace/caption-remover-main/src/stages/inpaint/../../../.torch_compile_cache/mx/cmxfdni4idgfpjic7wtlch6zojgwtyomsjzcijhu2swl4y7k6l6o.py
# Source Nodes: [hidden_states_4], Original ATen: [aten._scaled_dot_product_flash_attention]
# hidden_states_4 => _scaled_dot_product_flash_attention
triton_poi_fused__scaled_dot_product_flash_attention_5 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__scaled_dot_product_flash_attention_5', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'backend_hash': 'a1fa794fabf29fbd973066d9ab65d9c16aea0ed2e098514ffda941ec2a0f87ad'},
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


# kernel path: /workspace/caption-remover-main/src/stages/inpaint/../../../.torch_compile_cache/da/cdavoca26xz5onnfwdxlss7dmj5mnffm3kwshcbzpgb4o6y6xsv5.py
# Source Nodes: [attn_output, hidden_states_10, mul_5, residual_2, result_13], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm]
# attn_output => div
# hidden_states_10 => add_9
# mul_5 => mul_22
# residual_2 => add_10, add_11, convert_element_type_31, convert_element_type_32, mul_23, mul_24, rsqrt_2, sub_2, var_mean_2
# result_13 => add_8
triton_red_fused_add_div_mul_native_layer_norm_6 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.reduction(
    size_hints=[2048, 2048],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: '*fp16', 5: '*fp16', 6: '*fp16', 7: '*fp16', 8: '*fp16', 9: 'i32', 10: 'i32', 11: 'i32', 12: 'i32', 13: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 13), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(13,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_div_mul_native_layer_norm_6', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'backend_hash': 'a1fa794fabf29fbd973066d9ab65d9c16aea0ed2e098514ffda941ec2a0f87ad'}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr2, ks0, ks1, ks2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    rnumel = 1280
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x3 = xindex
    x0 = xindex % ks0
    x1 = (xindex // ks0)
    tmp18_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp18_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp18_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_out_ptr0 + (r2 + (1280*x3)), rmask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tl.load(in_ptr0 + (r2), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp3 = tl.load(in_ptr1 + (r2 + (1280*x3)), rmask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp9 = tl.load(in_ptr2 + (x0 + (ks1*ks2*r2) + (1280*ks1*ks2*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp10 = tl.load(in_ptr3 + (r2), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp12 = tl.load(in_ptr4 + (x0 + (ks1*ks2*r2) + (1280*ks1*ks2*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
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
        tmp17 = tl.broadcast_to(tmp16, [XBLOCK, RBLOCK])
        tmp18_mean_next, tmp18_m2_next, tmp18_weight_next = triton_helpers.welford_reduce(
            tmp17, tmp18_mean, tmp18_m2, tmp18_weight, roffset == 0
        )
        tmp18_mean = tl.where(rmask & xmask, tmp18_mean_next, tmp18_mean)
        tmp18_m2 = tl.where(rmask & xmask, tmp18_m2_next, tmp18_m2)
        tmp18_weight = tl.where(rmask & xmask, tmp18_weight_next, tmp18_weight)
        tl.store(in_out_ptr0 + (r2 + (1280*x3)), tmp15, rmask & xmask)
    tmp18_tmp, tmp19_tmp, tmp20_tmp = triton_helpers.welford(
        tmp18_mean, tmp18_m2, tmp18_weight, 1
    )
    tmp18 = tmp18_tmp[:, None]
    tmp19 = tmp19_tmp[:, None]
    tmp20 = tmp20_tmp[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp21 = tl.load(in_out_ptr0 + (r2 + (1280*x3)), rmask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp30 = tl.load(in_ptr5 + (r2), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp33 = tl.load(in_ptr6 + (r2), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp22 = tmp21.to(tl.float32)
        tmp23 = tmp22 - tmp18
        tmp24 = 1280.0
        tmp25 = tmp19 / tmp24
        tmp26 = 1e-05
        tmp27 = tmp25 + tmp26
        tmp28 = libdevice.rsqrt(tmp27)
        tmp29 = tmp23 * tmp28
        tmp31 = tmp30.to(tl.float32)
        tmp32 = tmp29 * tmp31
        tmp34 = tmp33.to(tl.float32)
        tmp35 = tmp32 + tmp34
        tmp36 = tmp35.to(tl.float32)
        tl.store(out_ptr2 + (r2 + (1280*x3)), tmp36, rmask & xmask)
''', device_str='cuda')


# kernel path: /workspace/caption-remover-main/src/stages/inpaint/../../../.torch_compile_cache/o2/co24h4kyphe6yj74ihadkztedzs4tvqqlfp7uncf3r7dsdt2bj3x.py
# Source Nodes: [hidden_states_11], Original ATen: [aten._scaled_dot_product_flash_attention]
# hidden_states_11 => _scaled_dot_product_flash_attention_1
triton_poi_fused__scaled_dot_product_flash_attention_7 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.pointwise(
    size_hints=[4194304], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp16', 1: '*fp16', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__scaled_dot_product_flash_attention_7', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'backend_hash': 'a1fa794fabf29fbd973066d9ab65d9c16aea0ed2e098514ffda941ec2a0f87ad'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2168320
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


# kernel path: /workspace/caption-remover-main/src/stages/inpaint/../../../.torch_compile_cache/ss/csssttu5zw2vrqocswjwai4uxpmr4bguz5afi4obrqtioh6cvbqd.py
# Source Nodes: [attn_output_1, hidden_states_17, mul_9, norm_hidden_states_2, result_25], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm]
# attn_output_1 => div_1
# hidden_states_17 => add_16
# mul_9 => mul_34
# norm_hidden_states_2 => add_17, add_18, convert_element_type_58, convert_element_type_59, mul_35, mul_36, rsqrt_3, sub_3, var_mean_3
# result_25 => add_15
triton_red_fused_add_div_mul_native_layer_norm_8 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.reduction(
    size_hints=[2048, 2048],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: '*fp16', 5: '*fp16', 6: '*fp16', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_div_mul_native_layer_norm_8', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': 'a1fa794fabf29fbd973066d9ab65d9c16aea0ed2e098514ffda941ec2a0f87ad'}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    rnumel = 1280
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    tmp13_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp13_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp13_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (1280*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp3 = tl.load(in_ptr2 + (r1 + (1280*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp9 = tl.load(in_ptr3 + (r1 + (1280*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp2 = tmp0 + tmp1
        tmp4 = 0.125
        tmp5 = tmp3 * tmp4
        tmp6 = tmp2 + tmp5
        tmp7 = 1.0
        tmp8 = tmp6 / tmp7
        tmp10 = tmp8 + tmp9
        tmp11 = tmp10.to(tl.float32)
        tmp12 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
        tmp13_mean_next, tmp13_m2_next, tmp13_weight_next = triton_helpers.welford_reduce(
            tmp12, tmp13_mean, tmp13_m2, tmp13_weight, roffset == 0
        )
        tmp13_mean = tl.where(rmask & xmask, tmp13_mean_next, tmp13_mean)
        tmp13_m2 = tl.where(rmask & xmask, tmp13_m2_next, tmp13_m2)
        tmp13_weight = tl.where(rmask & xmask, tmp13_weight_next, tmp13_weight)
    tmp13_tmp, tmp14_tmp, tmp15_tmp = triton_helpers.welford(
        tmp13_mean, tmp13_m2, tmp13_weight, 1
    )
    tmp13 = tmp13_tmp[:, None]
    tmp14 = tmp14_tmp[:, None]
    tmp15 = tmp15_tmp[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp16 = tl.load(in_ptr0 + (r1 + (1280*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp17 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp19 = tl.load(in_ptr2 + (r1 + (1280*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp25 = tl.load(in_ptr3 + (r1 + (1280*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp35 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp38 = tl.load(in_ptr5 + (r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp18 = tmp16 + tmp17
        tmp20 = 0.125
        tmp21 = tmp19 * tmp20
        tmp22 = tmp18 + tmp21
        tmp23 = 1.0
        tmp24 = tmp22 / tmp23
        tmp26 = tmp24 + tmp25
        tmp27 = tmp26.to(tl.float32)
        tmp28 = tmp27 - tmp13
        tmp29 = 1280.0
        tmp30 = tmp14 / tmp29
        tmp31 = 1e-05
        tmp32 = tmp30 + tmp31
        tmp33 = libdevice.rsqrt(tmp32)
        tmp34 = tmp28 * tmp33
        tmp36 = tmp35.to(tl.float32)
        tmp37 = tmp34 * tmp36
        tmp39 = tmp38.to(tl.float32)
        tmp40 = tmp37 + tmp39
        tmp41 = tmp40.to(tl.float32)
        tl.store(out_ptr2 + (r1 + (1280*x0)), tmp41, rmask & xmask)
''', device_str='cuda')


# kernel path: /workspace/caption-remover-main/src/stages/inpaint/../../../.torch_compile_cache/2e/c2ehcmintow4q4zkyybb7tdqj2k6qucsy5wnbkk7m2vubmdtquhf.py
# Source Nodes: [gelu, hidden_states_20], Original ATen: [aten.gelu, aten.mul]
# gelu => add_20, convert_element_type_67, convert_element_type_68, erf, mul_41, mul_42, mul_43
# hidden_states_20 => mul_44
triton_poi_fused_gelu_mul_9 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_mul_9', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': 'a1fa794fabf29fbd973066d9ab65d9c16aea0ed2e098514ffda941ec2a0f87ad'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 5120
    x1 = (xindex // 5120)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (10240*x1)), None).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp3 = tl.load(in_ptr2 + (x0 + (10240*x1)), None).to(tl.float32)
    tmp7 = tl.load(in_ptr0 + (5120 + x0 + (10240*x1)), None).to(tl.float32)
    tmp8 = tl.load(in_ptr1 + (5120 + x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp10 = tl.load(in_ptr2 + (5120 + x0 + (10240*x1)), None).to(tl.float32)
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
    tl.store(out_ptr0 + (x2), tmp23, None)
''', device_str='cuda')


# kernel path: /workspace/caption-remover-main/src/stages/inpaint/../../../.torch_compile_cache/xw/cxwvfu7cqljsrplfa6wbnzhmpi47d5pi3umlmw2b52uxp6lsijxj.py
# Source Nodes: [hidden_states_25], Original ATen: [aten.clone]
# hidden_states_25 => clone_4
triton_poi_fused_clone_10 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.pointwise(
    size_hints=[2048, 2048], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: '*fp16', 5: '*fp16', 6: '*fp16', 7: '*fp16', 8: 'i32', 9: 'i32', 10: 'i32', 11: 'i32', 12: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 12), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(12,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_10', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': 'a1fa794fabf29fbd973066d9ab65d9c16aea0ed2e098514ffda941ec2a0f87ad'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, ks0, ks1, ks2, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    xnumel = 1280
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
    tmp0 = tl.load(in_ptr0 + (x2 + (1280*y3)), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x2), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp3 = tl.load(in_ptr2 + (x2 + (1280*y3)), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp7 = tl.load(in_ptr3 + (x2 + (1280*y3)), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp8 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp10 = tl.load(in_ptr5 + (x2 + (1280*y3)), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp15 = tl.load(in_ptr6 + (x2 + (1280*y3)), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
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
    tl.store(out_ptr0 + (y0 + (ks1*ks2*x2) + (1280*ks1*ks2*y1)), tmp17, xmask & ymask)
''', device_str='cuda')


# kernel path: /workspace/caption-remover-main/src/stages/inpaint/../../../.torch_compile_cache/7s/c7syqd4wq7hjhsaubucb3skbqxwb4v7txuhfxezrjmys34ujtsz6.py
# Source Nodes: [mul_13, output, result_33, result_34], Original ATen: [aten.add, aten.convolution, aten.mul]
# mul_13 => mul_49
# output => add_24
# result_33 => convolution_3
# result_34 => add_23
triton_poi_fused_add_convolution_mul_11 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_mul_11', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'backend_hash': 'a1fa794fabf29fbd973066d9ab65d9c16aea0ed2e098514ffda941ec2a0f87ad'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, ks0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // ks0) % 1280
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
    assert_size_stride(arg0_1, (1280, ), (1, ))
    assert_size_stride(arg1_1, (1280, ), (1, ))
    assert_size_stride(arg2_1, (1280, 1280, 1, 1), (1280, 1, 1, 1))
    assert_size_stride(arg3_1, (1280, ), (1, ))
    assert_size_stride(arg4_1, (64, 1280, 1, 1), (1280, 1, 1, 1))
    assert_size_stride(arg5_1, (1280, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(arg6_1, (1280, ), (1, ))
    assert_size_stride(arg7_1, (1280, ), (1, ))
    assert_size_stride(arg8_1, (1280, 1280), (1280, 1))
    assert_size_stride(arg9_1, (64, 1280), (1280, 1))
    assert_size_stride(arg10_1, (1280, 64), (64, 1))
    assert_size_stride(arg11_1, (1280, 1280), (1280, 1))
    assert_size_stride(arg12_1, (64, 1280), (1280, 1))
    assert_size_stride(arg13_1, (1280, 64), (64, 1))
    assert_size_stride(arg14_1, (1280, 1280), (1280, 1))
    assert_size_stride(arg15_1, (64, 1280), (1280, 1))
    assert_size_stride(arg16_1, (1280, 64), (64, 1))
    assert_size_stride(arg17_1, (1280, 1280), (1280, 1))
    assert_size_stride(arg18_1, (1280, ), (1, ))
    assert_size_stride(arg19_1, (64, 1280), (1280, 1))
    assert_size_stride(arg20_1, (1280, 64), (64, 1))
    assert_size_stride(arg21_1, (1280, ), (1, ))
    assert_size_stride(arg22_1, (1280, ), (1, ))
    assert_size_stride(arg23_1, (1280, 1280), (1280, 1))
    assert_size_stride(arg24_1, (64, 1280), (1280, 1))
    assert_size_stride(arg25_1, (1280, 64), (64, 1))
    assert_size_stride(arg26_1, (1280, 768), (768, 1))
    assert_size_stride(arg27_1, (64, 768), (768, 1))
    assert_size_stride(arg28_1, (1280, 64), (64, 1))
    assert_size_stride(arg29_1, (1280, 768), (768, 1))
    assert_size_stride(arg30_1, (64, 768), (768, 1))
    assert_size_stride(arg31_1, (1280, 64), (64, 1))
    assert_size_stride(arg32_1, (1280, 1280), (1280, 1))
    assert_size_stride(arg33_1, (1280, ), (1, ))
    assert_size_stride(arg34_1, (64, 1280), (1280, 1))
    assert_size_stride(arg35_1, (1280, 64), (64, 1))
    assert_size_stride(arg36_1, (1280, ), (1, ))
    assert_size_stride(arg37_1, (1280, ), (1, ))
    assert_size_stride(arg38_1, (10240, 1280), (1280, 1))
    assert_size_stride(arg39_1, (10240, ), (1, ))
    assert_size_stride(arg40_1, (64, 1280), (1280, 1))
    assert_size_stride(arg41_1, (10240, 64), (64, 1))
    assert_size_stride(arg42_1, (1280, 5120), (5120, 1))
    assert_size_stride(arg43_1, (1280, ), (1, ))
    assert_size_stride(arg44_1, (64, 5120), (5120, 1))
    assert_size_stride(arg45_1, (1280, 64), (64, 1))
    assert_size_stride(arg46_1, (1280, 1280, 1, 1), (1280, 1, 1, 1))
    assert_size_stride(arg47_1, (1280, ), (1, ))
    assert_size_stride(arg48_1, (64, 1280, 1, 1), (1280, 1, 1, 1))
    assert_size_stride(arg49_1, (1280, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(arg52_1, (22, 1280, s1, s2), (1280*s1*s2, s1*s2, s2, 1))
    assert_size_stride(arg53_1, (22, 77, 768), (59136, 768, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((22, 32, 1, 1), (32, 1, 704, 704), torch.float32)
        buf1 = empty_strided_cuda((22, 32, 1, 1), (32, 1, 704, 704), torch.float32)
        # Source Nodes: [hidden_states], Original ATen: [aten.native_group_norm]
        triton_red_fused_native_group_norm_0_rnumel = 40*s1*s2
        stream0 = get_raw_stream(0)
        triton_red_fused_native_group_norm_0.run(arg52_1, buf0, buf1, s1, s2, 704, triton_red_fused_native_group_norm_0_rnumel, grid=grid(704), stream=stream0)
        ps0 = s1*s2
        ps1 = 1280*s1*s2
        buf3 = empty_strided_cuda((22, 1280, s1, s2), (1280*s1*s2, s1*s2, s2, 1), torch.float16)
        # Source Nodes: [hidden_states], Original ATen: [aten.native_group_norm]
        triton_poi_fused_native_group_norm_1_xnumel = 28160*s1*s2
        triton_poi_fused_native_group_norm_1.run(arg52_1, buf0, buf1, arg0_1, arg1_1, buf3, ps0, ps1, s1, s2, triton_poi_fused_native_group_norm_1_xnumel, grid=grid(triton_poi_fused_native_group_norm_1_xnumel), stream=stream0)
        del arg0_1
        del arg1_1
        del buf0
        del buf1
        # Source Nodes: [result], Original ATen: [aten.convolution]
        buf4 = extern_kernels.convolution(buf3, arg2_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf4, (22, 1280, s1, s2), (1280*s1*s2, s1*s2, s2, 1))
        del arg2_1
        # Source Nodes: [l__self___proj_in_lora_a_default_0], Original ATen: [aten.convolution]
        buf5 = extern_kernels.convolution(buf3, arg4_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf5, (22, 64, s1, s2), (64*s1*s2, s1*s2, s2, 1))
        del arg4_1
        # Source Nodes: [l__self___proj_in_lora_b_default_0], Original ATen: [aten.convolution]
        buf6 = extern_kernels.convolution(buf5, arg5_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf6, (22, 1280, s1, s2), (1280*s1*s2, s1*s2, s2, 1))
        del arg5_1
        buf7 = empty_strided_cuda((22, s1*s2, 1, 10), (10*s1*s2, 1, 220*s1*s2, s1*s2), torch.float32)
        buf8 = empty_strided_cuda((22, s1*s2, 1, 10), (10*s1*s2, 1, 220*s1*s2, s1*s2), torch.float32)
        buf9 = empty_strided_cuda((22, s1*s2, 1, 10), (10*s1*s2, 1, 220*s1*s2, s1*s2), torch.float32)
        # Source Nodes: [encoder_hidden_states], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_2_xnumel = 220*s1*s2
        triton_red_fused_native_layer_norm_2.run(buf4, arg3_1, buf6, buf7, buf8, buf9, ps0, s1, s2, triton_red_fused_native_layer_norm_2_xnumel, 128, grid=grid(triton_red_fused_native_layer_norm_2_xnumel), stream=stream0)
        buf10 = empty_strided_cuda((22, s1*s2, 1), (s1*s2, 1, 22*s1*s2), torch.float32)
        buf11 = empty_strided_cuda((22, s1*s2, 1), (s1*s2, 1, 22*s1*s2), torch.float32)
        # Source Nodes: [encoder_hidden_states], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_3_xnumel = 22*s1*s2
        triton_per_fused_native_layer_norm_3.run(buf7, buf8, buf9, buf10, buf11, ps0, s1, s2, triton_per_fused_native_layer_norm_3_xnumel, 10, grid=grid(triton_per_fused_native_layer_norm_3_xnumel), stream=stream0)
        del buf7
        del buf8
        del buf9
        buf13 = reinterpret_tensor(buf3, (22, s1*s2, 1280), (1280*s1*s2, 1280, 1), 0); del buf3  # reuse
        # Source Nodes: [encoder_hidden_states], Original ATen: [aten.native_layer_norm]
        triton_poi_fused_native_layer_norm_4_xnumel = s1*s2
        triton_poi_fused_native_layer_norm_4.run(buf4, arg3_1, buf6, buf10, buf11, arg6_1, arg7_1, buf13, s1, s2, 28160, triton_poi_fused_native_layer_norm_4_xnumel, grid=grid(28160, triton_poi_fused_native_layer_norm_4_xnumel), stream=stream0)
        del arg6_1
        del arg7_1
        del buf10
        del buf11
        buf14 = empty_strided_cuda((22*s1*s2, 1280), (1280, 1), torch.float16)
        # Source Nodes: [result_3], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf13, (22*s1*s2, 1280), (1280, 1), 0), reinterpret_tensor(arg8_1, (1280, 1280), (1, 1280), 0), out=buf14)
        del arg8_1
        buf15 = reinterpret_tensor(buf5, (22*s1*s2, 64), (64, 1), 0); del buf5  # reuse
        # Source Nodes: [l__self___transformer_blocks_0_attn1_to_q_lora_a_default_0], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf13, (22*s1*s2, 1280), (1280, 1), 0), reinterpret_tensor(arg9_1, (1280, 64), (1, 1280), 0), out=buf15)
        del arg9_1
        buf16 = empty_strided_cuda((22*s1*s2, 1280), (1280, 1), torch.float16)
        # Source Nodes: [l__self___transformer_blocks_0_attn1_to_q_lora_b_default_0], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf15, (22*s1*s2, 64), (64, 1), 0), reinterpret_tensor(arg10_1, (64, 1280), (1, 64), 0), out=buf16)
        del arg10_1
        buf17 = empty_strided_cuda((22*s1*s2, 1280), (1280, 1), torch.float16)
        # Source Nodes: [result_6], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf13, (22*s1*s2, 1280), (1280, 1), 0), reinterpret_tensor(arg11_1, (1280, 1280), (1, 1280), 0), out=buf17)
        del arg11_1
        buf18 = buf15; del buf15  # reuse
        # Source Nodes: [l__self___transformer_blocks_0_attn1_to_k_lora_a_default_0], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf13, (22*s1*s2, 1280), (1280, 1), 0), reinterpret_tensor(arg12_1, (1280, 64), (1, 1280), 0), out=buf18)
        del arg12_1
        buf19 = empty_strided_cuda((22*s1*s2, 1280), (1280, 1), torch.float16)
        # Source Nodes: [l__self___transformer_blocks_0_attn1_to_k_lora_b_default_0], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf18, (22*s1*s2, 64), (64, 1), 0), reinterpret_tensor(arg13_1, (64, 1280), (1, 64), 0), out=buf19)
        del arg13_1
        buf20 = empty_strided_cuda((22*s1*s2, 1280), (1280, 1), torch.float16)
        # Source Nodes: [result_9], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf13, (22*s1*s2, 1280), (1280, 1), 0), reinterpret_tensor(arg14_1, (1280, 1280), (1, 1280), 0), out=buf20)
        del arg14_1
        buf21 = buf18; del buf18  # reuse
        # Source Nodes: [l__self___transformer_blocks_0_attn1_to_v_lora_a_default_0], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf13, (22*s1*s2, 1280), (1280, 1), 0), reinterpret_tensor(arg15_1, (1280, 64), (1, 1280), 0), out=buf21)
        del arg15_1
        buf22 = reinterpret_tensor(buf13, (22*s1*s2, 1280), (1280, 1), 0); del buf13  # reuse
        # Source Nodes: [l__self___transformer_blocks_0_attn1_to_v_lora_b_default_0], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf21, (22*s1*s2, 64), (64, 1), 0), reinterpret_tensor(arg16_1, (64, 1280), (1, 64), 0), out=buf22)
        del arg16_1
        buf23 = reinterpret_tensor(buf14, (22, 8, s1*s2, 160), (1280*s1*s2, 160, 1280, 1), 0); del buf14  # reuse
        # Source Nodes: [hidden_states_4], Original ATen: [aten._scaled_dot_product_flash_attention]
        triton_poi_fused__scaled_dot_product_flash_attention_5_xnumel = 28160*s1*s2
        triton_poi_fused__scaled_dot_product_flash_attention_5.run(buf23, buf16, triton_poi_fused__scaled_dot_product_flash_attention_5_xnumel, grid=grid(triton_poi_fused__scaled_dot_product_flash_attention_5_xnumel), stream=stream0)
        del buf16
        buf24 = reinterpret_tensor(buf17, (22, 8, s1*s2, 160), (1280*s1*s2, 160, 1280, 1), 0); del buf17  # reuse
        # Source Nodes: [hidden_states_4], Original ATen: [aten._scaled_dot_product_flash_attention]
        triton_poi_fused__scaled_dot_product_flash_attention_5_xnumel = 28160*s1*s2
        triton_poi_fused__scaled_dot_product_flash_attention_5.run(buf24, buf19, triton_poi_fused__scaled_dot_product_flash_attention_5_xnumel, grid=grid(triton_poi_fused__scaled_dot_product_flash_attention_5_xnumel), stream=stream0)
        del buf19
        buf25 = reinterpret_tensor(buf20, (22, 8, s1*s2, 160), (1280*s1*s2, 160, 1280, 1), 0); del buf20  # reuse
        # Source Nodes: [hidden_states_4], Original ATen: [aten._scaled_dot_product_flash_attention]
        triton_poi_fused__scaled_dot_product_flash_attention_5_xnumel = 28160*s1*s2
        triton_poi_fused__scaled_dot_product_flash_attention_5.run(buf25, buf22, triton_poi_fused__scaled_dot_product_flash_attention_5_xnumel, grid=grid(triton_poi_fused__scaled_dot_product_flash_attention_5_xnumel), stream=stream0)
        del buf22
        # Source Nodes: [hidden_states_4], Original ATen: [aten._scaled_dot_product_flash_attention]
        buf26 = aten._scaled_dot_product_flash_attention.default(buf23, buf24, buf25, scale=0.07905694150420949)
        del buf23
        buf27 = buf26[0]
        del buf26
        buf32 = reinterpret_tensor(buf25, (22*s1*s2, 1280), (1280, 1), 0); del buf25  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf27, (22*s1*s2, 1280), (1280, 1), 0), reinterpret_tensor(arg17_1, (1280, 1280), (1, 1280), 0), out=buf32)
        del arg17_1
        buf33 = buf21; del buf21  # reuse
        # Source Nodes: [l__self___transformer_blocks_0_attn1_to_out_0_lora_a_default_0], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf27, (22*s1*s2, 1280), (1280, 1), 0), reinterpret_tensor(arg19_1, (1280, 64), (1, 1280), 0), out=buf33)
        del arg19_1
        buf34 = reinterpret_tensor(buf27, (22*s1*s2, 1280), (1280, 1), 0); del buf27  # reuse
        # Source Nodes: [l__self___transformer_blocks_0_attn1_to_out_0_lora_b_default_0], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf33, (22*s1*s2, 64), (64, 1), 0), reinterpret_tensor(arg20_1, (64, 1280), (1, 64), 0), out=buf34)
        del arg20_1
        buf35 = reinterpret_tensor(buf32, (22, s1*s2, 1280), (1280*s1*s2, 1280, 1), 0); del buf32  # reuse
        buf39 = reinterpret_tensor(buf24, (22, s1*s2, 1280), (1280*s1*s2, 1280, 1), 0); del buf24  # reuse
        # Source Nodes: [attn_output, hidden_states_10, mul_5, residual_2, result_13], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm]
        triton_red_fused_add_div_mul_native_layer_norm_6_xnumel = 22*s1*s2
        triton_red_fused_add_div_mul_native_layer_norm_6.run(buf35, arg18_1, buf34, buf4, arg3_1, buf6, arg21_1, arg22_1, buf39, ps0, s1, s2, triton_red_fused_add_div_mul_native_layer_norm_6_xnumel, 1280, grid=grid(triton_red_fused_add_div_mul_native_layer_norm_6_xnumel), stream=stream0)
        del arg18_1
        del arg21_1
        del arg22_1
        del arg3_1
        buf40 = reinterpret_tensor(buf6, (22*s1*s2, 1280), (1280, 1), 0); del buf6  # reuse
        # Source Nodes: [result_15], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf39, (22*s1*s2, 1280), (1280, 1), 0), reinterpret_tensor(arg23_1, (1280, 1280), (1, 1280), 0), out=buf40)
        del arg23_1
        buf41 = buf33; del buf33  # reuse
        # Source Nodes: [l__self___transformer_blocks_0_attn2_to_q_lora_a_default_0], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf39, (22*s1*s2, 1280), (1280, 1), 0), reinterpret_tensor(arg24_1, (1280, 64), (1, 1280), 0), out=buf41)
        del arg24_1
        buf42 = reinterpret_tensor(buf39, (22*s1*s2, 1280), (1280, 1), 0); del buf39  # reuse
        # Source Nodes: [l__self___transformer_blocks_0_attn2_to_q_lora_b_default_0], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf41, (22*s1*s2, 64), (64, 1), 0), reinterpret_tensor(arg25_1, (64, 1280), (1, 64), 0), out=buf42)
        del arg25_1
        buf43 = empty_strided_cuda((1694, 1280), (1280, 1), torch.float16)
        # Source Nodes: [result_18], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(arg53_1, (1694, 768), (768, 1), 0), reinterpret_tensor(arg26_1, (768, 1280), (1, 768), 0), out=buf43)
        del arg26_1
        buf44 = empty_strided_cuda((1694, 64), (64, 1), torch.float16)
        # Source Nodes: [l__self___transformer_blocks_0_attn2_to_k_lora_a_default_0], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(arg53_1, (1694, 768), (768, 1), 0), reinterpret_tensor(arg27_1, (768, 64), (1, 768), 0), out=buf44)
        del arg27_1
        buf45 = empty_strided_cuda((1694, 1280), (1280, 1), torch.float16)
        # Source Nodes: [l__self___transformer_blocks_0_attn2_to_k_lora_b_default_0], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf44, (1694, 64), (64, 1), 0), reinterpret_tensor(arg28_1, (64, 1280), (1, 64), 0), out=buf45)
        del arg28_1
        buf46 = empty_strided_cuda((1694, 1280), (1280, 1), torch.float16)
        # Source Nodes: [result_21], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(arg53_1, (1694, 768), (768, 1), 0), reinterpret_tensor(arg29_1, (768, 1280), (1, 768), 0), out=buf46)
        del arg29_1
        buf47 = buf44; del buf44  # reuse
        # Source Nodes: [l__self___transformer_blocks_0_attn2_to_v_lora_a_default_0], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(arg53_1, (1694, 768), (768, 1), 0), reinterpret_tensor(arg30_1, (768, 64), (1, 768), 0), out=buf47)
        del arg30_1
        del arg53_1
        buf48 = empty_strided_cuda((1694, 1280), (1280, 1), torch.float16)
        # Source Nodes: [l__self___transformer_blocks_0_attn2_to_v_lora_b_default_0], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf47, (1694, 64), (64, 1), 0), reinterpret_tensor(arg31_1, (64, 1280), (1, 64), 0), out=buf48)
        del arg31_1
        del buf47
        buf49 = reinterpret_tensor(buf40, (22, 8, s1*s2, 160), (1280*s1*s2, 160, 1280, 1), 0); del buf40  # reuse
        # Source Nodes: [hidden_states_11], Original ATen: [aten._scaled_dot_product_flash_attention]
        triton_poi_fused__scaled_dot_product_flash_attention_5_xnumel = 28160*s1*s2
        triton_poi_fused__scaled_dot_product_flash_attention_5.run(buf49, buf42, triton_poi_fused__scaled_dot_product_flash_attention_5_xnumel, grid=grid(triton_poi_fused__scaled_dot_product_flash_attention_5_xnumel), stream=stream0)
        buf50 = reinterpret_tensor(buf43, (22, 8, 77, 160), (98560, 160, 1280, 1), 0); del buf43  # reuse
        # Source Nodes: [hidden_states_11], Original ATen: [aten._scaled_dot_product_flash_attention]
        triton_poi_fused__scaled_dot_product_flash_attention_7.run(buf50, buf45, 2168320, grid=grid(2168320), stream=stream0)
        del buf45
        buf51 = reinterpret_tensor(buf46, (22, 8, 77, 160), (98560, 160, 1280, 1), 0); del buf46  # reuse
        # Source Nodes: [hidden_states_11], Original ATen: [aten._scaled_dot_product_flash_attention]
        triton_poi_fused__scaled_dot_product_flash_attention_7.run(buf51, buf48, 2168320, grid=grid(2168320), stream=stream0)
        del buf48
        # Source Nodes: [hidden_states_11], Original ATen: [aten._scaled_dot_product_flash_attention]
        buf52 = aten._scaled_dot_product_flash_attention.default(buf49, buf50, buf51, scale=0.07905694150420949)
        del buf50
        del buf51
        buf53 = buf52[0]
        del buf52
        buf58 = reinterpret_tensor(buf49, (22*s1*s2, 1280), (1280, 1), 0); del buf49  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf53, (22*s1*s2, 1280), (1280, 1), 0), reinterpret_tensor(arg32_1, (1280, 1280), (1, 1280), 0), out=buf58)
        del arg32_1
        buf59 = buf41; del buf41  # reuse
        # Source Nodes: [l__self___transformer_blocks_0_attn2_to_out_0_lora_a_default_0], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf53, (22*s1*s2, 1280), (1280, 1), 0), reinterpret_tensor(arg34_1, (1280, 64), (1, 1280), 0), out=buf59)
        del arg34_1
        buf60 = reinterpret_tensor(buf53, (22*s1*s2, 1280), (1280, 1), 0); del buf53  # reuse
        # Source Nodes: [l__self___transformer_blocks_0_attn2_to_out_0_lora_b_default_0], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf59, (22*s1*s2, 64), (64, 1), 0), reinterpret_tensor(arg35_1, (64, 1280), (1, 64), 0), out=buf60)
        del arg35_1
        buf64 = reinterpret_tensor(buf42, (22, s1*s2, 1280), (1280*s1*s2, 1280, 1), 0); del buf42  # reuse
        # Source Nodes: [attn_output_1, hidden_states_17, mul_9, norm_hidden_states_2, result_25], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm]
        triton_red_fused_add_div_mul_native_layer_norm_8_xnumel = 22*s1*s2
        triton_red_fused_add_div_mul_native_layer_norm_8.run(buf58, arg33_1, buf60, buf35, arg36_1, arg37_1, buf64, triton_red_fused_add_div_mul_native_layer_norm_8_xnumel, 1280, grid=grid(triton_red_fused_add_div_mul_native_layer_norm_8_xnumel), stream=stream0)
        del arg36_1
        del arg37_1
        buf65 = empty_strided_cuda((22*s1*s2, 10240), (10240, 1), torch.float16)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf64, (22*s1*s2, 1280), (1280, 1), 0), reinterpret_tensor(arg38_1, (1280, 10240), (1, 1280), 0), out=buf65)
        del arg38_1
        buf66 = buf59; del buf59  # reuse
        # Source Nodes: [l__self___transformer_blocks_0_ff_net_0_proj_lora_a_default_0], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf64, (22*s1*s2, 1280), (1280, 1), 0), reinterpret_tensor(arg40_1, (1280, 64), (1, 1280), 0), out=buf66)
        del arg40_1
        buf67 = empty_strided_cuda((22*s1*s2, 10240), (10240, 1), torch.float16)
        # Source Nodes: [l__self___transformer_blocks_0_ff_net_0_proj_lora_b_default_0], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf66, (22*s1*s2, 64), (64, 1), 0), reinterpret_tensor(arg41_1, (64, 10240), (1, 64), 0), out=buf67)
        del arg41_1
        buf68 = empty_strided_cuda((22, s1*s2, 5120), (5120*s1*s2, 5120, 1), torch.float16)
        # Source Nodes: [gelu, hidden_states_20], Original ATen: [aten.gelu, aten.mul]
        triton_poi_fused_gelu_mul_9_xnumel = 112640*s1*s2
        triton_poi_fused_gelu_mul_9.run(buf65, arg39_1, buf67, buf68, triton_poi_fused_gelu_mul_9_xnumel, grid=grid(triton_poi_fused_gelu_mul_9_xnumel), stream=stream0)
        del arg39_1
        del buf65
        del buf67
        buf69 = reinterpret_tensor(buf64, (22*s1*s2, 1280), (1280, 1), 0); del buf64  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf68, (22*s1*s2, 5120), (5120, 1), 0), reinterpret_tensor(arg42_1, (5120, 1280), (1, 5120), 0), out=buf69)
        del arg42_1
        buf70 = buf66; del buf66  # reuse
        # Source Nodes: [l__self___transformer_blocks_0_ff_net_2_lora_a_default_0], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf68, (22*s1*s2, 5120), (5120, 1), 0), reinterpret_tensor(arg44_1, (5120, 64), (1, 5120), 0), out=buf70)
        del arg44_1
        del buf68
        buf71 = reinterpret_tensor(buf4, (22*s1*s2, 1280), (1280, 1), 0); del buf4  # reuse
        # Source Nodes: [l__self___transformer_blocks_0_ff_net_2_lora_b_default_0], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf70, (22*s1*s2, 64), (64, 1), 0), reinterpret_tensor(arg45_1, (64, 1280), (1, 64), 0), out=buf71)
        del arg45_1
        del buf70
        buf72 = reinterpret_tensor(buf34, (22, 1280, s1, s2), (1280*s1*s2, s1*s2, s2, 1), 0); del buf34  # reuse
        # Source Nodes: [hidden_states_25], Original ATen: [aten.clone]
        triton_poi_fused_clone_10_ynumel = 22*s1*s2
        triton_poi_fused_clone_10.run(buf69, arg43_1, buf71, buf58, arg33_1, buf60, buf35, buf72, ps0, s1, s2, triton_poi_fused_clone_10_ynumel, 1280, grid=grid(triton_poi_fused_clone_10_ynumel, 1280), stream=stream0)
        del arg33_1
        del arg43_1
        del buf35
        del buf58
        del buf60
        del buf69
        del buf71
        # Source Nodes: [result_33], Original ATen: [aten.convolution]
        buf73 = extern_kernels.convolution(buf72, arg46_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf73, (22, 1280, s1, s2), (1280*s1*s2, s1*s2, s2, 1))
        del arg46_1
        # Source Nodes: [l__self___proj_out_lora_a_default_0], Original ATen: [aten.convolution]
        buf74 = extern_kernels.convolution(buf72, arg48_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf74, (22, 64, s1, s2), (64*s1*s2, s1*s2, s2, 1))
        del arg48_1
        del buf72
        # Source Nodes: [l__self___proj_out_lora_b_default_0], Original ATen: [aten.convolution]
        buf75 = extern_kernels.convolution(buf74, arg49_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf75, (22, 1280, s1, s2), (1280*s1*s2, s1*s2, s2, 1))
        del arg49_1
        del buf74
        buf76 = buf73; del buf73  # reuse
        # Source Nodes: [mul_13, output, result_33, result_34], Original ATen: [aten.add, aten.convolution, aten.mul]
        triton_poi_fused_add_convolution_mul_11_xnumel = 28160*s1*s2
        triton_poi_fused_add_convolution_mul_11.run(buf76, arg47_1, buf75, arg52_1, ps0, triton_poi_fused_add_convolution_mul_11_xnumel, grid=grid(triton_poi_fused_add_convolution_mul_11_xnumel), stream=stream0)
        del arg47_1
        del arg52_1
        del buf75
    return (buf76, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((1280, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg1_1 = rand_strided((1280, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg2_1 = rand_strided((1280, 1280, 1, 1), (1280, 1, 1, 1), device='cuda:0', dtype=torch.float16)
    arg3_1 = rand_strided((1280, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg4_1 = rand_strided((64, 1280, 1, 1), (1280, 1, 1, 1), device='cuda:0', dtype=torch.float16)
    arg5_1 = rand_strided((1280, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float16)
    arg6_1 = rand_strided((1280, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg7_1 = rand_strided((1280, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg8_1 = rand_strided((1280, 1280), (1280, 1), device='cuda:0', dtype=torch.float16)
    arg9_1 = rand_strided((64, 1280), (1280, 1), device='cuda:0', dtype=torch.float16)
    arg10_1 = rand_strided((1280, 64), (64, 1), device='cuda:0', dtype=torch.float16)
    arg11_1 = rand_strided((1280, 1280), (1280, 1), device='cuda:0', dtype=torch.float16)
    arg12_1 = rand_strided((64, 1280), (1280, 1), device='cuda:0', dtype=torch.float16)
    arg13_1 = rand_strided((1280, 64), (64, 1), device='cuda:0', dtype=torch.float16)
    arg14_1 = rand_strided((1280, 1280), (1280, 1), device='cuda:0', dtype=torch.float16)
    arg15_1 = rand_strided((64, 1280), (1280, 1), device='cuda:0', dtype=torch.float16)
    arg16_1 = rand_strided((1280, 64), (64, 1), device='cuda:0', dtype=torch.float16)
    arg17_1 = rand_strided((1280, 1280), (1280, 1), device='cuda:0', dtype=torch.float16)
    arg18_1 = rand_strided((1280, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg19_1 = rand_strided((64, 1280), (1280, 1), device='cuda:0', dtype=torch.float16)
    arg20_1 = rand_strided((1280, 64), (64, 1), device='cuda:0', dtype=torch.float16)
    arg21_1 = rand_strided((1280, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg22_1 = rand_strided((1280, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg23_1 = rand_strided((1280, 1280), (1280, 1), device='cuda:0', dtype=torch.float16)
    arg24_1 = rand_strided((64, 1280), (1280, 1), device='cuda:0', dtype=torch.float16)
    arg25_1 = rand_strided((1280, 64), (64, 1), device='cuda:0', dtype=torch.float16)
    arg26_1 = rand_strided((1280, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    arg27_1 = rand_strided((64, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    arg28_1 = rand_strided((1280, 64), (64, 1), device='cuda:0', dtype=torch.float16)
    arg29_1 = rand_strided((1280, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    arg30_1 = rand_strided((64, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    arg31_1 = rand_strided((1280, 64), (64, 1), device='cuda:0', dtype=torch.float16)
    arg32_1 = rand_strided((1280, 1280), (1280, 1), device='cuda:0', dtype=torch.float16)
    arg33_1 = rand_strided((1280, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg34_1 = rand_strided((64, 1280), (1280, 1), device='cuda:0', dtype=torch.float16)
    arg35_1 = rand_strided((1280, 64), (64, 1), device='cuda:0', dtype=torch.float16)
    arg36_1 = rand_strided((1280, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg37_1 = rand_strided((1280, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg38_1 = rand_strided((10240, 1280), (1280, 1), device='cuda:0', dtype=torch.float16)
    arg39_1 = rand_strided((10240, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg40_1 = rand_strided((64, 1280), (1280, 1), device='cuda:0', dtype=torch.float16)
    arg41_1 = rand_strided((10240, 64), (64, 1), device='cuda:0', dtype=torch.float16)
    arg42_1 = rand_strided((1280, 5120), (5120, 1), device='cuda:0', dtype=torch.float16)
    arg43_1 = rand_strided((1280, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg44_1 = rand_strided((64, 5120), (5120, 1), device='cuda:0', dtype=torch.float16)
    arg45_1 = rand_strided((1280, 64), (64, 1), device='cuda:0', dtype=torch.float16)
    arg46_1 = rand_strided((1280, 1280, 1, 1), (1280, 1, 1, 1), device='cuda:0', dtype=torch.float16)
    arg47_1 = rand_strided((1280, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg48_1 = rand_strided((64, 1280, 1, 1), (1280, 1, 1, 1), device='cuda:0', dtype=torch.float16)
    arg49_1 = rand_strided((1280, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float16)
    arg50_1 = 4
    arg51_1 = 12
    arg52_1 = rand_strided((22, 1280, 4, 12), (61440, 48, 12, 1), device='cuda:0', dtype=torch.float16)
    arg53_1 = rand_strided((22, 77, 768), (59136, 768, 1), device='cuda:0', dtype=torch.float16)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
