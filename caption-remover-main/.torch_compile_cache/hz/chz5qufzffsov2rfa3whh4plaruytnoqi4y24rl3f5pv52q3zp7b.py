
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


# kernel path: /workspace/caption-remover-main/src/stages/inpaint/../../../.torch_compile_cache/bl/cblpdadv7pbfw2grld5ck2igdgqqlqqswsnfc7g2hk6mobgptk7x.py
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
    size_hints=[1024, 32768],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_group_norm_0', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': 'a1fa794fabf29fbd973066d9ab65d9c16aea0ed2e098514ffda941ec2a0f87ad'}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 704
    rnumel = 27000
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
        tmp0 = tl.load(in_ptr0 + (r1 + (27000*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
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


# kernel path: /workspace/caption-remover-main/src/stages/inpaint/../../../.torch_compile_cache/qb/cqb32uiw36vwnyvtgujzwoc6yznuefj5hj5slb6u5za62hz3c3bq.py
# Source Nodes: [hidden_states_1, hidden_states_2], Original ATen: [aten.native_group_norm, aten.silu]
# hidden_states_1 => add_1, mul_1
# hidden_states_2 => convert_element_type_5, mul_2, sigmoid
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
    size_hints=[8192, 4096], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*fp16', 4: '*fp16', 5: '*fp16', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_group_norm_silu_1', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': 'a1fa794fabf29fbd973066d9ab65d9c16aea0ed2e098514ffda941ec2a0f87ad'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    tmp2 = tl.load(in_ptr1 + ((32*y1) + (y0 // 10)), ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + ((32*y1) + (y0 // 10)), ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last').to(tl.float32)
    tmp14 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 - tmp2
    tmp5 = 27000.0
    tmp6 = tmp4 / tmp5
    tmp7 = 1e-05
    tmp8 = tmp6 + tmp7
    tmp9 = libdevice.rsqrt(tmp8)
    tmp10 = tmp3 * tmp9
    tmp12 = tmp11.to(tl.float32)
    tmp13 = tmp10 * tmp12
    tmp15 = tmp14.to(tl.float32)
    tmp16 = tmp13 + tmp15
    tmp17 = tl.sigmoid(tmp16)
    tmp18 = tmp16 * tmp17
    tmp19 = tmp18.to(tl.float32)
    tl.store(out_ptr1 + (y0 + (320*x2) + (864000*y1)), tmp19, xmask & ymask)
''', device_str='cuda')


# kernel path: /workspace/caption-remover-main/src/stages/inpaint/../../../.torch_compile_cache/sa/csayshawgkpasruosg7ggmm5uw53juiqda4r3jp4vgzgxqaukoz6.py
# Source Nodes: [hidden_states_2, result], Original ATen: [aten.convolution, aten.silu]
# hidden_states_2 => convert_element_type_5, mul_2, sigmoid
# result => convolution
triton_poi_fused_convolution_silu_2 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_silu_2', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': 'a1fa794fabf29fbd973066d9ab65d9c16aea0ed2e098514ffda941ec2a0f87ad'},
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


# kernel path: /workspace/caption-remover-main/src/stages/inpaint/../../../.torch_compile_cache/se/cse7kba7noqxyyd4mtzoh5yib5tpg6bmwnegjutcpjdjgbhtsl3v.py
# Source Nodes: [l__self___conv1_lora_a_default_0], Original ATen: [aten.convolution]
# l__self___conv1_lora_a_default_0 => convolution_1
triton_poi_fused_convolution_3 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_3', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': 'a1fa794fabf29fbd973066d9ab65d9c16aea0ed2e098514ffda941ec2a0f87ad'},
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


# kernel path: /workspace/caption-remover-main/src/stages/inpaint/../../../.torch_compile_cache/rs/crscx4tyqoy3tqi4z7zk5ktv6gihtkkkznt6kg2j3pyy3rhc2civ.py
# Source Nodes: [temb], Original ATen: [aten.silu]
# temb => convert_element_type_6, convert_element_type_7, mul_4, sigmoid_1
triton_poi_fused_silu_4 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_silu_4', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': 'a1fa794fabf29fbd973066d9ab65d9c16aea0ed2e098514ffda941ec2a0f87ad'},
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


# kernel path: /workspace/caption-remover-main/src/stages/inpaint/../../../.torch_compile_cache/ho/cholmkukuuz5awjtg5ygcfhprv5g62vgdghgmucvmms5cr2whda3.py
# Source Nodes: [hidden_states_2, hidden_states_4, mul, result, result_1], Original ATen: [aten.add, aten.convolution, aten.mul, aten.silu]
# hidden_states_2 => convert_element_type_5, mul_2, sigmoid
# hidden_states_4 => add_4
# mul => mul_3
# result => convolution
# result_1 => add_2
triton_poi_fused_add_convolution_mul_silu_5 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: '*fp16', 5: '*fp16', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_mul_silu_5', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'backend_hash': 'a1fa794fabf29fbd973066d9ab65d9c16aea0ed2e098514ffda941ec2a0f87ad'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 19008000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 320
    x2 = (xindex // 864000)
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask).to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp3 = tl.load(in_ptr1 + (x3), xmask).to(tl.float32)
    tmp7 = tl.load(in_ptr2 + (x0 + (320*x2)), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp8 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp10 = tl.load(in_ptr4 + (x0 + (320*x2)), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp4 = 0.125
    tmp5 = tmp3 * tmp4
    tmp6 = tmp2 + tmp5
    tmp9 = tmp7 + tmp8
    tmp11 = tmp10 * tmp4
    tmp12 = tmp9 + tmp11
    tmp13 = tmp6 + tmp12
    tl.store(in_out_ptr0 + (x3), tmp13, xmask)
''', device_str='cuda')


# kernel path: /workspace/caption-remover-main/src/stages/inpaint/../../../.torch_compile_cache/jn/cjnurs6kp75pzncq6t6h2r6yy6rzotarjsjur4xxhbyc4wrjm7qo.py
# Source Nodes: [hidden_states_5], Original ATen: [aten.native_group_norm]
# hidden_states_5 => convert_element_type_15, var_mean_1
triton_red_fused_native_group_norm_6 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.reduction(
    size_hints=[1024, 32768],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_group_norm_6', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': 'a1fa794fabf29fbd973066d9ab65d9c16aea0ed2e098514ffda941ec2a0f87ad'}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 704
    rnumel = 27000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 32
    x1 = (xindex // 32)
    tmp3_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp3_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp3_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x4 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex % 10
        r3 = (rindex // 10)
        tmp0 = tl.load(in_ptr0 + (r2 + (10*x0) + (320*r3) + (864000*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
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
    tl.store(out_ptr0 + (x4), tmp3, xmask)
    tl.store(out_ptr1 + (x4), tmp4, xmask)
''', device_str='cuda')


# kernel path: /workspace/caption-remover-main/src/stages/inpaint/../../../.torch_compile_cache/hd/chd4sseflmha3lnf53u2npw5gmvrxwe7bsxfkxczvjc7bfltsgyd.py
# Source Nodes: [hidden_states_5, hidden_states_6], Original ATen: [aten.native_group_norm, aten.silu]
# hidden_states_5 => add_6, mul_7
# hidden_states_6 => convert_element_type_20, mul_8, sigmoid_2
triton_poi_fused_native_group_norm_silu_7 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*fp16', 4: '*fp16', 5: '*fp16', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_group_norm_silu_7', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': 'a1fa794fabf29fbd973066d9ab65d9c16aea0ed2e098514ffda941ec2a0f87ad'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 19008000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 320
    x2 = (xindex // 864000)
    tmp0 = tl.load(in_ptr0 + (x3), xmask).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + ((32*x2) + (x0 // 10)), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + ((32*x2) + (x0 // 10)), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp14 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 - tmp2
    tmp5 = 27000.0
    tmp6 = tmp4 / tmp5
    tmp7 = 1e-05
    tmp8 = tmp6 + tmp7
    tmp9 = libdevice.rsqrt(tmp8)
    tmp10 = tmp3 * tmp9
    tmp12 = tmp11.to(tl.float32)
    tmp13 = tmp10 * tmp12
    tmp15 = tmp14.to(tl.float32)
    tmp16 = tmp13 + tmp15
    tmp17 = tl.sigmoid(tmp16)
    tmp18 = tmp16 * tmp17
    tmp19 = tmp18.to(tl.float32)
    tl.store(out_ptr1 + (x3), tmp19, xmask)
''', device_str='cuda')


# kernel path: /workspace/caption-remover-main/src/stages/inpaint/../../../.torch_compile_cache/vk/cvk77foil2cmgtxf3tky32mtvmynvcpxcyxs7fk4towlrhqtdrni.py
# Source Nodes: [add_4, hidden_states_6, mul_2, output_tensor, result_6, result_7], Original ATen: [aten.add, aten.convolution, aten.div, aten.mul, aten.silu]
# add_4 => add_8
# hidden_states_6 => convert_element_type_20, mul_8, sigmoid_2
# mul_2 => mul_9
# output_tensor => div
# result_6 => convolution_3
# result_7 => add_7
triton_poi_fused_add_convolution_div_mul_silu_8 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: '*fp16', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_div_mul_silu_8', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': 'a1fa794fabf29fbd973066d9ab65d9c16aea0ed2e098514ffda941ec2a0f87ad'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    tmp1 = tl.load(in_ptr1 + (y0 + (320*x2) + (864000*y1)), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last').to(tl.float32)
    tmp4 = tl.load(in_ptr3 + (y0 + (320*x2) + (864000*y1)), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp3 = tmp1 + tmp2
    tmp5 = 0.125
    tmp6 = tmp4 * tmp5
    tmp7 = tmp3 + tmp6
    tmp8 = tmp0 + tmp7
    tmp9 = 1.0
    tmp10 = tmp8 / tmp9
    tl.store(out_ptr0 + (x2 + (2700*y3)), tmp10, xmask & ymask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1 = args
    args.clear()
    assert_size_stride(arg0_1, (320, ), (1, ))
    assert_size_stride(arg1_1, (320, ), (1, ))
    assert_size_stride(arg2_1, (320, 320, 3, 3), (2880, 9, 3, 1))
    assert_size_stride(arg3_1, (320, ), (1, ))
    assert_size_stride(arg4_1, (64, 320, 3, 3), (2880, 9, 3, 1))
    assert_size_stride(arg5_1, (320, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(arg6_1, (320, 1280), (1280, 1))
    assert_size_stride(arg7_1, (320, ), (1, ))
    assert_size_stride(arg8_1, (64, 1280), (1280, 1))
    assert_size_stride(arg9_1, (320, 64), (64, 1))
    assert_size_stride(arg10_1, (320, ), (1, ))
    assert_size_stride(arg11_1, (320, ), (1, ))
    assert_size_stride(arg12_1, (320, 320, 3, 3), (2880, 9, 3, 1))
    assert_size_stride(arg13_1, (320, ), (1, ))
    assert_size_stride(arg14_1, (64, 320, 3, 3), (2880, 9, 3, 1))
    assert_size_stride(arg15_1, (320, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(arg16_1, (22, 320, 30, 90), (864000, 2700, 90, 1))
    assert_size_stride(arg17_1, (22, 1280), (1280, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((22, 32, 1, 1), (32, 1, 704, 704), torch.float32)
        buf1 = empty_strided_cuda((22, 32, 1, 1), (32, 1, 704, 704), torch.float32)
        # Source Nodes: [hidden_states_1], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_native_group_norm_0.run(arg16_1, buf0, buf1, 704, 27000, grid=grid(704), stream=stream0)
        buf4 = empty_strided_cuda((22, 320, 30, 90), (864000, 1, 28800, 320), torch.float16)
        # Source Nodes: [hidden_states_1, hidden_states_2], Original ATen: [aten.native_group_norm, aten.silu]
        triton_poi_fused_native_group_norm_silu_1.run(arg16_1, buf0, buf1, arg0_1, arg1_1, buf4, 7040, 2700, grid=grid(7040, 2700), stream=stream0)
        del arg0_1
        del arg1_1
        buf5 = empty_strided_cuda((320, 320, 3, 3), (2880, 1, 960, 320), torch.float16)
        # Source Nodes: [hidden_states_2, result], Original ATen: [aten.convolution, aten.silu]
        triton_poi_fused_convolution_silu_2.run(arg2_1, buf5, 102400, 9, grid=grid(102400, 9), stream=stream0)
        del arg2_1
        # Source Nodes: [hidden_states_2, result], Original ATen: [aten.convolution, aten.silu]
        buf6 = extern_kernels.convolution(buf4, buf5, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf6, (22, 320, 30, 90), (864000, 1, 28800, 320))
        buf7 = empty_strided_cuda((64, 320, 3, 3), (2880, 1, 960, 320), torch.float16)
        # Source Nodes: [l__self___conv1_lora_a_default_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_3.run(arg4_1, buf7, 20480, 9, grid=grid(20480, 9), stream=stream0)
        del arg4_1
        # Source Nodes: [l__self___conv1_lora_a_default_0], Original ATen: [aten.convolution]
        buf8 = extern_kernels.convolution(buf4, buf7, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf8, (22, 64, 30, 90), (172800, 1, 5760, 64))
        del buf4
        # Source Nodes: [l__self___conv1_lora_b_default_0], Original ATen: [aten.convolution]
        buf9 = extern_kernels.convolution(buf8, arg5_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf9, (22, 320, 30, 90), (864000, 1, 28800, 320))
        del arg5_1
        del buf8
        buf10 = empty_strided_cuda((22, 1280), (1280, 1), torch.float16)
        # Source Nodes: [temb], Original ATen: [aten.silu]
        triton_poi_fused_silu_4.run(arg17_1, buf10, 28160, grid=grid(28160), stream=stream0)
        del arg17_1
        buf11 = empty_strided_cuda((22, 320), (320, 1), torch.float16)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf10, reinterpret_tensor(arg6_1, (1280, 320), (1, 1280), 0), out=buf11)
        del arg6_1
        buf12 = empty_strided_cuda((22, 64), (64, 1), torch.float16)
        # Source Nodes: [l__self___time_emb_proj_lora_a_default_0], Original ATen: [aten.mm]
        extern_kernels.mm(buf10, reinterpret_tensor(arg8_1, (1280, 64), (1, 1280), 0), out=buf12)
        del arg8_1
        del buf10
        buf13 = empty_strided_cuda((22, 320), (320, 1), torch.float16)
        # Source Nodes: [l__self___time_emb_proj_lora_b_default_0], Original ATen: [aten.mm]
        extern_kernels.mm(buf12, reinterpret_tensor(arg9_1, (64, 320), (1, 64), 0), out=buf13)
        del arg9_1
        del buf12
        buf14 = buf6; del buf6  # reuse
        # Source Nodes: [hidden_states_2, hidden_states_4, mul, result, result_1], Original ATen: [aten.add, aten.convolution, aten.mul, aten.silu]
        triton_poi_fused_add_convolution_mul_silu_5.run(buf14, arg3_1, buf9, buf11, arg7_1, buf13, 19008000, grid=grid(19008000), stream=stream0)
        del arg3_1
        del arg7_1
        del buf11
        del buf13
        buf15 = buf1; del buf1  # reuse
        buf16 = buf0; del buf0  # reuse
        # Source Nodes: [hidden_states_5], Original ATen: [aten.native_group_norm]
        triton_red_fused_native_group_norm_6.run(buf14, buf15, buf16, 704, 27000, grid=grid(704), stream=stream0)
        buf19 = buf9; del buf9  # reuse
        # Source Nodes: [hidden_states_5, hidden_states_6], Original ATen: [aten.native_group_norm, aten.silu]
        triton_poi_fused_native_group_norm_silu_7.run(buf14, buf15, buf16, arg10_1, arg11_1, buf19, 19008000, grid=grid(19008000), stream=stream0)
        del arg10_1
        del arg11_1
        del buf14
        del buf15
        del buf16
        buf20 = buf5; del buf5  # reuse
        # Source Nodes: [hidden_states_6, result_6], Original ATen: [aten.convolution, aten.silu]
        triton_poi_fused_convolution_silu_2.run(arg12_1, buf20, 102400, 9, grid=grid(102400, 9), stream=stream0)
        del arg12_1
        # Source Nodes: [hidden_states_6, result_6], Original ATen: [aten.convolution, aten.silu]
        buf21 = extern_kernels.convolution(buf19, buf20, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf21, (22, 320, 30, 90), (864000, 1, 28800, 320))
        del buf20
        buf22 = buf7; del buf7  # reuse
        # Source Nodes: [l__self___conv2_lora_a_default_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_3.run(arg14_1, buf22, 20480, 9, grid=grid(20480, 9), stream=stream0)
        del arg14_1
        # Source Nodes: [l__self___conv2_lora_a_default_0], Original ATen: [aten.convolution]
        buf23 = extern_kernels.convolution(buf19, buf22, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf23, (22, 64, 30, 90), (172800, 1, 5760, 64))
        del buf22
        # Source Nodes: [l__self___conv2_lora_b_default_0], Original ATen: [aten.convolution]
        buf24 = extern_kernels.convolution(buf23, arg15_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf24, (22, 320, 30, 90), (864000, 1, 28800, 320))
        del arg15_1
        del buf23
        buf25 = reinterpret_tensor(buf19, (22, 320, 30, 90), (864000, 2700, 90, 1), 0); del buf19  # reuse
        # Source Nodes: [add_4, hidden_states_6, mul_2, output_tensor, result_6, result_7], Original ATen: [aten.add, aten.convolution, aten.div, aten.mul, aten.silu]
        triton_poi_fused_add_convolution_div_mul_silu_8.run(arg16_1, buf21, arg13_1, buf24, buf25, 7040, 2700, grid=grid(7040, 2700), stream=stream0)
        del arg13_1
        del arg16_1
        del buf21
        del buf24
    return (buf25, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg1_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg2_1 = rand_strided((320, 320, 3, 3), (2880, 9, 3, 1), device='cuda:0', dtype=torch.float16)
    arg3_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg4_1 = rand_strided((64, 320, 3, 3), (2880, 9, 3, 1), device='cuda:0', dtype=torch.float16)
    arg5_1 = rand_strided((320, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float16)
    arg6_1 = rand_strided((320, 1280), (1280, 1), device='cuda:0', dtype=torch.float16)
    arg7_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg8_1 = rand_strided((64, 1280), (1280, 1), device='cuda:0', dtype=torch.float16)
    arg9_1 = rand_strided((320, 64), (64, 1), device='cuda:0', dtype=torch.float16)
    arg10_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg11_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg12_1 = rand_strided((320, 320, 3, 3), (2880, 9, 3, 1), device='cuda:0', dtype=torch.float16)
    arg13_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg14_1 = rand_strided((64, 320, 3, 3), (2880, 9, 3, 1), device='cuda:0', dtype=torch.float16)
    arg15_1 = rand_strided((320, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float16)
    arg16_1 = rand_strided((22, 320, 30, 90), (864000, 2700, 90, 1), device='cuda:0', dtype=torch.float16)
    arg17_1 = rand_strided((22, 1280), (1280, 1), device='cuda:0', dtype=torch.float16)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
