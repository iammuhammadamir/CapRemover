
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.pointwise(
    size_hints=[512], 
    filename=__file__,
    triton_meta={'signature': {0: '*i64', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_0', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': 'a1fa794fabf29fbd973066d9ab65d9c16aea0ed2e098514ffda941ec2a0f87ad'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 320
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp5 = tl.load(in_ptr0 + (0))
    tmp6 = tl.broadcast_to(tmp5, [XBLOCK])
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 160, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp7 = tmp6.to(tl.float32)
    tmp8 = tmp0.to(tl.float32)
    tmp9 = 1.0
    tmp10 = tmp8 * tmp9
    tmp11 = 0.0
    tmp12 = tmp10 + tmp11
    tmp13 = -9.210340371976184
    tmp14 = tmp12 * tmp13
    tmp15 = 160.0
    tmp16 = tmp14 / tmp15
    tmp17 = tl_math.exp(tmp16)
    tmp18 = tmp7 * tmp17
    tmp19 = tmp18 * tmp9
    tmp20 = tl_math.sin(tmp19)
    tmp21 = tl.full(tmp20.shape, 0.0, tmp20.dtype)
    tmp22 = tl.where(tmp4, tmp20, tmp21)
    tmp23 = tmp0 >= tmp3
    tmp24 = tl.full([1], 320, tl.int64)
    tmp25 = tmp0 < tmp24
    tmp26 = (-160) + x0
    tmp27 = tmp26.to(tl.float32)
    tmp28 = tmp27 * tmp9
    tmp29 = tmp28 + tmp11
    tmp30 = tmp29 * tmp13
    tmp31 = tmp30 / tmp15
    tmp32 = tl_math.exp(tmp31)
    tmp33 = tmp7 * tmp32
    tmp34 = tmp33 * tmp9
    tmp35 = tl_math.cos(tmp34)
    tmp36 = tl.full(tmp35.shape, 0.0, tmp35.dtype)
    tmp37 = tl.where(tmp23, tmp35, tmp36)
    tmp38 = tl.where(tmp4, tmp22, tmp37)
    tl.store(out_ptr0 + (x0), tmp38, xmask)
