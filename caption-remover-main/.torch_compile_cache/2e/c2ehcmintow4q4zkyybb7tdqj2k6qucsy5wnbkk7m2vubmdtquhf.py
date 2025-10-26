
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
