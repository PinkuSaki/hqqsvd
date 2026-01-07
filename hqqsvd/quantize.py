from typing import Tuple
from functools import lru_cache
import importlib.util
import torch
from .optimize import optimize_weights
from .bitpack import pack, unpack

_HAS_TRITON = importlib.util.find_spec("triton") is not None
if _HAS_TRITON:
    try:
        import triton
        import triton.language as tl
    except Exception:
        triton = None
        tl = None
        _HAS_TRITON = False
else:
    triton = None
    tl = None

if _HAS_TRITON:

    @triton.jit
    def _dequant_uint4_kernel(
        packed_ptr,
        scale_ptr,
        zero_ptr,
        output_ptr,
        n_elements,
        n_groups,
        group_size,
        BLOCK: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offsets = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < n_elements

        packed_index = offsets // 2
        packed = tl.load(packed_ptr + packed_index, mask=mask, other=0)
        low = packed & 0x0F
        high = packed >> 4
        is_high = offsets & 1
        values = tl.where(is_high == 1, high, low).to(tl.float32)

        group_stride = n_groups * group_size
        out_feature = offsets // group_stride
        group = (offsets % group_stride) // group_size
        scale_index = out_feature * n_groups + group

        scale = tl.load(scale_ptr + scale_index, mask=mask, other=1.0)
        zero = tl.load(zero_ptr + scale_index, mask=mask, other=0.0)
        output = zero + values * scale
        tl.store(output_ptr + offsets, output, mask=mask)

else:
    _dequant_uint4_kernel = None


@lru_cache(maxsize=1)
def _get_dequant_uint4_kernel():
    if _dequant_uint4_kernel is None:
        raise RuntimeError("Triton is not available for uint4 dequantization.")
    return _dequant_uint4_kernel


def _dequantize_uint4_triton(
    packed_tensor: torch.ByteTensor,
    scale: torch.Tensor,
    zero: torch.Tensor,
    q_shape: torch.Size,
) -> torch.Tensor:
    import triton

    packed_tensor = packed_tensor.contiguous().view(-1)
    n_elements = int(q_shape.numel())
    output = torch.empty(n_elements, device=packed_tensor.device, dtype=scale.dtype)
    scale_flat = scale.contiguous().view(-1)
    zero_flat = zero.contiguous().view(-1)
    kernel = _get_dequant_uint4_kernel()
    grid = (triton.cdiv(n_elements, 1024),)
    kernel[grid](
        packed_tensor,
        scale_flat,
        zero_flat,
        output,
        n_elements,
        scale.shape[1],
        q_shape[-1],
        BLOCK=1024,
    )
    return output.view(q_shape)


def apply_svdquant(
    weight: torch.FloatTensor, rank: int = 32, niter: int = 8
) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
    reshape_weight = False
    if weight.ndim > 2:  # convs
        reshape_weight = True
        weight_shape = weight.shape
        weight = weight.flatten(1, -1)
    weight = weight.to(dtype=torch.float32)
    U, S, svd_down = torch.svd_lowrank(weight, q=rank, niter=niter)
    svd_up = torch.mul(U, S.unsqueeze(0))
    svd_down = svd_down.t_()
    weight = weight.sub(torch.mm(svd_up, svd_down))
    if reshape_weight:
        weight = weight.unflatten(
            -1, (*weight_shape[1:],)
        )  # pylint: disable=possibly-used-before-assignment
    return weight, svd_up, svd_down


def quantize(
    W,
    svd_rank: int = 128,
    svd_steps: int = 8,
    group_size: int = 128,
    nbits: int = 4,
    fast=True,
):
    dtype = W.dtype
    shape = W.shape

    W, svd_up, svd_down = apply_svdquant(W, rank=svd_rank, niter=svd_steps)

    W = W.reshape([-1, group_size])

    _min = W.min(axis=1, keepdim=True)[0]
    _max = W.max(axis=1, keepdim=True)[0]
    max_v = round(2**nbits - 1)
    min_v = 0
    min_max = [min_v, max_v]

    # Note: here we work with the inverse of the scale to avoid division and quantize instead via W*scale + zero, the scale is inverted later on.
    denom = _max - _min
    scale = max_v / denom
    scale = torch.where(
        denom.abs() <= 1e-4, torch.full_like(scale, 1.0), scale
    )  # Avoid small denom values
    scale = scale.clamp(max=2e4)  # clamp to avoid half-precision problems
    zero = -_min * scale

    if fast:
        W_q = (W * scale + zero).round_().clamp_(min_max[0], min_max[1])
    else:
        W_q, scale, zero = optimize_weights(W, scale, zero, min_max, 1)

    W_q = W_q.reshape((shape[0], -1, group_size))
    W_q = torch.clamp(W_q, min_v, max_v).to(torch.uint8)
    W_q = pack(W_q, nbits)
    scale = 1.0 / scale.reshape((shape[0], -1, 1))
    zero = -zero.reshape((shape[0], -1, 1)) * scale
    return W_q, svd_up.to(dtype), svd_down.to(dtype), scale.to(dtype), zero.to(dtype)


def dequantize(
    W_q,
    svd_up,
    svd_down,
    scale,
    zero,
    q_shape,
    o_shape,
    nbits: int,
    use_fused_kernel: bool = True,
):
    use_triton = (
        use_fused_kernel
        and nbits == 4
        and W_q.is_cuda
        and scale.is_cuda
        and zero.is_cuda
        and _HAS_TRITON
    )
    if use_triton:
        W_f = _dequantize_uint4_triton(W_q, scale, zero, q_shape)
    else:
        W_f = unpack(W_q, q_shape, nbits).to(dtype=scale.dtype)
        W_f = torch.addcmul(zero, W_f, scale)
    W_f = W_f.view(o_shape)
    W_f.addmm_(svd_up, svd_down)
    return W_f
