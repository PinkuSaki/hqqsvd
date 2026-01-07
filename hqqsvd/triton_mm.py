import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64}, num_stages=3, num_warps=8
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 256, "BLOCK_K": 32}, num_stages=4, num_warps=4
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32}, num_stages=4, num_warps=4
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32}, num_stages=4, num_warps=4
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32}, num_stages=4, num_warps=4
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 32, "BLOCK_K": 32}, num_stages=4, num_warps=4
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 32, "BLOCK_K": 32}, num_stages=5, num_warps=2
        ),
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 32}, num_stages=5, num_warps=2
        ),
        # Larger block sizes for bigger matrices
        triton.Config(
            {"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 64}, num_stages=3, num_warps=8
        ),
        triton.Config(
            {"BLOCK_M": 256, "BLOCK_N": 64, "BLOCK_K": 64}, num_stages=4, num_warps=4
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 128}, num_stages=3, num_warps=8
        ),
        triton.Config(
            {"BLOCK_M": 256, "BLOCK_N": 256, "BLOCK_K": 64}, num_stages=3, num_warps=8
        ),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def int8_matmul_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Int8xInt8 matrix multiplication kernel.
    Computes C = A @ B where A is (M, K) and B is (K, N).
    Uses int32 accumulation to prevent overflow.
    """
    # Block coordinates
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Offsets for the current block
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Pointers to the start of A and B blocks
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    # Accumulator - use int32 to prevent overflow
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)

    # Loop over K dimension in blocks
    for k in range(0, K, BLOCK_K):
        # Load blocks from A and B
        # Mask out-of-bounds accesses
        k_remaining = K - k
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < k_remaining)
        b_mask = (offs_k[:, None] < k_remaining) & (offs_n[None, :] < N)

        a = tl.load(a_ptrs, mask=a_mask, other=0)
        b = tl.load(b_ptrs, mask=b_mask, other=0)

        # Accumulate - cast to int32 for accumulation
        acc += tl.dot(a, b, out_dtype=tl.int32)

        # Advance pointers
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # Write back result
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    tl.store(c_ptrs, acc, mask=c_mask)


@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64}, num_stages=3, num_warps=8
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 256, "BLOCK_K": 32}, num_stages=4, num_warps=4
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32}, num_stages=4, num_warps=4
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32}, num_stages=4, num_warps=4
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32}, num_stages=4, num_warps=4
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 32, "BLOCK_K": 32}, num_stages=4, num_warps=4
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 32, "BLOCK_K": 32}, num_stages=5, num_warps=2
        ),
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 32}, num_stages=5, num_warps=2
        ),
        triton.Config(
            {"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 64}, num_stages=3, num_warps=8
        ),
        triton.Config(
            {"BLOCK_M": 256, "BLOCK_N": 64, "BLOCK_K": 64}, num_stages=4, num_warps=4
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 128}, num_stages=3, num_warps=8
        ),
        triton.Config(
            {"BLOCK_M": 256, "BLOCK_N": 256, "BLOCK_K": 64}, num_stages=3, num_warps=8
        ),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def scaled_int8_matmul_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    scale_a_ptr,
    scale_b_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Scaled Int8xInt8 matrix multiplication kernel.
    Computes C = (A @ B) * (scale_a[:, None] * scale_b[None, :])

    Args:
        a_ptr: Pointer to A matrix (M, K) int8
        b_ptr: Pointer to B matrix (K, N) int8
        c_ptr: Pointer to C matrix (M, N) output
        scale_a_ptr: Pointer to row scales (M, 1) or (M,) float32/bfloat16
        scale_b_ptr: Pointer to col scales (1, N) or (N,) float32/bfloat16
    """
    # Block coordinates
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Offsets for the current block
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Pointers to the start of A and B blocks
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    # Accumulator - use int32 to prevent overflow
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)

    # Loop over K dimension in blocks
    for k in range(0, K, BLOCK_K):
        # Load blocks from A and B
        # Mask out-of-bounds accesses
        k_remaining = K - k
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < k_remaining)
        b_mask = (offs_k[:, None] < k_remaining) & (offs_n[None, :] < N)

        a = tl.load(a_ptrs, mask=a_mask, other=0)
        b = tl.load(b_ptrs, mask=b_mask, other=0)

        # Accumulate
        acc += tl.dot(a, b, out_dtype=tl.int32)

        # Advance pointers
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # Load scaling factors for this block
    # scale_a has shape (M, 1) or (M,), scale_b has shape (1, N) or (N,)
    scale_a_mask = offs_m < M
    scale_b_mask = offs_n < N

    scale_a = tl.load(scale_a_ptr + offs_m, mask=scale_a_mask, other=1.0)
    scale_b = tl.load(scale_b_ptr + offs_n, mask=scale_b_mask, other=1.0)

    # Convert accumulator to float32, apply scaling
    acc_f = acc.to(tl.float32)
    # Broadcast: scale_a is (BLOCK_M,), scale_b is (BLOCK_N,)
    # Result: scale_a[:, None] * scale_b[None, :] -> (BLOCK_M, BLOCK_N)
    scaled = acc_f * scale_a[:, None] * scale_b[None, :]

    # Write back result
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    tl.store(c_ptrs, scaled, mask=c_mask)


def int8_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Int8xInt8 matrix multiplication using Triton.

    Signature matches torch._int_mm:
    - a: (M, K) int8 tensor
    - b: (K, N) int8 tensor
    - returns: (M, N) int32 tensor

    Args:
        a: Input matrix A of shape (M, K) with dtype torch.int8
        b: Input matrix B of shape (K, N) with dtype torch.int8

    Returns:
        Output matrix C of shape (M, N) with dtype torch.int32
    """
    # Validate inputs
    assert a.dtype == torch.int8, f"a must be int8, got {a.dtype}"
    assert b.dtype == torch.int8, f"b must be int8, got {b.dtype}"
    assert a.ndim == 2 and b.ndim == 2, "Inputs must be 2D"
    assert a.shape[1] == b.shape[0], f"Incompatible dimensions: {a.shape} @ {b.shape}"
    assert a.is_cuda and b.is_cuda, "Inputs must be on CUDA"

    M, K = a.shape
    K_b, N = b.shape

    # Allocate output (int32 to match torch._int_mm)
    c = torch.empty((M, N), dtype=torch.int32, device=a.device)

    # Grid configuration
    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_M"]),
        triton.cdiv(N, meta["BLOCK_N"]),
    )

    # Launch kernel
    int8_matmul_kernel[grid](
        a,
        b,
        c,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
    )

    return c


def scaled_int8_matmul(
    a: torch.Tensor,
    b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Scaled Int8xInt8 matrix multiplication using Triton.

    Computes: C = (A @ B) * (scale_a[:, None] * scale_b[None, :])

    Args:
        a: Input matrix A of shape (M, K) with dtype torch.int8
        b: Input matrix B of shape (K, N) with dtype torch.int8
        scale_a: Row scaling factors of shape (M,) or (M, 1) with dtype float32 or bfloat16
        scale_b: Column scaling factors of shape (N,) or (1, N) with dtype float32 or bfloat16
        dtype: Output dtype (default: torch.float32)

    Returns:
        Output matrix C of shape (M, N) with specified dtype
    """
    # Validate inputs
    assert a.dtype == torch.int8, f"a must be int8, got {a.dtype}"
    assert b.dtype == torch.int8, f"b must be int8, got {b.dtype}"
    assert a.ndim == 2 and b.ndim == 2, "Inputs must be 2D"
    assert a.shape[1] == b.shape[0], f"Incompatible dimensions: {a.shape} @ {b.shape}"
    assert a.is_cuda and b.is_cuda, "Inputs must be on CUDA"

    M, K = a.shape
    K_b, N = b.shape

    # Validate and reshape scales
    if scale_a.ndim == 2:
        assert scale_a.shape == (
            M,
            1,
        ), f"scale_a shape {scale_a.shape} incompatible with M={M}"
        scale_a = scale_a.squeeze(1)
    assert scale_a.shape == (
        M,
    ), f"scale_a must be shape (M,) or (M, 1), got {scale_a.shape}"

    if scale_b.ndim == 2:
        assert scale_b.shape == (
            1,
            N,
        ), f"scale_b shape {scale_b.shape} incompatible with N={N}"
        scale_b = scale_b.squeeze(0)
    assert scale_b.shape == (
        N,
    ), f"scale_b must be shape (N,) or (1, N), got {scale_b.shape}"

    assert scale_a.dtype in [
        torch.float32,
        torch.bfloat16,
    ], f"scale_a must be float32 or bfloat16, got {scale_a.dtype}"
    assert scale_b.dtype in [
        torch.float32,
        torch.bfloat16,
    ], f"scale_b must be float32 or bfloat16, got {scale_b.dtype}"
    assert scale_a.is_cuda and scale_b.is_cuda, "Scales must be on CUDA"

    # Allocate output
    c = torch.empty((M, N), dtype=dtype, device=a.device)

    # Grid configuration
    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_M"]),
        triton.cdiv(N, meta["BLOCK_N"]),
    )

    # Launch kernel
    scaled_int8_matmul_kernel[grid](
        a,
        b,
        c,
        scale_a,
        scale_b,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
    )

    return c
