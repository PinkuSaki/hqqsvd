import importlib.util
from functools import lru_cache
import torch

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
    def _unpack_uint4_kernel(packed_ptr, output_ptr, n_elements, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offsets = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < n_elements
        packed = tl.load(packed_ptr + offsets, mask=mask, other=0)
        low = packed & 0x0F
        high = packed >> 4
        out_offsets = offsets * 2
        tl.store(output_ptr + out_offsets, low, mask=mask)
        tl.store(output_ptr + out_offsets + 1, high, mask=mask)

else:
    _unpack_uint4_kernel = None


@lru_cache(maxsize=1)
def _get_unpack_uint4_kernel():
    if _unpack_uint4_kernel is None:
        raise RuntimeError("Triton is not available for uint4 unpack.")
    return _unpack_uint4_kernel


def unpack_uint4_triton(
    packed_tensor: torch.ByteTensor, shape: torch.Size
) -> torch.ByteTensor:
    import triton

    packed_tensor = packed_tensor.contiguous().view(-1)
    n_elements = packed_tensor.numel()
    output = torch.empty(n_elements * 2, device=packed_tensor.device, dtype=torch.uint8)
    kernel = _get_unpack_uint4_kernel()
    grid = (triton.cdiv(n_elements, 1024),)
    kernel[grid](packed_tensor, output, n_elements, BLOCK=1024)
    return output.view(shape)


def pack_uint7(tensor: torch.ByteTensor) -> torch.ByteTensor:
    packed_tensor = tensor.contiguous().view(-1, 8)
    packed_tensor = torch.bitwise_or(
        packed_tensor[:, :7],
        torch.bitwise_and(
            torch.stack(
                (
                    torch.bitwise_left_shift(packed_tensor[:, 7], 1),
                    torch.bitwise_left_shift(packed_tensor[:, 7], 2),
                    torch.bitwise_left_shift(packed_tensor[:, 7], 3),
                    torch.bitwise_left_shift(packed_tensor[:, 7], 4),
                    torch.bitwise_left_shift(packed_tensor[:, 7], 5),
                    torch.bitwise_left_shift(packed_tensor[:, 7], 6),
                    torch.bitwise_left_shift(packed_tensor[:, 7], 7),
                ),
                dim=-1,
            ),
            128,
        ),
    )
    return packed_tensor


def pack_uint6(tensor: torch.ByteTensor) -> torch.ByteTensor:
    packed_tensor = tensor.contiguous().view(-1, 4)
    packed_tensor = torch.cat(
        (
            torch.bitwise_or(
                packed_tensor[:, :2],
                torch.bitwise_and(
                    torch.stack(
                        (
                            torch.bitwise_left_shift(packed_tensor[:, 3], 2),
                            torch.bitwise_left_shift(packed_tensor[:, 3], 4),
                        ),
                        dim=-1,
                    ),
                    192,
                ),
            ),
            torch.bitwise_or(
                packed_tensor[:, 2], torch.bitwise_left_shift(packed_tensor[:, 3], 6)
            ).unsqueeze(-1),
        ),
        dim=-1,
    )
    return packed_tensor


def pack_uint5(tensor: torch.ByteTensor) -> torch.ByteTensor:
    packed_tensor = tensor.contiguous().view(-1, 8)
    packed_tensor = torch.cat(
        (
            torch.bitwise_or(
                packed_tensor[:, :3], torch.bitwise_left_shift(packed_tensor[:, 5:8], 5)
            ),
            torch.bitwise_or(
                packed_tensor[:, 3],
                torch.bitwise_or(
                    torch.bitwise_and(
                        torch.bitwise_left_shift(packed_tensor[:, 5], 2), 96
                    ),
                    torch.bitwise_and(
                        torch.bitwise_left_shift(packed_tensor[:, 7], 3), 128
                    ),
                ),
            ).unsqueeze(-1),
            torch.bitwise_or(
                packed_tensor[:, 4],
                torch.bitwise_or(
                    torch.bitwise_and(
                        torch.bitwise_left_shift(packed_tensor[:, 6], 2), 96
                    ),
                    torch.bitwise_and(
                        torch.bitwise_left_shift(packed_tensor[:, 7], 4), 128
                    ),
                ),
            ).unsqueeze(-1),
        ),
        dim=-1,
    )
    return packed_tensor


def pack_uint4(tensor: torch.ByteTensor) -> torch.ByteTensor:
    packed_tensor = tensor.contiguous().view(-1, 2)
    packed_tensor = torch.bitwise_or(
        packed_tensor[:, 0], torch.bitwise_left_shift(packed_tensor[:, 1], 4)
    )
    return packed_tensor


def pack_uint3(tensor: torch.ByteTensor) -> torch.ByteTensor:
    packed_tensor = tensor.contiguous().view(-1, 8)
    packed_tensor = torch.bitwise_or(
        torch.bitwise_or(
            packed_tensor[:, :3], torch.bitwise_left_shift(packed_tensor[:, 3:6], 3)
        ),
        torch.cat(
            (
                torch.bitwise_left_shift(packed_tensor[:, 6:8], 6),
                torch.bitwise_or(
                    torch.bitwise_and(
                        torch.bitwise_left_shift(packed_tensor[:, 6], 4), 64
                    ),
                    torch.bitwise_and(
                        torch.bitwise_left_shift(packed_tensor[:, 7], 5), 128
                    ),
                ).unsqueeze(-1),
            ),
            dim=-1,
        ),
    )
    return packed_tensor


def pack_uint2(tensor: torch.ByteTensor) -> torch.ByteTensor:
    packed_tensor = tensor.contiguous().view(-1, 4)
    packed_tensor = torch.bitwise_or(
        torch.bitwise_or(
            packed_tensor[:, 0], torch.bitwise_left_shift(packed_tensor[:, 1], 2)
        ),
        torch.bitwise_or(
            torch.bitwise_left_shift(packed_tensor[:, 2], 4),
            torch.bitwise_left_shift(packed_tensor[:, 3], 6),
        ),
    )
    return packed_tensor


def pack_uint1(tensor: torch.Tensor) -> torch.Tensor:
    packed_tensor = tensor.contiguous().view(-1, 8)
    packed_tensor = torch.bitwise_or(
        torch.bitwise_or(
            torch.bitwise_or(
                packed_tensor[:, 0], torch.bitwise_left_shift(packed_tensor[:, 1], 1)
            ),
            torch.bitwise_or(
                torch.bitwise_left_shift(packed_tensor[:, 2], 2),
                torch.bitwise_left_shift(packed_tensor[:, 3], 3),
            ),
        ),
        torch.bitwise_or(
            torch.bitwise_or(
                torch.bitwise_left_shift(packed_tensor[:, 4], 4),
                torch.bitwise_left_shift(packed_tensor[:, 5], 5),
            ),
            torch.bitwise_or(
                torch.bitwise_left_shift(packed_tensor[:, 6], 6),
                torch.bitwise_left_shift(packed_tensor[:, 7], 7),
            ),
        ),
    )
    return packed_tensor


def unpack_uint7(
    packed_tensor: torch.ByteTensor, shape: torch.Size
) -> torch.ByteTensor:
    result = torch.cat(
        (
            torch.bitwise_and(packed_tensor[:, :7], 127),
            torch.bitwise_or(
                torch.bitwise_or(
                    torch.bitwise_or(
                        torch.bitwise_and(
                            torch.bitwise_right_shift(packed_tensor[:, 0], 1), 64
                        ),
                        torch.bitwise_and(
                            torch.bitwise_right_shift(packed_tensor[:, 1], 2), 32
                        ),
                    ),
                    torch.bitwise_or(
                        torch.bitwise_and(
                            torch.bitwise_right_shift(packed_tensor[:, 2], 3), 16
                        ),
                        torch.bitwise_and(
                            torch.bitwise_right_shift(packed_tensor[:, 3], 4), 8
                        ),
                    ),
                ),
                torch.bitwise_or(
                    torch.bitwise_or(
                        torch.bitwise_and(
                            torch.bitwise_right_shift(packed_tensor[:, 4], 5), 4
                        ),
                        torch.bitwise_and(
                            torch.bitwise_right_shift(packed_tensor[:, 5], 6), 2
                        ),
                    ),
                    torch.bitwise_right_shift(packed_tensor[:, 6], 7),
                ),
            ).unsqueeze(-1),
        ),
        dim=-1,
    ).view(shape)
    return result


def unpack_uint6(
    packed_tensor: torch.ByteTensor, shape: torch.Size
) -> torch.ByteTensor:
    result = torch.cat(
        (
            torch.bitwise_and(packed_tensor[:, 0:3], 63),
            torch.bitwise_or(
                torch.bitwise_or(
                    torch.bitwise_and(
                        torch.bitwise_right_shift(packed_tensor[:, 0], 2), 48
                    ),
                    torch.bitwise_and(
                        torch.bitwise_right_shift(packed_tensor[:, 1], 4), 12
                    ),
                ),
                torch.bitwise_right_shift(packed_tensor[:, 2], 6),
            ).unsqueeze(-1),
        ),
        dim=-1,
    ).view(shape)
    return result


def unpack_uint5(
    packed_tensor: torch.ByteTensor, shape: torch.Size
) -> torch.ByteTensor:
    result_bitwise_right_shift = torch.bitwise_right_shift(packed_tensor[:, :3], 5)
    result = torch.cat(
        (
            torch.bitwise_and(packed_tensor[:, :5], 31),
            torch.bitwise_or(
                result_bitwise_right_shift[:, :2],
                torch.bitwise_and(
                    torch.bitwise_right_shift(packed_tensor[:, 3:5], 2), 24
                ),
            ),
            torch.bitwise_or(
                result_bitwise_right_shift[:, 2],
                torch.bitwise_or(
                    torch.bitwise_and(
                        torch.bitwise_right_shift(packed_tensor[:, 3], 3), 16
                    ),
                    torch.bitwise_and(
                        torch.bitwise_right_shift(packed_tensor[:, 4], 4), 8
                    ),
                ),
            ).unsqueeze(-1),
        ),
        dim=-1,
    ).view(shape)
    return result


def unpack_uint4(
    packed_tensor: torch.ByteTensor, shape: torch.Size
) -> torch.ByteTensor:
    result = torch.stack(
        (
            torch.bitwise_and(packed_tensor, 15),
            torch.bitwise_right_shift(packed_tensor, 4),
        ),
        dim=-1,
    ).view(shape)
    return result


def unpack_uint3(
    packed_tensor: torch.ByteTensor, shape: torch.Size
) -> torch.ByteTensor:
    result = torch.bitwise_and(
        torch.cat(
            (
                packed_tensor[:, :3],
                torch.bitwise_right_shift(packed_tensor[:, :3], 3),
                torch.bitwise_or(
                    torch.bitwise_right_shift(packed_tensor[:, :2], 6),
                    torch.bitwise_and(
                        torch.stack(
                            (
                                torch.bitwise_right_shift(packed_tensor[:, 2], 4),
                                torch.bitwise_right_shift(packed_tensor[:, 2], 5),
                            ),
                            dim=-1,
                        ),
                        4,
                    ),
                ),
            ),
            dim=-1,
        ),
        7,
    ).view(shape)
    return result


def unpack_uint2(
    packed_tensor: torch.ByteTensor, shape: torch.Size
) -> torch.ByteTensor:
    result = torch.bitwise_and(
        torch.stack(
            (
                packed_tensor,
                torch.bitwise_right_shift(packed_tensor, 2),
                torch.bitwise_right_shift(packed_tensor, 4),
                torch.bitwise_right_shift(packed_tensor, 6),
            ),
            dim=-1,
        ),
        3,
    ).view(shape)
    return result


def unpack_uint1(packed_tensor: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    result = torch.bitwise_and(
        torch.stack(
            (
                packed_tensor,
                torch.bitwise_right_shift(packed_tensor, 1),
                torch.bitwise_right_shift(packed_tensor, 2),
                torch.bitwise_right_shift(packed_tensor, 3),
                torch.bitwise_right_shift(packed_tensor, 4),
                torch.bitwise_right_shift(packed_tensor, 5),
                torch.bitwise_right_shift(packed_tensor, 6),
                torch.bitwise_right_shift(packed_tensor, 7),
            ),
            dim=-1,
        ),
        1,
    ).view(shape)
    return result


pack_functions = {
    8: lambda x: x.flatten(),
    7: pack_uint7,
    6: pack_uint6,
    5: pack_uint5,
    4: pack_uint4,
    3: pack_uint3,
    2: pack_uint2,
    1: pack_uint1,
}


unpack_functions = {
    8: lambda x, y: x.view(y),
    7: unpack_uint7,
    6: unpack_uint6,
    5: unpack_uint5,
    4: unpack_uint4,
    3: unpack_uint3,
    2: unpack_uint2,
    1: unpack_uint1,
}


def pack(tensor: torch.ByteTensor, nbits):
    return pack_functions[nbits](tensor)


def unpack(packed_tensor: torch.ByteTensor, shape: torch.Size, nbits: int):
    global _HAS_TRITON
    if nbits == 4 and packed_tensor.is_cuda and _HAS_TRITON:
        try:
            return unpack_uint4_triton(packed_tensor, shape)
        except Exception:
            _HAS_TRITON = False
    return unpack_functions[nbits](packed_tensor, shape)
