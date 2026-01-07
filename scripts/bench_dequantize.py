#!/usr/bin/env python3
import argparse
import time

import torch

from hqqsvd import bitpack as bitpack_module
from hqqsvd import quantize as quantize_module


def _synchronize_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _time_it(fn, iterations: int, device: torch.device) -> float:
    _synchronize_if_cuda(device)
    start = time.perf_counter()
    for _ in range(iterations):
        fn()
    _synchronize_if_cuda(device)
    end = time.perf_counter()
    return (end - start) / iterations


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark 4-bit dequantization (fused Triton vs. fallback)."
    )
    parser.add_argument("--out-features", type=int, default=4096)
    parser.add_argument("--in-features", type=int, default=4096)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--svd-rank", type=int, default=128)
    parser.add_argument("--svd-steps", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is not available on this system.")

    if args.dtype == "bfloat16":
        dtype = torch.bfloat16
    else:
        dtype = torch.float16

    torch.manual_seed(0)
    weight = torch.randn(
        args.out_features,
        args.in_features,
        device=device,
        dtype=dtype,
    )
    W_q, svd_up, svd_down, scale, zero = quantize_module.quantize(
        weight,
        svd_rank=args.svd_rank,
        svd_steps=args.svd_steps,
        group_size=args.group_size,
        nbits=4,
        fast=True,
    )
    q_shape = torch.Size((args.out_features, args.in_features // args.group_size, args.group_size))
    o_shape = torch.Size((args.out_features, args.in_features))

    def run_fused() -> torch.Tensor:
        return quantize_module.dequantize(
            W_q,
            svd_up,
            svd_down,
            scale,
            zero,
            q_shape,
            o_shape,
            nbits=4,
        )

    original_has_triton_quantize = quantize_module._HAS_TRITON
    original_has_triton_bitpack = bitpack_module._HAS_TRITON

    def run_fallback() -> torch.Tensor:
        quantize_module._HAS_TRITON = False
        bitpack_module._HAS_TRITON = False
        try:
            return quantize_module.dequantize(
                W_q,
                svd_up,
                svd_down,
                scale,
                zero,
                q_shape,
                o_shape,
                nbits=4,
            )
        finally:
            quantize_module._HAS_TRITON = original_has_triton_quantize
            bitpack_module._HAS_TRITON = original_has_triton_bitpack

    for _ in range(args.warmup):
        run_fused()
        run_fallback()

    fused_time = _time_it(run_fused, args.iterations, device)
    fallback_time = _time_it(run_fallback, args.iterations, device)

    print("4-bit dequantization benchmark")
    print(f"device: {device}")
    print(f"shape: ({args.out_features}, {args.in_features})")
    print(f"group size: {args.group_size}")
    print(f"dtype: {dtype}")
    print(f"fused (Triton) avg: {fused_time * 1e3:.3f} ms")
    print(f"fallback avg: {fallback_time * 1e3:.3f} ms")
    if fused_time > 0:
        print(f"speedup: {fallback_time / fused_time:.2f}x")


if __name__ == "__main__":
    main()
