import torch
import numpy as np
from torch import float32, float16, Tensor
from functools import partial
from typing import Union

# Greedy local search: Only tested with axis==0
@torch.compile
def update_scale_grid_search(
    W_f: Tensor, scale: Tensor, zero: Tensor, axis: int, min_max: list, N: int = 128 + 1
) -> Tensor:
    # Make sure it's an odd number so that the original scale is included
    assert N % 2 == 1, "Please check whether N: odd number"
    rng_dump = 0.05  # 0.05 / 1.
    z_val = 2e-4

    device = scale.device
    dtype = scale.dtype
    ###############################
    W_q = torch.round(W_f * scale + zero).clamp(min_max[0], min_max[1])
    n_clusters = max(W_q.shape[0], W_q.shape[1])
    rng = torch.abs(scale).mean() * rng_dump if (rng_dump < 1.0) else rng_dump

    scale_shifted = (
        torch.linspace(-rng, rng, N)[:, None]
        .to(dtype=dtype, device=device)
        .repeat(1, n_clusters)
        + scale
    )

    # Safe inverse
    scale_shifted[
        torch.logical_and(scale_shifted >= 0, torch.abs(scale_shifted) <= z_val)
    ] = z_val
    scale_shifted[
        torch.logical_and(scale_shifted < 0, torch.abs(scale_shifted) <= z_val)
    ] = -z_val

    err = torch.empty([N, n_clusters], dtype=dtype, device=device)
    for i in range(N):
        W_r = (W_q - zero) / scale_shifted[i][None, :]
        err[i] = torch.abs(W_f - W_r).mean(axis=axis, keepdim=True)

    ind_r = torch.argmin(err, axis=axis).to(torch.int32)
    ind_c = torch.arange(len(ind_r), dtype=torch.int32, device=device)
    scale_b = scale_shifted[ind_r, ind_c]

    return scale_b

# Shrinking operator
@torch.compile
def shrink_lp_op(x: Tensor, beta: float, lp_norm: float) -> Tensor:
    if lp_norm == 1: 
        #torch.sign(x) * torch.nn.functional.relu(torch.abs(x) - 1.0 / beta)
        out = torch.abs(x)
        out.sub_(1.0 / beta).clamp_min_(0.0)
        out.mul_(torch.sign(x))
        return out
    else:
        #torch.sign(x) * torch.nn.functional.relu(torch.abs(x) - (1.0 / beta) * torch.pow(torch.abs(x), lp_norm - 1))
        out = torch.abs(x)
        out.sub_((1.0 / beta) * out.pow(lp_norm - 1)).clamp_min_(0.0)
        out.mul_(torch.sign(x))
        return out


@torch.inference_mode()
def optimize_weights_proximal_v2(
    tensor: Tensor,
    scale: Tensor,
    zero: Tensor,
    min_max: list,
    axis: int = 0,
    device: Union[str, None] = None,
    dtype: Union[torch.dtype, None] = None,
    opt_params: dict = {
        "lp_norm": 0.7,
        "beta": 1e1,
        "kappa": 1.01,
        "iters": 20,
        "tol": 0.0,
        "early_stop": True,
        "scale_gridsearch": False,
    },
    verbose: bool = False,
) -> tuple:
    # Params
    lp_norm = max(opt_params["lp_norm"], 0.1)
    beta = opt_params["beta"]
    kappa = opt_params["kappa"]
    iters = opt_params["iters"]
    early_stop = opt_params["early_stop"]
    tol = opt_params["tol"]

    # Check
    assert lp_norm <= 1.0, "lp_norm should be <=1"
    assert beta > 0.0, "beta should be > 0"
    assert kappa > 1.0, "kappa should be > 1"
    assert iters > 1, "iters should be > 1"

    # Cast/device
    if device is None:
        device = tensor.device
    else:
        device = torch.device(device)

    if dtype is None:
        dtype = float16 if (device.type == "cuda") else float32

    W_f = tensor.to(device=device, dtype=dtype)
    scale = scale.to(device=device, dtype=dtype)
    zero = zero.to(device=device, dtype=dtype)

    # Update scale: works slightly better. Tested on Llama2 only
    if opt_params["scale_gridsearch"]:
        scale = update_scale_grid_search(W_f, scale, zero, axis, min_max)

    # Optimize for zero-point
    best_error = torch.tensor(1e4, dtype=float32, device=device)
    scale_prev, zero_prev = scale.clone(), zero.clone()
    for i in range(iters):
        W_q = torch.round(W_f * scale + zero).clamp(min_max[0], min_max[1])
        W_r = (W_q - zero) / scale

        # current_error = float(torch.pow(torch.abs(W_f - W_r), max(0.80, lp_norm)).mean())
        current_error = torch.abs(W_f - W_r).mean().float()

        if verbose:
            print(i, np.round(current_error, 6))

        if early_stop:
            if best_error - current_error > tol:
                best_error = current_error
                scale_prev, zero_prev = scale.clone(), zero.clone()
            else:
                scale, zero = scale_prev.clone(), zero_prev.clone()
                break

        W_e = shrink_lp_op(W_f - W_r, beta, lp_norm)
        zero = torch.mean(W_q - (W_f - W_e) * scale, axis=axis, keepdim=True)
        beta *= kappa

    # Clean-up
    scale = scale.to(tensor.device)
    zero = zero.to(tensor.device)
    del W_f, W_q, W_r, W_e, scale_prev, zero_prev
    torch.cuda.empty_cache()

    W_q = (torch.floor(tensor * scale + zero) + (torch.rand_like(tensor) < (tensor * scale + zero - torch.floor(tensor * scale + zero))).float()).clamp(min_max[0], min_max[1])

    return W_q, scale, zero


optimize_weights = partial(
    optimize_weights_proximal_v2,
    dtype=torch.float32,
    opt_params={
        "lp_norm": 0.7,
        "beta": 1e1,
        "kappa": 1.01,
        "iters": 100,
        "tol": 0.0,
        "early_stop": True,
        "scale_gridsearch": False,
    },
)
