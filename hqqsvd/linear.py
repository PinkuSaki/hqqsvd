import torch
from .quantize import quantize, dequantize


class HQQSVDLinear(torch.nn.Module):
    def __init__(
        self, W_q, svd_up, svd_down, scale, zero_point, bias, nbits
    ):
        super().__init__()
        self.in_features = svd_down.shape[1]
        self.out_features = svd_up.shape[0]
        self.svd_rank = svd_down.shape[0]
        self.group_size = self.in_features // scale.shape[1]
        self.n_groups = scale.shape[1]
        self.q_shape = torch.Size((self.out_features, self.n_groups, self.group_size))
        self.o_shape = torch.Size((self.out_features, self.in_features))
        self.weight = torch.nn.Parameter(W_q, False)
        self.svd_up = torch.nn.Parameter(svd_up, False)
        self.svd_down = torch.nn.Parameter(svd_down, False)
        self.scale = torch.nn.Parameter(scale, False)
        self.zero_point = torch.nn.Parameter(zero_point, False)
        self.bias = torch.nn.Parameter(bias, False)
        self.nbits = torch.nn.Parameter(torch.tensor([nbits]), False) # for serialization
        self._nbits = nbits

    @classmethod
    def from_linear(
        cls,
        linear: torch.nn.Linear,
        svd_rank: int = 128,
        svd_steps: int = 8,
        group_size: int = 128,
        nbits: int = 4,
    ):
        W_q, svd_up, svd_down, scale, zero_point = quantize(
            linear.weight, svd_rank, svd_steps, group_size, nbits
        )
        return cls(
            W_q, svd_up, svd_down, scale, zero_point, linear.bias, nbits
        )

    def dequantize(self):
        return dequantize(
            self.weight,
            self.svd_up,
            self.svd_down,
            self.scale,
            self.zero_point,
            self.q_shape,
            self.o_shape,
            self._nbits
        )

    @torch.compile
    def forward(self, x):
        W_f = self.dequantize()
        return torch.nn.functional.linear(x, W_f, self.bias)
