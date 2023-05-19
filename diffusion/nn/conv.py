import torch
from functools import partial
from torch.nn import functional as F
from typing import Generic, Optional, Type, TypeVar
from einops import rearrange, reduce

M = TypeVar("M", bound=torch.nn.Module)


class WeightStandardizedConv2d(torch.nn.Conv2d):
    """
    https://arxiv.org/abs/1903.10520
    weight standardization purportedly works synergistically with group normalization
    """
    @property
    def normalized_weight(self) -> torch.Tensor:
        eps = 1e-5
        mean = reduce(self.weight, "o ... -> o 1 1 1", "mean")
        var = reduce(self.weight, "o ... -> o 1 1 1", partial(torch.var, unbiased=False))
        return (self.weight - mean) * (var + eps).rsqrt()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(
            x,
            self.normalized_weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class ConvBlock(torch.nn.Module):
    def __init__(self, dim: int, dim_out: int, conv_type: Type[torch.nn.Conv2d] = torch.nn.Conv2d, groups: int = 8) -> None:
        super().__init__()
        Conv2d = conv_type
        self.proj = Conv2d(dim, dim_out, 3, padding=1)
        self.norm = torch.nn.GroupNorm(groups, dim_out)
        self.act = torch.nn.SiLU()

    def forward(self, x: torch.Tensor, scale_shift: Optional[tuple[torch.Tensor, torch.Tensor]] = None) -> torch.Tensor:
        x = self.proj(x)
        x = self.norm(x)
        if scale_shift is not None:
            scale, shift = scale_shift
            x = x * (scale + 1) + shift
        x = self.act(x)
        return x


class ConvNextBlock(torch.nn.Module):
    """https://arxiv.org/abs/2201.03545"""

    def __init__(self, dim: int, dim_out: int, *, conv_type: Type[torch.nn.Conv2d] = torch.nn.Conv2d, dropout: float = 0, time_emb_dim: Optional[int] = None, mult: int = 2, norm: bool = True) -> None:
        super().__init__()
        Conv2d = conv_type
        self.mlp = (torch.nn.Sequential(torch.nn.GELU(), torch.nn.Linear(time_emb_dim, dim)) if time_emb_dim is not None else None)
        self.dropout = torch.nn.Dropout2d(dropout)
        self.ds_conv = Conv2d(dim, dim, 7, padding=3, groups=dim)
        self.net = torch.nn.Sequential(
            torch.nn.GroupNorm(1, dim) if norm else torch.nn.Identity(),
            Conv2d(dim, dim_out * mult, 3, padding=1),
            torch.nn.GELU(),
            torch.nn.GroupNorm(1, dim_out * mult),
            Conv2d(dim_out * mult, dim_out, 3, padding=1),
        )
        self.res_conv = Conv2d(dim, dim_out, 1) if dim != dim_out else torch.nn.Identity()

    def forward(self, x: torch.Tensor, time_emb: Optional[int] = None) -> torch.Tensor:
        h = self.ds_conv(x)
        if self.mlp is not None and time_emb is not None:
            assert time_emb is not None, "Time embedding must be passed in"
            condition = self.mlp(time_emb)
            h = h + rearrange(condition, "b c -> b c 1 1")
        h = self.net(h)
        h = self.dropout(h)
        return h + self.res_conv(x)


class Residual(torch.nn.Module, Generic[M]):
    def __init__(self, fn: M) -> None:
        super().__init__()
        self.fn = fn

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return self.fn(x, *args, **kwargs) + x


class ResnetBlock(torch.nn.Module):
    """https://arxiv.org/abs/1512.03385"""

    def __init__(self, dim: int, dim_out: int, *, conv_type: Type[torch.nn.Conv2d] = torch.nn.Conv2d, dropout: float = 0, time_emb_dim: Optional[int] = None, groups: int = 8) -> None:
        super().__init__()
        Conv2d = conv_type
        self.mlp = (torch.nn.Sequential(torch.nn.SiLU(), torch.nn.Linear(time_emb_dim, dim_out)) if time_emb_dim is not None else None)
        self.block1 = ConvBlock(dim, dim_out, conv_type=Conv2d, groups=groups)
        self.dropout = torch.nn.Dropout2d(dropout)
        self.block2 = ConvBlock(dim_out, dim_out, conv_type=Conv2d, groups=groups)
        self.res_conv = Conv2d(dim, dim_out, 1) if dim != dim_out else torch.nn.Identity()

    def forward(self, x: torch.Tensor, time_emb: Optional[int] = None) -> torch.Tensor:
        h = self.block1(x)
        if self.mlp is not None and time_emb is not None:
            time_emb = self.mlp(time_emb)
            h = rearrange(time_emb, "b c -> b c 1 1") + h
        h = self.dropout(h)
        h = self.block2(h)
        return h + self.res_conv(x)
