from typing_extensions import override
import torch


class PreNorm(torch.nn.Module):
    @override
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = torch.nn.GroupNorm(1, dim)

    @override
    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)
