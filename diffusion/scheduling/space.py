from torch.nn import functional as F
from torchmanager_core import torch
from torchmanager_core.typing import NamedTuple


class BetaSpace(NamedTuple):
    """
    The scheduled beta space for diffusion model

    * extends: `NamedTuple`

    - Properties:
        - alphas: A `torch.Tensor` of related alphas
        - alphas_cumprod: A `torch.Tensor` of alpha bar
        - alphas_cumprod_prev: A `torch.Tensor` of previous alpha bar
        - device: A `torch.device` of the device of betas
        - betas: A `torch.Tensor` of scheduled betas
        - posterior_variance: A `torch.Tensor` of posterior variance
        - sqrt_alphas_cumprod: A `torch.Tensor` of square root of alpha bar
        - sqrt_one_minus_alphas_cumprod: A `torch.Tensor` of square root of one minus alpha bar
        - sqrt_recip_alphas: A `torch.Tensor` of square root of reciprocal alpha
        - shape: A `torch.Size` of the shape of betas
        - space: An optional `torch.Tensor` of t space for sampling
    """
    betas: torch.Tensor
    """A `torch.Tensor` of scheduled betas"""

    @property
    def alphas(self) -> torch.Tensor:
        """A `torch.Tensor` of related alphas"""
        return 1 - self.betas

    @property
    def alphas_cumprod(self) -> torch.Tensor:
        """A `torch.Tensor` of alpha bar"""
        return self.alphas.cumprod(dim=0)

    @property
    def alphas_cumprod_prev(self) -> torch.Tensor:
        """A `torch.Tensor` of previous alpha bar"""
        return F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)

    @property
    def device(self) -> torch.device:
        return self.betas.device

    @property
    def posterior_variance(self) -> torch.Tensor:
        return self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)

    @property
    def sqrt_alphas_cumprod(self) -> torch.Tensor:
        """A `torch.Tensor` of square root of alpha bar"""
        return self.alphas_cumprod.sqrt()

    @property
    def sqrt_one_minus_alphas_cumprod(self) -> torch.Tensor:
        """A `torch.Tensor` of square root of one minus alpha bar"""
        return (1 - self.alphas_cumprod).sqrt()

    @property
    def sqrt_recip_alphas(self) -> torch.Tensor:
        """A `torch.Tensor` of square root of reciprocal alpha"""
        return (1 / self.alphas).sqrt()

    @property
    def shape(self) -> torch.Size:
        """A `torch.Size` of the shape of betas"""
        return self.betas.shape
    
    def __len__(self) -> int:
        return len(self.betas)

    def __repr__(self) -> str:
        return f"<BetaSpace {self.shape}>:\n \
                beta={self.betas}>"
    
    def sample_betas(self, t: torch.Tensor, shape: torch.Size, /) -> torch.Tensor:
        return _get_index_from_list(self.betas, t, shape)

    def sample_alphas(self, t: torch.Tensor, shape: torch.Size, /) -> torch.Tensor:
        return _get_index_from_list(self.alphas, t, shape)

    def sample_posterior_variance(self, t: torch.Tensor, shape: torch.Size, /) -> torch.Tensor:
        return _get_index_from_list(self.posterior_variance, t, shape)

    def sample_sqrt_alphas_cumprod(self, t: torch.Tensor, shape: torch.Size, /) -> torch.Tensor:
        return _get_index_from_list(self.sqrt_alphas_cumprod, t, shape)

    def sample_sqrt_one_minus_alphas_cumprod(self, t: torch.Tensor, shape: torch.Size, /) -> torch.Tensor:
        return _get_index_from_list(self.sqrt_one_minus_alphas_cumprod, t, shape)

    def sample_sqrt_recip_alphas(self, t: torch.Tensor, shape: torch.Size, /) -> torch.Tensor:
        return _get_index_from_list(self.sqrt_recip_alphas, t, shape)

    def sample(self, batch_size: int, time_steps: int, *, start_index: int = 1) -> torch.Tensor:
        """
        Random samples t in given space (linear if not given)

        - Parameters:
            - batch_size: An `int` of the batch size
            - time_steps: An `int` of the total time steps
        - Returns: A random sampled `torch.Tensor` for t
        """
        return torch.randint(start_index, time_steps+start_index, (batch_size,), device=self.device).long()

    def to(self, device: torch.device):
        return BetaSpace(self.betas.to(device))


def _get_index_from_list(vals: torch.Tensor, t: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
    """
    Returns a specific index t of a passed list of values vals
    while considering the batch dimension.
    """
    batch_size = t.shape[0]
    vals = vals.gather(-1, t-1)
    return vals.reshape(batch_size, *((1,) * (len(x_shape) - 1)))
