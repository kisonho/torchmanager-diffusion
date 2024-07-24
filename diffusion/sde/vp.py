import torch
from typing import Collection, Optional

from ..scheduling import BetaScheduler, BetaSpace
from .base import SDE


class VPSDE(SDE):
    beta_space: BetaSpace

    @property
    def beta_0(self) -> torch.Tensor:
        return self.beta_space.betas[0]

    @property
    def beta_1(self) -> torch.Tensor:
        return self.beta_space.betas[-1]
    
    def __init__(self, N: int, /, beta_space: Optional[BetaSpace] = None) -> None:
        """
        Construct a VP-SDE.

        - Parameters:
            - N: An `int` of the number of discretization time steps.
            - beta_space: An optional `BetaSpace` of the beta space, a linear schedule by default if not provided.
        """
        super().__init__(N)
        self.beta_space = BetaScheduler.LINEAR.calculate_space(N) if beta_space is None else beta_space
        self.N = N

    def __call__(self, x: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        beta_t = self.beta_0 + t * (self.beta_1 - self.beta_0)
        drift = -0.5 * beta_t[:, None, None, None] * x
        diffusion = torch.sqrt(beta_t)
        return drift, diffusion

    def discretize(self, x: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """DDPM discretization."""
        timestep = (t * (self.N - 1) / self.T).long()
        beta = self.beta_space.betas[timestep]
        alpha = self.beta_space.alphas[timestep]
        sqrt_beta = torch.sqrt(beta)
        f = torch.sqrt(alpha)[:, None, None, None] * x - x
        G = sqrt_beta
        return f, G

    def marginal_prob(self, x: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        log_mean_coeff = -0.25 * t ** 2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
        mean = torch.exp(log_mean_coeff[:, None, None, None]) * x
        std = torch.sqrt(1. - torch.exp(2. * log_mean_coeff))
        return mean, std

    def prior_logp(self, z: torch.Tensor) -> torch.Tensor:
        shape = z.shape
        N = torch.tensor(shape[1:], device=z.device).prod()
        logps = - N / 2. * torch.tensor(2 * torch.pi, device=z.device).log() - (z ** 2).sum(dim=(1, 2, 3)) / 2.
        return logps

    def prior_sampling(self, shape: Collection[int]) -> torch.Tensor:
        return torch.randn(*shape)


class SubVPSDE(VPSDE):
    def __call__(self, x: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        drift, diffusion = super().__call__(x, t)
        discount = 1. - torch.exp(-2 * self.beta_0 * t - (self.beta_1 - self.beta_0) * t ** 2)
        return drift, diffusion * discount.sqrt()

    def marginal_prob(self, x: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, std = super().marginal_prob(x, t)
        return mean, std ** 2
