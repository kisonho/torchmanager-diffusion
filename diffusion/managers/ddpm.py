from torchmanager_core import torch
from torchmanager_core.typing import Any, TypeVar

from diffusion.data import DiffusionData

from ..data import DiffusionData
from ..nn import DiffusionModule
from .diffusion import DiffusionManager

Module = TypeVar("Module", bound=DiffusionModule)


class DDPMManager(DiffusionManager[Module]):
    """Main DDPM Manager"""

    def forward_diffusion(self, data: torch.Tensor, **kwargs: Any) -> tuple[DiffusionData, torch.Tensor]:
        """
        Forward pass of diffusion model, sample noises

        - Parameters:
            - data: A clear image in `torch.Tensor`
        - Returns: A `tuple` of noisy images and sampled time step in `DiffusionData` and noises in `torch.Tensor`
        """
        # initialize
        x_start = data.to(self.beta_space.device)
        batch_size = x_start.shape[0]
        t = self.beta_space.sample(batch_size, self.time_steps)

        # initialize noises
        noise = torch.randn_like(x_start, device=x_start.device)
        sqrt_alphas_cumprod_t = self.beta_space.sqrt_alphas_cumprod_t(t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self.beta_space.sqrt_one_minus_alphas_cumprod_t(t, x_start.shape)
        x = sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
        return DiffusionData(x, t), noise

    def sampling_step(self, data: DiffusionData, i: int, /) -> torch.Tensor:
        # initialize betas by given t
        betas_t = self.beta_space.betas_t(data.t, data.x.shape)
        sqrt_one_minus_alphas_cumprod_t = self.beta_space.sqrt_one_minus_alphas_cumprod_t(data.t, data.x.shape)
        sqrt_recip_alphas_t = self.beta_space.sqrt_recip_alphas_t(data.t, data.x.shape)

        # Equation 11 in the paper
        # Use our model (noise predictor) to predict the mean
        y, _ = self.forward(data)
        y: torch.Tensor = sqrt_recip_alphas_t * (data.x - betas_t * y / sqrt_one_minus_alphas_cumprod_t)
        if i > 0:
            posterior_variance_t = self.beta_space.posterior_variance_t(data.t, data.x.shape).to(y.device)
            noise = torch.randn_like(data.x, device=y.device)
            # Algorithm 2 line 4:
            y += torch.sqrt(posterior_variance_t) * noise
        return y
