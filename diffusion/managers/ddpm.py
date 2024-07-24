from torchmanager import losses, metrics
from torchmanager_core import torch, deprecated
from torchmanager_core.typing import Module, Optional, Union

from diffusion.version import CURRENT
from diffusion.data import DiffusionData
from diffusion.scheduling import BetaSpace
from .diffusion import DiffusionManager


class DDPMManager(DiffusionManager[Module]):
    """
    Main DDPM Manager

    - Properties:
        - beta_space: A scheduled `BetaSpace`
    """
    beta_space: BetaSpace

    def __init__(self, model: Module, beta_space: BetaSpace, time_steps: int, optimizer: Optional[torch.optim.Optimizer] = None, loss_fn: Optional[Union[losses.Loss, dict[str, losses.Loss]]] = None, metrics: dict[str, metrics.Metric] = {}) -> None:
        super().__init__(model, time_steps, optimizer, loss_fn, metrics)
        self.beta_space = beta_space

    def forward_diffusion(self, data: torch.Tensor, condition: Optional[torch.Tensor] = None, t: Optional[torch.Tensor] = None) -> tuple[DiffusionData, torch.Tensor]:
        """
        Forward pass of diffusion model, sample noises

        - Parameters:
            - data: A clear image in `torch.Tensor`
            - condition: An optional condition in `torch.Tensor`
            - t: An optional time step in `torch.Tensor`
        - Returns: A `tuple` of noisy images and sampled time step in `DiffusionData` and noises in `torch.Tensor`
        """
        # initialize
        x_start = data.to(self.beta_space.device)
        batch_size = x_start.shape[0]
        t = self.beta_space.sample(batch_size, self.time_steps) if t is None else t.to(x_start.device)

        # initialize noises
        noise = torch.randn_like(x_start, device=x_start.device)
        sqrt_alphas_cumprod_t = self.beta_space.sample_sqrt_alphas_cumprod(t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self.beta_space.sample_sqrt_one_minus_alphas_cumprod(t, x_start.shape)
        x = sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
        return DiffusionData(x, t, condition=condition), noise

    def to(self, device: torch.device) -> None:
        self.beta_space = self.beta_space.to(device)
        return super().to(device)

    def sampling_step(self, data: DiffusionData, i, /, *, return_noise: bool = False) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        # initialize betas by given t
        betas_t = self.beta_space.sample_betas(data.t, data.x.shape)
        sqrt_one_minus_alphas_cumprod_t = self.beta_space.sample_sqrt_one_minus_alphas_cumprod(data.t, data.x.shape)
        sqrt_recip_alphas_t = self.beta_space.sample_sqrt_recip_alphas(data.t, data.x.shape)

        # Equation 11 in the paper
        # Use our model (noise predictor) to predict the mean
        predicted_noise, _ = self.forward(data)
        assert isinstance(predicted_noise, torch.Tensor), "The model must return a `torch.Tensor` as predicted noise."
        y: torch.Tensor = sqrt_recip_alphas_t * (data.x - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t)
        if i > 1:
            posterior_variance_t = self.beta_space.sample_posterior_variance(data.t, data.x.shape).to(y.device)
            noise = torch.randn_like(data.x, device=y.device)
            # Algorithm 2 line 4:
            y += torch.sqrt(posterior_variance_t) * noise
        return (y, predicted_noise) if return_noise else y
