import torch
from typing import Optional, TypeVar, Union

from diffusion.data import DiffusionData
from diffusion.scheduling.space import BetaSpace
from .diffusion import DiffusionModule, TimedModule

Module = TypeVar('Module', bound=TimedModule)


class DDPM(DiffusionModule[Module]):
    """
    The main DDPM model

    * extends: `DiffusionModel`
    * Generic: `Module`

    - Properties:
        - beta_space: A scheduled `BetaSpace`
    """
    beta_space: BetaSpace

    def __init__(self, model: Module, beta_space: BetaSpace, time_steps: int) -> None:
        super().__init__(model, time_steps)
        self.beta_space = beta_space

    def forward_diffusion(self, data: torch.Tensor, t: torch.Tensor, /, condition: Optional[torch.Tensor] = None) -> tuple[DiffusionData, torch.Tensor]:
        """
        Forward pass of diffusion model, sample noises

        - Parameters:
            - data: A clear image in `torch.Tensor`
            - t: A `torch.Tensor` of the time step, sampling uniformly if not given
            - condition: An optional condition in `torch.Tensor`
        - Returns: A `tuple` of noisy images and sampled time step in `DiffusionData` and noises in `torch.Tensor`
        """
        # initialize noises
        x_start = data.to(self.beta_space.device)
        noise = torch.randn_like(x_start, device=x_start.device)
        sqrt_alphas_cumprod_t = self.beta_space.sample_sqrt_alphas_cumprod(t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self.beta_space.sample_sqrt_one_minus_alphas_cumprod(t, x_start.shape)
        x = sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
        return DiffusionData(x, t, condition=condition), noise

    def sampling_step(self, data: DiffusionData, i: int, /, *, return_noise: bool = False) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        Sampling step of diffusion model

        - Parameters:
            - data: A `DiffusionData` object
            - i: An `int` of current time step
            - return_noise: A `bool` flag to return predicted noise
        - Returns: A `torch.Tensor` of noised image if not returning noise or a `tuple` of noised image and predicted noise in `torch.Tensor` if returning noise
        """
        # initialize betas by given t
        betas_t = self.beta_space.sample_betas(data.t, data.x.shape)
        sqrt_one_minus_alphas_cumprod_t = self.beta_space.sample_sqrt_one_minus_alphas_cumprod(data.t, data.x.shape)
        sqrt_recip_alphas_t = self.beta_space.sample_sqrt_recip_alphas(data.t, data.x.shape)

        # Equation 11 in the paper
        # Use our model (noise predictor) to predict the mean
        predicted_noise, _ = self(data)
        assert isinstance(predicted_noise, torch.Tensor), "The model must return a `torch.Tensor` as predicted noise."
        y: torch.Tensor = sqrt_recip_alphas_t * (data.x - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t)
        if i > 1:
            posterior_variance_t = self.beta_space.sample_posterior_variance(data.t, data.x.shape).to(y.device)
            noise = torch.randn_like(data.x, device=y.device)
            # Algorithm 2 line 4:
            y += torch.sqrt(posterior_variance_t) * noise
        return (y, predicted_noise) if return_noise else y
