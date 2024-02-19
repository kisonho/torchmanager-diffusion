from torchmanager_core import abc, torch
from torchmanager_core.typing import Any, Generic, Module, Optional, Union, overload

from diffusion.data import DiffusionData
from diffusion.scheduling.space import BetaSpace
from .protocols import TimedData


class TimedModule(torch.nn.Module, abc.ABC):
    """
    The basic diffusion model

    * extends: `torch.nn.Module`
    * Abstract class

    - method to implement:
        - unpack_data: The method that accepts inputs perform to `.protocols.TimedData` to unpack the given inputs and passed to `forward` method
    """

    def __call__(self, x_in: TimedData, *args: Any, **kwargs: Any) -> Any:
        data = self.unpack_data(x_in)
        return super().__call__(*data, *args, **kwargs)

    @abc.abstractmethod
    def unpack_data(self, x_in: TimedData) -> tuple[Any, ...]:
        """
        Method to unpack `TimedData`, the unpacked data will be passed as positional arguments to `forward` method

        - Parameters:
            x_in: The `TimedData` to unpack
        - Returns: A `tuple` of returned unpacked data
        """
        return NotImplemented


class DiffusionModel(TimedModule, Generic[Module], abc.ABC):
    """
    The diffusion model that has the forward diffusion and sampling step algorithm implemented

    * extends: `TimedModule`
    * Abstract class
    * Generic: `Module`

    - Properties:
        - model: The model to use for diffusion in `Module`
    - method to implement:
        - forward_diffusion: The forward pass of diffusion model, sample noises
        - sampling_step: The sampling step of diffusion model
    """
    model: Module

    def __init__(self, model: Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.model(*args, **kwargs)

    @abc.abstractmethod
    def forward_diffusion(self, data: Any, t: torch.Tensor, /, condition: Optional[torch.Tensor] = None) -> tuple[Any, torch.Tensor]:
        """
        Forward pass of diffusion model, sample noises

        - Parameters:
            - data: Any kind of noised data
            - t: A `torch.Tensor` of the time step, sampling uniformly if not given
            - condition: An optional `torch.Tensor` of the condition to generate images
        - Returns: A `tuple` of noisy images and sampled time step in `DiffusionData` and noises in `torch.Tensor`
        """
        return NotImplemented

    @overload
    @abc.abstractmethod
    def sampling_step(self, data: DiffusionData, i: int, /, *, return_noise: bool = False) -> torch.Tensor:
        ...

    @overload
    @abc.abstractmethod
    def sampling_step(self, data: DiffusionData, i: int, /, *, return_noise: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
        ...

    @abc.abstractmethod
    def sampling_step(self, data: DiffusionData, i: int, /, *, return_noise: bool = False) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        Sampling step of diffusion model

        - Parameters:
            - data: A `DiffusionData` object
            - i: An `int` of current time step
            - return_noise: A `bool` flag to return predicted noise
        - Returns: A `torch.Tensor` of noised image if not returning noise or a `tuple` of noised image and predicted noise in `torch.Tensor` if returning noise
        """
        return NotImplemented


class DDPM(DiffusionModel[Module]):
    """
    The main DDPM model

    * extends: `DiffusionModel`
    * Generic: `Module`

    - Properties:
        - beta_space: A scheduled `BetaSpace`
    """
    beta_space: BetaSpace

    def __init__(self, model: Module, beta_space: BetaSpace) -> None:
        super().__init__(model)
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
        predicted_noise, _ = self.forward(data)
        assert isinstance(predicted_noise, torch.Tensor), "The model must return a `torch.Tensor` as predicted noise."
        y: torch.Tensor = sqrt_recip_alphas_t * (data.x - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t)
        if i > 1:
            posterior_variance_t = self.beta_space.sample_posterior_variance(data.t, data.x.shape).to(y.device)
            noise = torch.randn_like(data.x, device=y.device)
            # Algorithm 2 line 4:
            y += torch.sqrt(posterior_variance_t) * noise
        return (y, predicted_noise) if return_noise else y
