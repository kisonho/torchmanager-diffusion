import torch
from typing import Any, Generic, Optional, Union, TypeVar

from diffusion.data import DiffusionData
from diffusion.scheduling import BetaSpace
from diffusion.sde import SDE, SubVPSDE, VESDE, VPSDE
from .diffusion import DiffusionModule

Module = TypeVar("Module", bound=torch.nn.Module)
SDEType = TypeVar("SDEType", bound=SDE)


class SDEModule(DiffusionModule[Module], Generic[Module, SDEType]):
    """
    A manager for training a neural network to predict the score function of a stochastic differential equation.

    * extends: `.diffusion.DiffusionManager`
    * generic: `Module` and `SDEType`
    * UserWarning: The `SDEManager` is still in beta testing with potential bugs.

    - Properties:
        - beta_space: A scheduled `BetaSpace`
        - epsilon: A `float` of the epsilon value for precision of continuous space
        - is_continous: A `bool` flag of whether the SDE is continous or discrete
        - sde: The SDE in `SDEType` to train
    """
    __epsilon: float
    beta_space: Optional[BetaSpace]
    is_continous: bool
    sde: SDEType

    @property
    def epsilon(self) -> float:
        """A `float` of the epsilon value"""
        return self.__epsilon
    
    @epsilon.setter
    def epsilon(self, value: float) -> None:
        assert value > 0 and value < 1, "The precision epsilon must be in range of (0, 1)."

    def __init__(self, model: Module, sde: SDEType, time_steps: int, *, beta_space: Optional[BetaSpace] = None, epsilon: float = 1e-5, is_continous: bool = False) -> None:
        super().__init__(model, time_steps)
        """
        Constructor

        - Parameters:
            - model: A neural network in `torch.nn.Module` to train
            - sde: The SDE in `SDEType` to train
            - time_steps: A `int` of the number of time steps
            - beta_space: A scheduled `BetaSpace`
            - epsilon: A `float` of the epsilon value for precision of continuous space
            - is_continous: A `bool` flag of whether the SDE is continous or discrete
            - optimizer: A `torch.optim.Optimizer` to optimize the model
            - loss_fn: A `torchmanager.losses.Loss` or a `dict` of `torchmanager.losses.Loss` to calculate loss
            - metrics: A `dict` of `torchmanager.metrics.Metric` to calculate metrics
        """
        super().__init__(model, time_steps)
        self.beta_space = beta_space
        self.epsilon = epsilon
        self.is_continous = is_continous
        self.sde = sde

        # check parameters
        if isinstance(self.sde, VPSDE) and self.beta_space is None:
            raise ValueError("Beta space is required for VPSDE.")

    def forward(self, data: DiffusionData) -> torch.Tensor:
        # Scale neural network output by standard deviation and flip sign
        # For VE-trained models, t=0 corresponds to the highest noise level
        if isinstance(self.sde, SubVPSDE) or (self.is_continous and isinstance(self.sde, VPSDE)):
            t = data.t * (self.sde.N - 1)
            _, std = self.sde.marginal_prob(torch.zeros_like(data.x), data.t)
        elif isinstance(self.sde, VPSDE):
            assert self.beta_space is not None, "Beta space is required for VPSDE."
            t = data.t * (self.sde.N - 1)
            std = self.beta_space.sqrt_one_minus_alphas_cumprod
        elif self.is_continous and isinstance(self.sde, VESDE):
            _, t = self.sde.marginal_prob(torch.zeros_like(data.x), data.t)
            std = 1
        elif isinstance(self.sde, VESDE):
            t = self.sde.T - data.t
            t *= self.sde.N - 1
            t = t.round().long()
            std = 1
        else:
            raise NotImplementedError(f"SDE class {type(self.sde)} not yet supported.")

        # calculate using score function
        x = DiffusionData(data.x, t, condition=data.condition)
        score = super().forward(x)
        y = score / std
        return y

    def forward_diffusion(self, data: torch.Tensor, condition: Optional[torch.Tensor] = None, t: Optional[torch.Tensor] = None) -> tuple[DiffusionData, torch.Tensor]:
        # sampling t
        if t is not None:
            t = t.to(data.device)
        elif self.beta_space is not None:
            t = self.beta_space.sample(data.shape[0], self.time_steps) / self.time_steps
        else:
            t = torch.rand((data.shape[0],), device=data.device)
            t /= self.epsilon
            t = t.long().float() * self.epsilon

        # add noise
        z = self.sde.prior_sampling(data.shape).to(data.device)
        mean, std = self.sde.marginal_prob(data, t)
        x = mean + std[:, None, None, None] * z
        noise = z / std[:, None, None, None]
        return DiffusionData(x, t, condition=condition), noise

    def to(self, device: Union[str, torch.device]) -> 'SDEModule[Module, SDEType]':
        if self.beta_space is not None:
            self.beta_space = self.beta_space.to(torch.device(device))
        return super().to(device)

    def sampling_step(self, data: DiffusionData, i: int, /, *, return_noise: bool = False) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        # predict
        if isinstance(self.sde, VESDE):
            # The ancestral sampling predictor for VESDE
            timestep = ((data.t / self.time_steps) * (self.sde.N - 1) / self.sde.T).long()
            sigma = self.sde.discrete_sigmas[timestep]
            adjacent_sigma = torch.where(timestep == 0, torch.zeros_like(data.t), self.sde.discrete_sigmas.to(data.t.device)[timestep - 1])
            m_t = timestep / self.time_steps
            x = DiffusionData(data.x, m_t, condition=data.condition)
            predicted_noise, _ = score, _ = self.forward(x)
            x_mean = data.x + score * (sigma ** 2 - adjacent_sigma ** 2)[:, None, None, None]
            std = torch.sqrt((adjacent_sigma ** 2 * (sigma ** 2 - adjacent_sigma ** 2)) / (sigma ** 2))
            noise = torch.randn_like(data.x)
            y = x_mean + std[:, None, None, None] * noise
        elif isinstance(self.sde, VPSDE):
            # The ancestral sampling predictor for VESDE
            assert self.beta_space is not None, "Beta space is required for VPSDE."
            timestep = ((data.t / self.time_steps) * (self.sde.N - 1) / self.sde.T).long()
            beta = self.beta_space.sample_betas(timestep, data.x.shape)
            m_t = timestep / self.time_steps
            x = DiffusionData(data.x, m_t, condition=data.condition)
            predicted_noise, _ = score, _ = self.forward(x)
            x_mean = (data.x + beta[:, None, None, None] * score) / torch.sqrt(1. - beta)[:, None, None, None]
            noise = torch.randn_like(data.x)
            y = x_mean + torch.sqrt(beta)[:, None, None, None] * noise
        else:
            # The traditional reverse diffusion predictor
            f, G = self.sde.discretize(data.x, data.t)
            f = f - G[:, None, None, None] ** 2 * self.model(data) * 0.5
            G = torch.zeros_like(G)
            z = torch.randn_like(data.x)
            predicted_noise = x_mean = data.x - f
            y = x_mean + G[:, None, None, None] * z
        return (y, predicted_noise) if return_noise else y
