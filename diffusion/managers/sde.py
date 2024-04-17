from torchmanager import losses, metrics
from torchmanager_core import torch, view
from torchmanager_core.typing import Any, Generic, Module, Optional, Union, TypeVar

from diffusion.data import DiffusionData
from diffusion.scheduling import BetaSpace
from diffusion.sde import SDE, SubVPSDE, VESDE, VPSDE
from .diffusion import DiffusionManager

SDEType = TypeVar("SDEType", bound=SDE)


class SDEManager(DiffusionManager[Module], Generic[Module, SDEType]):
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

    def __init__(self, model: Module, /, sde: SDEType, time_steps: int, beta_space: Optional[BetaSpace] = None, *, epsilon: float = 1e-5, is_continous: bool = False, optimizer: Optional[torch.optim.Optimizer] = None, loss_fn: Optional[Union[losses.Loss, dict[str, losses.Loss]]] = None, metrics: dict[str, metrics.Metric] = {}) -> None:
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
        super().__init__(model, time_steps, optimizer, loss_fn, metrics)
        self.beta_space = beta_space
        self.epsilon = epsilon
        self.is_continous = is_continous
        self.sde = sde
        view.warnings.warn("The `SDEManager` is still in beta testing with potential bugs.", category=UserWarning)

        # check parameters
        if isinstance(self.sde, VPSDE) and self.beta_space is None:
            raise ValueError("Beta space is required for VPSDE.")

    def forward(self, x_train: DiffusionData, y_train: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Scale neural network output by standard deviation and flip sign
        # For VE-trained models, t=0 corresponds to the highest noise level
        if isinstance(self.sde, SubVPSDE) or (self.is_continous and isinstance(self.sde, VPSDE)):
            t = x_train.t * (self.sde.N - 1)
            _, std = self.sde.marginal_prob(torch.zeros_like(x_train.x), x_train.t)
        elif isinstance(self.sde, VPSDE):
            assert self.beta_space is not None, "Beta space is required for VPSDE."
            t = x_train.t * (self.sde.N - 1)
            std = self.beta_space.sqrt_one_minus_alphas_cumprod
        elif self.is_continous and isinstance(self.sde, VESDE):
            _, t = self.sde.marginal_prob(torch.zeros_like(x_train.x), x_train.t)
            std = 1
        elif isinstance(self.sde, VESDE):
            t = self.sde.T - x_train.t
            t *= self.sde.N - 1
            t = t.round().long()
            std = 1
        else:
            raise NotImplementedError(f"SDE class {type(self.sde)} not yet supported.")

        # calculate using score function
        x = DiffusionData(x_train.x, t, condition=x_train.condition)
        score = self.model(x)
        y = score / std

        # calculate loss
        loss = self.compiled_losses(y, y_train) if self.loss_fn is not None and y_train is not None else None
        return y, loss

    def forward_diffusion(self, data: torch.Tensor, condition: Optional[torch.Tensor] = None, t: Optional[torch.Tensor] = None) -> tuple[Any, torch.Tensor]:
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
        z = torch.randn_like(data, device=t.device)
        mean, std = self.sde.marginal_prob(data, t)
        x = mean + std[:, None, None, None] * z
        noise = z / std[:, None, None, None]
        return DiffusionData(x, t, condition=condition), noise

    def to(self, device: torch.device) -> None:
        if self.beta_space is not None:
            self.beta_space = self.beta_space.to(device)
        return super().to(device)

    def sampling(self, num_images: int, x_t: torch.Tensor, condition: Optional[torch.Tensor] = None, *, sampling_range: Optional[range] = None, show_verbose: bool = False) -> list[torch.Tensor]:
        sampling_range = range(self.time_steps) if sampling_range is None else sampling_range
        return super().sampling(num_images, x_t, condition, sampling_range=sampling_range, show_verbose=show_verbose)

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
