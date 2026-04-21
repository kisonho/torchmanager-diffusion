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
        self.__epsilon = value

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
        view.warnings.warn("The `SDEManager` has been deprecated, use `nn.diffusion.SDEModule` along with `Manager` instead.", category=DeprecationWarning)

        # check parameters
        if isinstance(self.sde, VPSDE) and self.beta_space is None:
            raise ValueError("Beta space is required for VPSDE.")

    @staticmethod
    def _expand_like(values: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        while values.ndim < x.ndim:
            values = values.unsqueeze(-1)
        return values

    def _continuous_time(self, t: torch.Tensor) -> torch.Tensor:
        t = t.to(dtype=torch.float32)
        if not torch.is_floating_point(t) or torch.any(t > self.sde.T):
            return t * (self.sde.T / self.time_steps)
        return t

    def _run_model(self, data: DiffusionData) -> torch.Tensor:
        if hasattr(self.model, "unpack_data"):
            return self.model(data)
        elif data.condition is not None:
            return self.model(*data)
        return self.model(data.x, data.t)

    def _score(self, x_train: DiffusionData) -> torch.Tensor:
        t = self._continuous_time(x_train.t)
        x = x_train.x

        if isinstance(self.sde, SubVPSDE) or (self.is_continous and isinstance(self.sde, VPSDE)):
            labels = t * 999
            raw_score = self._run_model(DiffusionData(x, labels, condition=x_train.condition))
            _, std = self.sde.marginal_prob(torch.zeros_like(x), t)
            return -raw_score / self._expand_like(std, x)
        elif isinstance(self.sde, VPSDE):
            assert self.beta_space is not None, "Beta space is required for VPSDE."
            labels = t * (self.sde.N - 1)
            raw_score = self._run_model(DiffusionData(x, labels, condition=x_train.condition))
            std = self.beta_space.sqrt_one_minus_alphas_cumprod.to(x.device)[labels.long()]
            return -raw_score / self._expand_like(std, x)
        elif self.is_continous and isinstance(self.sde, VESDE):
            _, labels = self.sde.marginal_prob(torch.zeros_like(x), t)
            return self._run_model(DiffusionData(x, labels, condition=x_train.condition))
        elif isinstance(self.sde, VESDE):
            labels = ((self.sde.T - t) * (self.sde.N - 1)).round().long()
            return self._run_model(DiffusionData(x, labels, condition=x_train.condition))
        raise NotImplementedError(f"SDE class {type(self.sde)} not yet supported.")

    def forward(self, x_train: DiffusionData, y_train: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        y = self._score(x_train)

        # calculate loss
        loss = self.compiled_losses(y, y_train) if self.loss_fn is not None and y_train is not None else None
        return y, loss

    def forward_diffusion(self, data: torch.Tensor, condition: Optional[torch.Tensor] = None, t: Optional[torch.Tensor] = None) -> tuple[Any, torch.Tensor]:
        # sampling t
        if t is not None:
            t = self._continuous_time(t.to(data.device))
        elif isinstance(self.sde, SubVPSDE) or self.is_continous:
            t = torch.rand((data.shape[0],), device=data.device) * (self.sde.T - self.epsilon) + self.epsilon
        elif isinstance(self.sde, VESDE):
            labels = torch.randint(0, self.sde.N, (data.shape[0],), device=data.device)
            t = self.sde.T - labels.float() / max(self.sde.N - 1, 1)
        elif self.beta_space is not None:
            labels = torch.randint(0, self.sde.N, (data.shape[0],), device=data.device)
            t = labels.float() / max(self.sde.N - 1, 1)
        else:
            t = torch.rand((data.shape[0],), device=data.device) * (self.sde.T - self.epsilon) + self.epsilon

        # add noise
        z = torch.randn_like(data, device=t.device)
        mean, std = self.sde.marginal_prob(data, t)
        std = self._expand_like(std, data)
        x = mean + std * z
        score = -z / std
        return DiffusionData(x, t, condition=condition), score

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
            t = self._continuous_time(data.t)
            timestep = (t * (self.sde.N - 1) / self.sde.T).long()
            sigmas = self.sde.discrete_sigmas.to(data.x.device)
            sigma = sigmas[timestep]
            adjacent_sigma = torch.where(timestep == 0, torch.zeros_like(sigma), sigmas[timestep - 1])
            score, _ = self.forward(DiffusionData(data.x, t, condition=data.condition))
            x_mean = data.x + score * (sigma ** 2 - adjacent_sigma ** 2)[:, None, None, None]
            std = torch.sqrt(torch.clamp((adjacent_sigma ** 2 * (sigma ** 2 - adjacent_sigma ** 2)) / torch.clamp(sigma ** 2, min=1e-12), min=0.0))
            noise = torch.randn_like(data.x)
            y = x_mean + std[:, None, None, None] * noise
            predicted_score = score
        elif isinstance(self.sde, VPSDE) and not isinstance(self.sde, SubVPSDE):
            # The ancestral sampling predictor for VPSDE
            assert self.beta_space is not None, "Beta space is required for VPSDE."
            t = self._continuous_time(data.t)
            timestep = (t * (self.sde.N - 1) / self.sde.T).long()
            beta = self.beta_space.betas.to(data.x.device)[timestep]
            score, _ = self.forward(DiffusionData(data.x, t, condition=data.condition))
            x_mean = (data.x + beta[:, None, None, None] * score) / torch.sqrt(1. - beta)[:, None, None, None]
            noise = torch.randn_like(data.x)
            y = x_mean + torch.sqrt(beta)[:, None, None, None] * noise
            predicted_score = score
        else:
            # The traditional reverse diffusion predictor
            t = self._continuous_time(data.t)
            score, _ = self.forward(DiffusionData(data.x, t, condition=data.condition))
            f, G = self.sde.discretize(data.x, t)
            f = f - G[:, None, None, None] ** 2 * score
            z = torch.randn_like(data.x)
            x_mean = data.x - f
            y = x_mean + G[:, None, None, None] * z
            predicted_score = score
        return (y, predicted_score) if return_noise else y
