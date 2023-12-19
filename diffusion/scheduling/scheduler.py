from torchmanager_core import torch
from torchmanager_core.typing import Callable, Enum

from .space import BetaSpace


class BetaScheduler(Enum):
    """The diffusion scheduler that used to calculate schedules according to given schedule"""
    CONSTANT = "constant"
    COSINE = "cosine"
    LINEAR = "linear"
    QUADRATIC = "quadratic"
    SIGMOID = "sigmoid"

    def calculate_space(self, time_steps: int, /) -> BetaSpace:
        """
        Calculate beta space by given steps

        - Parameters:
            - timesteps: An `int` of total time steps required
        - Returns: A `torch.Tensor` of betas in target schedule
        """
        scheduler_scope = globals()
        schedule_fn: Callable[[int], BetaSpace] = scheduler_scope[f"{self.value}_schedule"]
        return schedule_fn(time_steps)

    def calculate_space_with_range(self, time_steps: int, /, beta_start: float, beta_end: float) -> BetaSpace:
        """
        Calculate beta space by given steps and beta range

        - Parameters:
            - timesteps: An `int` of total time steps required
        - Returns: A `torch.Tensor` of betas in target schedule
        """
        if self == BetaScheduler.LINEAR:
            return linear_schedule(time_steps, beta_start=beta_start, beta_end=beta_end)
        elif self == BetaScheduler.SIGMOID:
            return sigmoid_schedule(time_steps, beta_start=beta_start, beta_end=beta_end)
        elif self == BetaScheduler.QUADRATIC:
            return quadratic_schedule(time_steps, beta_start=beta_start, beta_end=beta_end)
        else:
            raise NotImplementedError(f"Schedule '{self.name}' does not support beta range.")


def constant_schedule(time_steps: int, /, beta: float = 0.015) -> BetaSpace:
    """
    A constant schedule that always return a constant beta

    - Parameters:
        - time_steps: An `int` of total time steps required
        - beta: A `float` of the beta value
    - Returns: A `torch.Tensor` of betas wuth given value
    """
    return BetaSpace(torch.zeros(time_steps) + beta)


def cosine_schedule(time_steps: int, /, s: float = 0.008) -> BetaSpace:
    """
    cosine schedule as proposed in https://arxiv.org/abs/2102.09672

    - Parameters:
        - time_steps: An `int` of total time steps required
        - s: A `float` of the cosine schedule parameter
    - Returns: A `torch.Tensor` of betas in cosine schedule
    """
    steps = time_steps + 1
    x = torch.linspace(0, time_steps, steps)
    alphas_cumprod = torch.cos(((x / time_steps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return BetaSpace(betas.clip(min=0, max=0.9999))


def linear_schedule(time_steps: int, /, beta_start: float = 0.0001, beta_end: float = 0.01) -> BetaSpace:
    """
    A linear schedule that always return a linear beta

    - Parameters:
        - time_steps: An `int` of total time steps required
        - beta_start: A `float` of the start beta value
        - beta_end: A `float` of the end beta value
    - Returns: A `torch.Tensor` of betas in linear schedule
    """
    return BetaSpace(torch.linspace(beta_start, beta_end, time_steps))


def quadratic_schedule(time_steps: int, /, beta_start: float = 0.0001, beta_end: float = 0.02) -> BetaSpace:
    """
    A quadratic schedule that always return a quadratic beta

    - Parameters:
        - time_steps: An `int` of total time steps required
        - beta_start: A `float` of the start beta value
        - beta_end: A `float` of the end beta value
    - Returns: A `torch.Tensor` of betas in quadratic schedule
    """
    return BetaSpace(torch.linspace(beta_start ** 0.5, beta_end ** 0.5, time_steps) ** 2)


def sigmoid_schedule(time_steps: int, /, beta_start: float = 0.0001, beta_end: float = 0.02) -> BetaSpace:
    """
    A sigmoid schedule that always return a sigmoid beta

    - Parameters:
        - time_steps: An `int` of total time steps required
        - beta_start: A `float` of the start beta value
        - beta_end: A `float` of the end beta value
    - Returns: A `torch.Tensor` of betas in sigmoid schedule
    """
    betas = torch.linspace(-3, 3, time_steps)
    betas = betas.sigmoid()
    betas = (betas - betas.min()) / (betas.max() - betas.min())
    return BetaSpace(betas * (beta_end - beta_start) + beta_start)
