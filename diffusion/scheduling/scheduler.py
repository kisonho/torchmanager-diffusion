from torchmanager_core import torch
from torchmanager_core.typing import Callable, Enum, Optional

from .space import BetaSpace


class BetaScheduler(Enum):
    """The diffusion scheduler that used to calculate schedules according to given schedule"""
    CONSTANT = "constant"
    COSINE = "cosine"
    HARMONIC = "harmonic"
    MID_LINEAR = "mid_linear"
    LINEAR = "linear"
    PEAK_LINEAR = "peak_linear"
    QUADRATIC = "quadratic"
    REVERSE_SIGMOID = "reverse_sigmoid"
    SIGMOID = "sigmoid"
    STEP = "step"

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
        if self == BetaScheduler.MID_LINEAR:
            return mid_linear_schedule(time_steps, beta_start=beta_start, beta_end=beta_end)
        elif self == BetaScheduler.LINEAR:
            return linear_schedule(time_steps, beta_start=beta_start, beta_end=beta_end)
        elif self == BetaScheduler.SIGMOID:
            return sigmoid_schedule(time_steps, beta_start=beta_start, beta_end=beta_end)
        elif self == BetaScheduler.REVERSE_SIGMOID:
            return reverse_sigmoid_schedle(time_steps, beta_start=beta_start, beta_end=beta_end)
        elif self == BetaScheduler.QUADRATIC:
            return quadratic_schedule(time_steps, beta_start=beta_start, beta_end=beta_end)
        else:
            raise NotImplementedError(f"Schedule '{self.name}' does not support beta range.")


def harmonic_schedule(time_steps: int, /, c_0: float = 0.00002, c_1: float = 12.8, steps_divide: int = 800) -> BetaSpace:
    """
    Harmonic schedule

    - Parameters:
        - time_steps: An `int` of given total steps
        - c_0: A `float` of c_0
        - c_1: A `float` of c_1
        - steps_divide: An `int` of steps where harmonic is used instead of linear
    - Returns: A `.data.BetaSpace` of scheduled beta space
    """
    # calculate betas
    betas = torch.zeros(time_steps)
    betas = torch.tensor([c_0 * (i + 1) if i < steps_divide else (c_1) / (i + 1) for i in range(time_steps)])
    betas[:5] = 0.0001
    return BetaSpace(betas.clip(min=0, max=1))


def constant_schedule(time_steps: int, /, beta: float = 0.015) -> BetaSpace:
    return BetaSpace(torch.zeros(time_steps) + beta)


def cosine_schedule(time_steps: int, /, s: float = 0.008) -> BetaSpace:
    """
    cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    steps = time_steps + 1
    x = torch.linspace(0, time_steps, steps)
    alphas_cumprod = torch.cos(((x / time_steps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return BetaSpace(betas.clip(min=0, max=0.9999))


def linear_schedule(time_steps: int, /, beta_start: float = 0.0001, beta_end: float = 0.01) -> BetaSpace:
    return BetaSpace(torch.linspace(beta_start, beta_end, time_steps))


def peak_linear_schedule(time_steps: int, /, beta_lower: float = 0.001, beta_peak: float = 0.021) -> BetaSpace:
    betas: list[float] = []
    for i in range(time_steps):
        # calculate rate
        rate = (beta_peak - beta_lower) / int(0.3 * time_steps)

        # switch steps
        if i < 0.3 * time_steps:
            betas.append(beta_lower + (i - 1) * rate)
        elif i > 0.7 * time_steps:
            betas.append(beta_peak - (i - 700) * rate)
        else:
            betas.append(beta_peak)
    return BetaSpace(torch.tensor(betas))


def quadratic_schedule(time_steps: int, /, beta_start: float = 0.0001, beta_end: float = 0.02) -> BetaSpace:
    return BetaSpace(torch.linspace(beta_start ** 0.5, beta_end ** 0.5, time_steps) ** 2)


def reverse_sigmoid_schedle(time_steps: int, /, beta_start: float = 1e-4, beta_end: float = 0.02) -> BetaSpace:
    sigmoid_betas = sigmoid_schedule(time_steps, beta_start=beta_start, beta_end=beta_end)
    linear_betas = linear_schedule(time_steps, beta_start=float(sigmoid_betas.betas[0]), beta_end=float(sigmoid_betas.betas[-1]))
    return BetaSpace(linear_betas.betas * 2 - sigmoid_betas.betas)


def sigmoid_schedule(time_steps: int, /, beta_start: float = 0.0001, beta_end: float = 0.02) -> BetaSpace:
    betas = torch.linspace(-3, 3, time_steps)
    betas = betas.sigmoid()
    betas = (betas - betas.min()) / (betas.max() - betas.min())
    return BetaSpace(betas * (beta_end - beta_start) + beta_start)


def mid_linear_schedule(time_steps: int, /, beta_start: float = 0.001, beta_end: float = 0.03) -> BetaSpace:
    betas: list[float] = []
    for i in range(time_steps):
        if i <= 0.3 * time_steps:
            betas.append(beta_start)
        elif i > 0.7 * time_steps:
            betas.append(beta_end)
        else:
            betas.append(beta_start + (i - 0.3 * time_steps) * (beta_end - beta_start) / (0.4 * time_steps))
    return BetaSpace(torch.tensor(betas))

def step_schedule(time_steps: int, /, high: float = 0.001, low: float = 0.0375) -> BetaSpace:
    betas = [high if t <= 0.3 * time_steps or t > 0.7 * time_steps else low for t in range(time_steps)]
    return BetaSpace(torch.tensor(betas))
