import torch
from typing import Protocol

from diffusion.data import DiffusionData
from diffusion.scheduling import BetaSpace
from diffusion.sde import SDE, SubVPSDE, VESDE, VPSDE


class TimedData(Protocol):
    @property
    def x(self) -> torch.Tensor:
        return NotImplemented

    @property
    def t(self) -> torch.Tensor:
        return NotImplemented

    @property
    def condition(self) -> torch.Tensor | None:
        return NotImplemented
