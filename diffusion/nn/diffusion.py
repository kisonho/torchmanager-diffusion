from torchmanager_core import abc, torch
from torchmanager_core.typing import Any

from .protocols import TimedData


class DiffusionModule(torch.nn.Module, abc.ABC):
    """
    The basic diffusion model

    * extends: `torch.nn.Module`
    * Abstract class

    - method to implement:
        - forward: The forward method that accepts inputs perform to `.protocols.TimedData`
    """

    def __call__(self, x_in: TimedData, *args: Any, **kwargs: Any) -> torch.Tensor:
        return super().__call__(x_in, *args, **kwargs)

    @abc.abstractmethod
    def forward(self, x_in: TimedData, *args: Any, **kwargs: Any) -> torch.Tensor:
        return NotImplemented
