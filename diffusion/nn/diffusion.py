from torchmanager_core import abc, torch
from torchmanager_core.typing import Any

from .protocols import TimedData


class DiffusionModule(torch.nn.Module, abc.ABC):
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
        Method to unpack `TimedData`

        - Parameters:
            x_in: The `TimedData` to unpack
        - Returns: A `tuple` of returned unpacked data
        """
        return NotImplemented


class ConditionalDiffusionModule(DiffusionModule):
    """
    The basic diffusion model

    * extends: `diffusion.DiffusionModule`
    """
    def unpack_data(self, x_in: TimedData) -> tuple[torch.Tensor, ...]:
        assert x_in.condition is not None, "The condition must be given."
        return x_in.x, x_in.t, x_in.condition
