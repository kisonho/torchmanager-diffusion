import abc, torch
from typing import Any, Generic, Optional, TypeVar, Union, overload

from diffusion.data import DiffusionData
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


Module = TypeVar('Module', bound=TimedModule)


class DiffusionModule(torch.nn.Module, Generic[Module], abc.ABC):
    """
    The diffusion model that has the forward diffusion and sampling step algorithm implemented

    * extends: `torch.nn.Module`
    * Abstract class
    * Generic: `Module`

    - Properties:
        - model: The model to use for diffusion in `Module`
    - method to implement:
        - forward_diffusion: The forward pass of diffusion model, sample noises
        - sampling_step: The sampling step of diffusion model
    """
    model: Module
    time_steps: int

    @property
    def sampling_range(self) -> range:
        return range(1, self.time_steps + 1)

    def __init__(self, model: Module, time_steps: int) -> None:
        super().__init__()
        self.model = model
        self.time_steps = time_steps

    def __call__(self, x_in: TimedData, sampling: bool = False) -> Any:
        if sampling:
            # initialize
            x_t = x_in.x

            # sampling loop time step
            for i in reversed(self.sampling_range):
                # fetch data
                t = torch.full((x_in.x.shape[0],), i, dtype=torch.long, device=x_t.device)

                # append to predicitions
                x = DiffusionData(x_t, t, condition=x_in.condition)
                y = self.sampling_step(x, i)
                x_t = y
        else:
            return super().__call__(x_in)

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
