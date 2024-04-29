import abc, torch
from enum import Enum
from torchmanager_core import view
from typing import Any, Generic, Optional, Sequence, TypeVar, Union, overload

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


Module = TypeVar('Module', bound=torch.nn.Module)


class DiffusionMode(Enum):
    """
    The diffusion mode to use

    * extends: `Enum`
    """
    FORWARD_DATA = 'forward_data'
    FORWARD_DIFFUSION = 'forward_diffusion'
    SAMPLING = 'sampling'


class DiffusionModule(torch.nn.Module, Generic[Module], abc.ABC):
    """
    The diffusion model that has the forward diffusion and sampling step algorithm implemented

    * extends: `torch.nn.Module`
    * Abstract class
    * Generic: `Module`

    - Properties:
        - model: The model to use for diffusion in `Module`
        - time_steps: The total time steps of diffusion model
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

    def __call__(self, *args: Any, mode: DiffusionMode = DiffusionMode.FORWARD_DATA, **kwds: Any) -> Any:
        return super().__call__(*args, mode=mode, **kwds)

    def forward(self, *args: Any, mode: DiffusionMode = DiffusionMode.FORWARD_DATA, **kwargs: Any) -> Any:
        if mode == DiffusionMode.FORWARD_DATA:
            return self.forward_data(*args, **kwargs)
        elif mode == DiffusionMode.FORWARD_DIFFUSION:
            return self.forward_diffusion(*args, **kwargs)
        elif mode == DiffusionMode.SAMPLING:
            return self.sampling_step(*args, **kwargs)
        else:
            raise NotImplementedError(f"Mode {mode} is not implemented for diffusion model")

    def forward_data(self, data: DiffusionData, /) -> Any:
        if isinstance(self.model, TimedModule):
            return self.model(data)
        elif data.condition is not None:
            return self.model(data.x, data.t, condition=data.condition)
        else:
            return self.model(data.x, data.t)

    @abc.abstractmethod
    def forward_diffusion(self, data: Any, /, condition: Optional[torch.Tensor] = None, t: Optional[torch.Tensor] = None) -> tuple[Any, torch.Tensor]:
        """
        Forward pass of diffusion model, sample noises

        - Parameters:
            - data: Any kind of noised data
            - t: An optional `torch.Tensor` of the time step, sampling uniformly if not given
            - condition: An optional `torch.Tensor` of the condition to generate images
        - Returns: A `tuple` of noisy images and sampled time step in `DiffusionData` and noises in `torch.Tensor`
        """
        return NotImplemented

    @torch.no_grad()
    def sampling(self, num_images: int, x_t: torch.Tensor, *args: Any, condition: Optional[torch.Tensor] = None, sampling_range: Optional[Union[Sequence[int], range]] = None, show_verbose: bool = False, **kwargs: Any) -> list[torch.Tensor]:
        '''
        Samples a given number of images

        - Parameters:
            - num_images: An `int` of number of images to generate
            - x_t: A `torch.Tensor` of the image at T step
            - condition: An optional `torch.Tensor` of the condition to generate images
            - sampling_range: An optional `Sequence[int]`, or `range` of the range of time steps to sample
            - start_index: An optional `int` of the start index of the time step
            - end_index: An `int` of the end index of the time step
            - show_verbose: A `bool` flag to show the progress bar during testing
        - Retruns: A `list` of `torch.Tensor` generated results
        '''
        # initialize
        imgs = x_t
        sampling_range = range(self.time_steps, 0, -1) if sampling_range is None else sampling_range
        progress_bar = view.tqdm(desc='Sampling loop time step', total=len(sampling_range), disable=not show_verbose)

        # sampling loop time step
        for i in sampling_range:
            # fetch data
            t = torch.full((num_images,), i, dtype=torch.long, device=imgs.device)

            # append to predicitions
            x = DiffusionData(imgs, t, condition=condition)
            y = self(x, i, mode=DiffusionMode.SAMPLING)
            imgs = y.to(imgs.device)
            progress_bar.update()

        # reset model and loss
        return [img for img in imgs]

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
